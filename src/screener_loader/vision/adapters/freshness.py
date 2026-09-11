"""Latest-daily freshness: calendar close times vs actual last-N / window dates."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable

from ..types import StaleSnapshotError, VisionError

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def coerce_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        text = str(value)[:10]
        return date.fromisoformat(text)


def expected_completed_session(now: datetime, *, calendar_name: str = "NYSE") -> date:
    """Latest exchange session whose close (including early close) is at or before ``now``."""

    now_utc = as_utc(now)
    start = now_utc.date() - timedelta(days=14)
    end = now_utc.date() + timedelta(days=1)
    completed = _completed_sessions_exchange_calendars(now_utc, start, end, calendar_name)
    if completed is None:
        completed = _completed_sessions_pandas_mcal(now_utc, start, end, calendar_name)
    if not completed:
        raise StaleSnapshotError(
            f"No completed {calendar_name} session at or before {now_utc.isoformat()}"
        )
    return completed[-1]


def authorized_daily_session(now: datetime, *, calendar_name: str = "NYSE") -> date:
    """Latest session grouped-daily vendors will serve (never same-calendar-day)."""

    from ...calendar_utils import TradingCalendar, calendar_local_date, latest_authorized_daily_session

    now_utc = as_utc(now)
    cal = TradingCalendar(calendar_name)
    today = calendar_local_date(now_utc, calendar_name)
    return latest_authorized_daily_session(today, cal, skip_same_calendar_day=True)


def _close_utc(value: object) -> datetime:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _session_date(value: object) -> date:
    coerced = coerce_date(value)
    if coerced is None:
        raise VisionError(f"Calendar session has no date: {value!r}")
    return coerced


def _completed_sessions_exchange_calendars(
    now_utc: datetime,
    start: date,
    end: date,
    calendar_name: str,
) -> list[date] | None:
    try:
        import exchange_calendars as xcals
        import pandas as pd
    except Exception:
        return None
    try:
        code_map = {"NYSE": "XNYS", "NASDAQ": "XNAS"}
        code = code_map.get(str(calendar_name).upper(), str(calendar_name))
        cal = xcals.get_calendar(code)
        sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        out: list[date] = []
        for session in sessions:
            close = cal.session_close(session)
            if _close_utc(close) <= now_utc:
                out.append(_session_date(session))
        return out
    except Exception:
        return None


def _completed_sessions_pandas_mcal(
    now_utc: datetime,
    start: date,
    end: date,
    calendar_name: str,
) -> list[date]:
    try:
        import pandas_market_calendars as mcal
    except Exception as exc:  # pragma: no cover
        raise VisionError(
            "A market calendar backend is required for vision freshness "
            "(pandas_market_calendars or exchange_calendars)."
        ) from exc
    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    out: list[date] = []
    for session, row in schedule.iterrows():
        if _close_utc(row["market_close"]) <= now_utc:
            out.append(_session_date(session))
    return out


def assert_dataset_freshness(
    *,
    max_row_date: date | None,
    expected: date,
) -> None:
    """Fail closed on a globally stale or future last-N snapshot.

    ``EligibilityResult.asof_date`` is a maximum over eligible rows and is not used here.
    """

    resolve_scan_session(mode="live", expected=expected, max_row_date=max_row_date)


def resolve_scan_session(
    *,
    mode: str,
    expected: date,
    max_row_date: date | None,
    authorized: date | None = None,
) -> date:
    """Session the scan should chart through.

    ``expected`` is the latest session that has closed. ``authorized`` is the
    latest session delayed grouped-daily vendors will serve (previous session
    while ``expected`` is still calendar-today). Live accepts last-N dates in
    ``[authorized, expected]``. Dry-run and demo also allow an older snapshot.
    """

    floor = authorized if authorized is not None else expected
    if max_row_date is None:
        raise StaleSnapshotError(
            f"Derived last-N snapshot has no dates; expected completed session {expected.isoformat()}"
        )
    if max_row_date > expected:
        raise StaleSnapshotError(
            f"Derived last-N snapshot contains future session {max_row_date.isoformat()} "
            f"(expected completed session {expected.isoformat()})"
        )
    if max_row_date < floor:
        if str(mode) in {"dry_run", "demo"}:
            return max_row_date
        raise StaleSnapshotError(
            f"Derived last-N snapshot is stale: latest session {max_row_date.isoformat()} "
            f"but expected completed session {expected.isoformat()} "
            f"(authorized daily {floor.isoformat()})"
        )
    return max_row_date
