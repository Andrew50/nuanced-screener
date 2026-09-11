from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo


EXCHANGE_TIMEZONES = {
    "NYSE": "America/New_York",
    "NASDAQ": "America/New_York",
    "XNYS": "America/New_York",
    "XNAS": "America/New_York",
}


@dataclass(frozen=True)
class TradingCalendar:
    name: str = "NYSE"

    def valid_trading_days(self, start: date, end: date) -> list[date]:
        """
        Return trading days inclusive in [start, end] using NYSE calendar.
        """
        # Prefer `exchange_calendars` when available:
        # - avoids some pandas_market_calendars / pandas version edge cases
        # - tends to be more stable and faster for simple session queries
        try:
            import exchange_calendars as xcals  # type: ignore
            import pandas as pd  # type: ignore

            code_map = {"NYSE": "XNYS", "NASDAQ": "XNAS"}
            code = code_map.get(str(self.name).upper(), str(self.name))
            cal = xcals.get_calendar(code)
            sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
            # sessions is a pandas.DatetimeIndex (naive, UTC-midnight-ish). Convert to python dates.
            return [d.date() for d in sessions.to_pydatetime()]
        except Exception:
            pass

        # Fallback: pandas_market_calendars
        try:
            import pandas_market_calendars as mcal  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "A market calendar backend is required. Install with:\n"
                "- `pip install exchange_calendars` (preferred)\n"
                "- or `pip install pandas_market_calendars`\n"
            ) from e

        cal = mcal.get_calendar(self.name)
        days = cal.valid_days(start_date=start.isoformat(), end_date=end.isoformat())
        return [d.date() for d in days.to_pydatetime()]


def subtract_years(d: date, years: int) -> date:
    years = int(years)
    if years <= 0:
        return d
    try:
        return date(d.year - years, d.month, d.day)
    except ValueError:
        # Handle Feb 29th etc by clamping to Feb 28.
        if d.month == 2 and d.day == 29:
            return date(d.year - years, 2, 28)
        # Fallback to a close approximation.
        return d - timedelta(days=365 * years)


def calendar_local_date(now: datetime, calendar_name: str = "NYSE") -> date:
    """Exchange-local calendar date. Do not use UTC ``date()`` — that rolls at 20:00 ET."""

    tz_name = EXCHANGE_TIMEZONES.get(str(calendar_name).upper(), "America/New_York")
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ZoneInfo(tz_name)).date()


def latest_trading_day_on_or_before(today: date, cal: TradingCalendar) -> date:
    # Find the latest trading day <= today using a small lookback window.
    # We use a 10-day range to cover weekends + holidays.
    start = today - timedelta(days=10)
    days = cal.valid_trading_days(start, today)
    if not days:
        raise RuntimeError("Could not determine latest trading day (calendar returned empty)")
    return days[-1]


def latest_authorized_daily_session(
    today: date,
    cal: TradingCalendar,
    *,
    skip_same_calendar_day: bool = True,
) -> date:
    """Latest session Polygon grouped-daily will authorize.

    Delayed plans reject same-calendar-day aggregates ("today's data before
    end of day"), so when ``today`` is a trading day we use the previous session.
    """
    end = latest_trading_day_on_or_before(today, cal)
    if skip_same_calendar_day and end == today:
        window = cal.valid_trading_days(today - timedelta(days=10), today)
        if len(window) >= 2:
            return window[-2]
    return end


def last_n_trading_days_ending_at(day: date, n: int, cal: TradingCalendar) -> list[date]:
    if n <= 0:
        return []
    # Search enough window to include n trading days.
    start = day - timedelta(days=30 + 3 * n)
    days = cal.valid_trading_days(start, day)
    if not days:
        return []
    return days[-n:]


def next_n_trading_days_starting_after(day: date, n: int, cal: TradingCalendar) -> list[date]:
    """
    Return the next n trading days strictly after `day`, in chronological order.
    """
    if n <= 0:
        return []
    # Search enough window to include n trading days after `day`.
    start = day + timedelta(days=1)
    end = day + timedelta(days=30 + 3 * n)
    days = cal.valid_trading_days(start, end)
    if not days:
        return []
    return days[:n]

