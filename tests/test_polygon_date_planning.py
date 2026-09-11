from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from screener_loader.config import LoaderConfig
from screener_loader.update import _plan_polygon_dates


@dataclass(frozen=True)
class _FakeCal:
    trading_days: list[date]

    def valid_trading_days(self, start: date, end: date) -> list[date]:
        return [d for d in self.trading_days if start <= d <= end]


def _ten_day_cal() -> _FakeCal:
    start = date(2026, 1, 1)
    return _FakeCal(trading_days=[start + timedelta(days=i) for i in range(10)])


def test_plan_polygon_dates_orders_latest_then_missing_then_existing_tail() -> None:
    cal = _ten_day_cal()
    cfg = LoaderConfig(lookback_years=1, refresh_tail_days=3, calls_per_minute=5)

    existing = {date(2026, 1, 2), date(2026, 1, 4), date(2026, 1, 9), date(2026, 1, 10)}
    planned = _plan_polygon_dates(cfg, cal=cal, today=date(2026, 1, 10), existing_partitions=existing, now_utc=None)

    assert planned[0] == date(2026, 1, 10)

    expected_missing = [date(2026, 1, d) for d in [8, 7, 6, 5, 3, 1]]
    assert planned[1 : 1 + len(expected_missing)] == expected_missing

    # Tail is the 3 newest existing days (latest is already planned, so 1/2 is dropped).
    assert planned[1 + len(expected_missing) :] == [date(2026, 1, 9), date(2026, 1, 4)]


def test_plan_polygon_dates_full_refresh_replays_all_existing() -> None:
    cal = _ten_day_cal()
    cfg = LoaderConfig(lookback_years=1, refresh_tail_days=3, full_refresh=True, calls_per_minute=5)

    existing = {date(2026, 1, 2), date(2026, 1, 4), date(2026, 1, 9), date(2026, 1, 10)}
    planned = _plan_polygon_dates(cfg, cal=cal, today=date(2026, 1, 10), existing_partitions=existing, now_utc=None)

    expected_missing = [date(2026, 1, d) for d in [8, 7, 6, 5, 3, 1]]
    assert planned[0] == date(2026, 1, 10)
    assert planned[1 : 1 + len(expected_missing)] == expected_missing
    assert planned[1 + len(expected_missing) :] == [date(2026, 1, 9), date(2026, 1, 4), date(2026, 1, 2)]


def test_plan_polygon_dates_daily_update_is_latest_plus_missing_plus_tail() -> None:
    cal = _ten_day_cal()
    cfg = LoaderConfig(lookback_years=1, refresh_tail_days=3, calls_per_minute=5)

    all_days = set(cal.trading_days)
    latest = date(2026, 1, 10)
    existing = all_days - {latest}
    planned = _plan_polygon_dates(cfg, cal=cal, today=latest, existing_partitions=existing, now_utc=None)

    assert planned[0] == latest
    assert latest not in existing
    # Only the new latest day is missing; then the 3 newest existing days.
    assert planned == [
        latest,
        date(2026, 1, 9),
        date(2026, 1, 8),
        date(2026, 1, 7),
    ]


def test_plan_polygon_dates_skips_same_calendar_day_after_close() -> None:
    cal = _ten_day_cal()
    cfg = LoaderConfig(lookback_years=1, refresh_tail_days=3, calls_per_minute=5)
    today = date(2026, 1, 10)
    now_utc = datetime(2026, 1, 11, 2, 0, tzinfo=timezone.utc)  # 21:00 ET on today
    planned = _plan_polygon_dates(
        cfg, cal=cal, today=today, existing_partitions=set(), now_utc=now_utc
    )
    assert date(2026, 1, 10) not in planned
    assert planned[0] == date(2026, 1, 9)
