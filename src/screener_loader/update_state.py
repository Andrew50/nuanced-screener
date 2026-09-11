from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import LoaderConfig
from .paths import DataPaths, atomic_replace, ensure_dirs

SCHEMA_VERSION = 1
DEFAULT_STALE_AFTER = timedelta(hours=24)


@dataclass(frozen=True)
class UpdateState:
    finished_at: datetime
    started_at: datetime | None
    vendor: str
    ok: bool
    newest_partition: date | None
    dates_updated: tuple[str, ...]
    dates_failed: tuple[str, ...]
    dates_no_data: tuple[str, ...]
    path: Path


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _parse_iso_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return _as_utc(parsed)


def _parse_iso_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text[:10])


def load_update_state(paths: DataPaths) -> UpdateState | None:
    path = paths.update_state_json
    if not path.exists():
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    finished_at = _parse_iso_datetime(payload.get("finished_at"))
    if finished_at is None:
        return None

    def _str_tuple(key: str) -> tuple[str, ...]:
        raw = payload.get(key) or []
        if not isinstance(raw, list):
            return ()
        return tuple(str(x) for x in raw)

    return UpdateState(
        finished_at=finished_at,
        started_at=_parse_iso_datetime(payload.get("started_at")),
        vendor=str(payload.get("vendor") or ""),
        ok=bool(payload.get("ok", True)),
        newest_partition=_parse_iso_date(payload.get("newest_partition")),
        dates_updated=_str_tuple("dates_updated"),
        dates_failed=_str_tuple("dates_failed"),
        dates_no_data=_str_tuple("dates_no_data"),
        path=path,
    )


def write_update_state(
    paths: DataPaths,
    *,
    vendor: str,
    started_at: datetime | None = None,
    newest_partition: date | None = None,
    dates_updated: list[str] | tuple[str, ...] | None = None,
    dates_failed: list[str] | tuple[str, ...] | None = None,
    dates_no_data: list[str] | tuple[str, ...] | None = None,
    ok: bool = True,
    finished_at: datetime | None = None,
) -> UpdateState:
    import json

    ensure_dirs(paths)
    done = _as_utc(finished_at or datetime.now(timezone.utc))
    started = _as_utc(started_at) if started_at is not None else None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started.isoformat() if started else None,
        "finished_at": done.isoformat(),
        "vendor": str(vendor),
        "ok": bool(ok),
        "newest_partition": newest_partition.isoformat() if newest_partition else None,
        "dates_updated": [str(x) for x in (dates_updated or ())],
        "dates_failed": [str(x) for x in (dates_failed or ())],
        "dates_no_data": [str(x) for x in (dates_no_data or ())],
    }
    out = paths.update_state_json
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    atomic_replace(tmp, out)
    return load_update_state(paths) or UpdateState(
        finished_at=done,
        started_at=started,
        vendor=str(vendor),
        ok=bool(ok),
        newest_partition=newest_partition,
        dates_updated=tuple(str(x) for x in (dates_updated or ())),
        dates_failed=tuple(str(x) for x in (dates_failed or ())),
        dates_no_data=tuple(str(x) for x in (dates_no_data or ())),
        path=out,
    )


def is_update_fresh(
    state: UpdateState | None,
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_STALE_AFTER,
) -> bool:
    if state is None or not state.ok:
        return False
    age = _as_utc(now or datetime.now(timezone.utc)) - state.finished_at
    return age <= max_age


def ensure_fresh_market_data(
    config: LoaderConfig,
    *,
    updater: Callable[[LoaderConfig], None],
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_STALE_AFTER,
) -> bool:
    """
    Run ``updater`` if the last successful stamp is missing or older than ``max_age``.

    Returns True if an update was triggered.
    """
    state = load_update_state(config.paths)
    if is_update_fresh(state, now=now, max_age=max_age):
        return False
    updater(config)
    return True
