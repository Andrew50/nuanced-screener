#!/usr/bin/env bash
# Incremental Polygon OHLCV sync: latest session + any gaps + a short refresh tail.
#
# Example crontab (America/New_York, after the 20:00 ET "data ready" cutoff):
#   5 21 * * 1-5 cd /home/aj/nuanced-screener && TZ=America/New_York ./scripts/sync_daily.sh >> data/meta/logs/sync_daily.log 2>&1
#
# WSL: cron only runs while the distro is up. Enable the cron service or use
# Windows Task Scheduler: wsl.exe -d <distro> -- /home/aj/nuanced-screener/scripts/sync_daily.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

mkdir -p "$ROOT/data/meta/logs"
LOCK="${XDG_RUNTIME_DIR:-/tmp}/ns-update.lock"

if command -v flock >/dev/null 2>&1; then
  exec flock -n "$LOCK" ns update \
    --repo-root "$ROOT" \
    --ohlcv-vendor polygon_grouped \
    --lookback-years 2 \
    --refresh-tail-days 3 \
    --calls-per-minute 5
fi

ns update \
  --repo-root "$ROOT" \
  --ohlcv-vendor polygon_grouped \
  --lookback-years 2 \
  --refresh-tail-days 3 \
  --calls-per-minute 5
