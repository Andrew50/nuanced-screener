from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from screener_loader.cli import app


def test_base_help_lists_vision_without_optional_extras() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0, res.output
    assert "vision" in res.output
    assert "setups" in res.output
    assert "screen" in res.output
    assert "query" in res.output
    vision = runner.invoke(app, ["vision", "--help"])
    assert vision.exit_code == 0, vision.output
    for name in ("scan", "resume", "runs", "export", "view"):
        assert name in vision.output
    scan = runner.invoke(app, ["vision", "scan", "--help"])
    assert scan.exit_code == 0, scan.output
    assert "--dry-run" in scan.output
    assert "--demo" in scan.output
    assert "--max-candidates" in scan.output
    assert "--skip-update" in scan.output
    screen = runner.invoke(app, ["screen", "--help"])
    assert screen.exit_code == 0, screen.output
    assert "--dry-run" in screen.output
    assert "--skip-update" in screen.output
    query = runner.invoke(app, ["query", "--help"])
    assert query.exit_code == 0, query.output
    assert "--query" in query.output
    vision = runner.invoke(app, ["vision", "--help"])
    assert vision.exit_code == 0, vision.output
    for name in ("scan", "resume", "runs", "export", "view"):
        assert name in vision.output
    scan = runner.invoke(app, ["vision", "scan", "--help"])
    assert scan.exit_code == 0, scan.output
    assert "--dry-run" in scan.output
    assert "--demo" in scan.output
    assert "--max-candidates" in scan.output


def test_vision_scan_live_requires_model(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["vision", "scan", "--repo-root", str(tmp_path)])
    assert res.exit_code != 0
    assert "model" in res.output.lower() or "NS_VISION_MODEL" in res.output


def test_screen_live_requires_model_and_skips_update(tmp_path: Path, monkeypatch) -> None:
    called: list[int] = []
    monkeypatch.setattr("screener_loader.cli.vision.update_market_data", lambda cfg: called.append(1))
    runner = CliRunner()
    res = runner.invoke(app, ["screen", "--repo-root", str(tmp_path)])
    assert res.exit_code != 0
    assert "model" in res.output.lower() or "NS_VISION_MODEL" in res.output
    assert called == []
