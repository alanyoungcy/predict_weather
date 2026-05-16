from datetime import UTC, datetime, timedelta
from pathlib import Path
import os

from wt.ops.housekeeping import plan_file_prune, plan_model_version_prune


def test_plan_file_prune_selects_only_old_files(tmp_path: Path) -> None:
    old_file = tmp_path / "old.txt"
    new_file = tmp_path / "new.txt"
    old_file.write_text("x", encoding="utf-8")
    new_file.write_text("y", encoding="utf-8")

    old_time = (datetime.now(tz=UTC) - timedelta(days=10)).timestamp()
    new_time = (datetime.now(tz=UTC) - timedelta(days=1)).timestamp()
    os.utime(old_file, (old_time, old_time))
    os.utime(new_file, (new_time, new_time))

    actions = plan_file_prune(tmp_path, keep_days=5, now=datetime.now(tz=UTC))
    assert [action.path.name for action in actions] == ["old.txt"]


def test_plan_model_version_prune_keeps_latest_versions(tmp_path: Path) -> None:
    for name in ["v20260101", "v20260102", "v20260103"]:
        (tmp_path / name).mkdir()

    actions = plan_model_version_prune(tmp_path, keep_latest=2)
    assert [action.path.name for action in actions] == ["v20260101"]
