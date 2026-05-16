"""Retention and local artifact housekeeping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from wt.config import RetentionConfig, StorageConfig, load_storage_config, load_settings


@dataclass(frozen=True, slots=True)
class HousekeepingAction:
    path: Path
    action: str
    reason: str


def _iter_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _is_older_than(path: Path, cutoff: datetime) -> bool:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return modified < cutoff


def plan_file_prune(root: Path, keep_days: int, now: datetime | None = None) -> list[HousekeepingAction]:
    if keep_days <= 0:
        return []
    current = now or datetime.now(tz=UTC)
    cutoff = current - timedelta(days=keep_days)
    actions: list[HousekeepingAction] = []
    for path in _iter_files(root):
        if _is_older_than(path, cutoff):
            actions.append(HousekeepingAction(path=path, action="delete_file", reason=f"older than {keep_days} days"))
    return actions


def plan_model_version_prune(models_dir: Path, keep_latest: int) -> list[HousekeepingAction]:
    if keep_latest <= 0 or not models_dir.exists():
        return []
    version_dirs = sorted(
        [path for path in models_dir.iterdir() if path.is_dir() and path.name.startswith("v")],
        key=lambda item: item.name,
        reverse=True,
    )
    actions: list[HousekeepingAction] = []
    for path in version_dirs[keep_latest:]:
        actions.append(HousekeepingAction(path=path, action="delete_tree", reason=f"keep latest {keep_latest} model versions"))
    return actions


def build_housekeeping_plan(
    storage: StorageConfig | None = None,
    now: datetime | None = None,
) -> list[HousekeepingAction]:
    settings = load_settings()
    storage_cfg = storage or load_storage_config()
    retention: RetentionConfig = storage_cfg.retention

    actions: list[HousekeepingAction] = []
    actions.extend(plan_file_prune(settings.data_dir / "raw", retention.local_raw_days, now=now))
    actions.extend(plan_file_prune(settings.data_dir / "interim", retention.local_interim_days, now=now))
    actions.extend(plan_file_prune(settings.data_dir / "features", retention.local_feature_days, now=now))
    actions.extend(plan_file_prune(settings.data_dir / "predictions", retention.local_prediction_days, now=now))
    actions.extend(plan_file_prune(settings.data_dir / "signals", retention.local_signal_days, now=now))
    actions.extend(plan_file_prune(settings.data_dir / "logs", retention.local_log_days, now=now))
    actions.extend(plan_model_version_prune(settings.models_dir, retention.keep_model_versions))
    return actions


def apply_housekeeping_plan(actions: list[HousekeepingAction], dry_run: bool = True) -> list[HousekeepingAction]:
    if dry_run:
        return actions

    for action in actions:
        if action.action == "delete_file":
            action.path.unlink(missing_ok=True)
        elif action.action == "delete_tree":
            for child in sorted(action.path.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            action.path.rmdir()
        else:
            raise ValueError(f"Unsupported action: {action.action}")
    return actions
