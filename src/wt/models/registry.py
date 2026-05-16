"""Model save/load helpers and versioning."""

from __future__ import annotations

import json
import pickle
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from wt.models.train import TrainedModel
from wt.utils.paths import MODELS_DIR



def current_git_sha() -> str:
    try:
        completed = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return 'nogit'



def model_version_tag(now: datetime | None = None) -> str:
    ts = now or datetime.now(tz=UTC)
    return f"{ts.strftime('%Y%m%d')}-{current_git_sha()}"



def save_trained_model(model: TrainedModel, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / f"model_{model.station}_{model.target}.pkl"
    with model_path.open('wb') as handle:
        pickle.dump(model, handle)
    metrics_path = destination / f"metrics_{model.station}_{model.target}.json"
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(model.metrics, handle, indent=2, sort_keys=True)
    return model_path



def load_trained_model(path: str | Path) -> TrainedModel:
    with Path(path).open('rb') as handle:
        return pickle.load(handle)



def default_model_dir(version: str | None = None) -> Path:
    return MODELS_DIR / (version or model_version_tag())
