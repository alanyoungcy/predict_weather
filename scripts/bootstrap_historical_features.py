"""Build historical training features from labels and archived NWP forecasts."""

from __future__ import annotations

from pathlib import Path
import sys

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from wt.orchestration.backfill import backfill_training_features


@click.command()
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--station", "station_filter", default="all", show_default=True)
@click.option("--labels", "labels_path", type=click.Path(path_type=Path), default=None)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--models",
    default=None,
    help="Comma-separated models; default uses config/models.yaml.",
)
@click.option("--init-hour-utc", default=18, show_default=True, type=int)
def main(
    start,
    end,
    station_filter: str,
    labels_path: Path | None,
    output_path: Path | None,
    models: str | None,
    init_hour_utc: int,
) -> None:
    model_list = [part.strip() for part in models.split(",") if part.strip()] if models else None
    features = backfill_training_features(
        start_date=start.date(),
        end_date=end.date(),
        station_filter=station_filter,
        labels_path=labels_path,
        output_path=output_path,
        models=model_list,
        init_hour_utc=init_hour_utc,
    )
    print(f"Wrote {len(features):,} feature rows")


if __name__ == "__main__":
    main()
