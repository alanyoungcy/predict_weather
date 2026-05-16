"""Apply local retention policies."""

from __future__ import annotations

from pathlib import Path
import sys

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from wt.ops.housekeeping import apply_housekeeping_plan, build_housekeeping_plan


@click.command()
@click.option("--apply", "apply_changes", is_flag=True, default=False, help="Delete files instead of dry-run only.")
def main(apply_changes: bool) -> None:
    actions = build_housekeeping_plan()
    if not actions:
        print("No housekeeping actions planned.")
        return

    for action in actions:
        print(f"{action.action}\t{action.path}\t{action.reason}")

    apply_housekeeping_plan(actions, dry_run=not apply_changes)
    print(f"\nPlanned {len(actions)} action(s). apply={apply_changes}")


if __name__ == "__main__":
    main()
