"""Check configured storage backends."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from wt.storage.motherduck import check_motherduck_connection
from wt.storage.mongodb import check_mongodb_connection
from wt.storage.supabase import check_supabase_connection


@click.command()
def main() -> None:
    checks = {
        "supabase": check_supabase_connection(),
        "mongodb": check_mongodb_connection(),
        "motherduck": check_motherduck_connection(),
    }
    for name, status in checks.items():
        detail = asdict(status)
        print(f"{name}\t{detail}")


if __name__ == "__main__":
    main()
