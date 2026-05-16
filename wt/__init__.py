"""Root package shim so `python -m wt...` works without installation."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_src_pkg = Path(__file__).resolve().parent.parent / "src" / "wt"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))

__all__ = ["__version__"]
__version__ = "0.1.0"
