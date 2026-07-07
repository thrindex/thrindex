"""thrindex — the neuromorphic infrastructure SDK.

Import conventions (canonical, per Playbook §28):
    import thrindex as thx
    import thrindex.snn as snn

Version is sourced exclusively from ``pyproject.toml [project] version`` and
read at runtime via :mod:`importlib.metadata`.  Do **not** hardcode it here.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from thrindex import encoders, snn, train

try:
    __version__: str = _pkg_version("thrindex")
except PackageNotFoundError:
    __version__ = "0+uninstalled"

__all__ = ["__version__", "encoders", "snn", "train"]
