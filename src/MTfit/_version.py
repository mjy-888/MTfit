"""
Version information for MTfit.

Uses importlib.metadata when installed as a package,
falls back to a default version string for development.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("MTfit")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


def get_versions() -> dict:
    """Return version dict for backward compatibility."""
    return {"version": __version__}
