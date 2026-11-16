"""Repository-relative path helpers so every module works from any working directory."""
import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("WINDLIDAR_ROOT")
    if env:
        return Path(env).resolve()
    for p in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _find_root()


def rp(*parts) -> str:
    """Absolute path below the repository root, e.g. rp("results", "fesstval")."""
    return str(ROOT.joinpath(*parts))
