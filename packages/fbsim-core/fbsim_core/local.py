"""Per-checkout locations: <repo>/data/<package> for output, <repo>/credentials for keys.

Both are normally symlinks to directories outside the checkout (see scripts/setup_worktree.sh).
"""

import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """The checkout holding this file (editable install), else the one holding the cwd (wheel install)."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for p in (start, *start.parents):
            if (p / "packages" / "fbsim-core").is_dir() and (p / "worlds").is_dir():
                return p
    print(f"fbsim_core.local: no forecastbench-sim checkout found; using data/ and credentials/ in {Path.cwd()}",
          file=sys.stderr)
    return Path.cwd().resolve()


REPO_ROOT = _find_repo_root()
DATA_ROOT = Path(os.environ.get("FBSIM_TEST_DATA_ROOT", str(REPO_ROOT / "data"))).expanduser().resolve()  # override: tests only
CREDENTIALS_DIR = REPO_ROOT / "credentials"
ENV_FILE = Path(os.environ.get("FBSIM_TEST_ENV_FILE", str(CREDENTIALS_DIR / ".env"))).expanduser()  # override: tests only


def data_dir(package: str) -> Path:
    """data/<package>; not created here."""
    return DATA_ROOT / package


def load_env() -> bool:
    """Load credentials/.env into os.environ without overriding exported variables."""
    from dotenv import load_dotenv

    return load_dotenv(ENV_FILE, override=False)
