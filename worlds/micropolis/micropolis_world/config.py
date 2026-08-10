"""Config-file loading for the scripts in worlds/micropolis/scripts/.

Every script takes an optional config file path as its first positional argument
and reads all of its non-flag parameters from there; behavior toggles
(--dry-run, --quiet, --plot) stay on the command line. Omitting the path falls
back to configs/default.json5.

A script asks for the keys it needs via Config.get* and errors out if one is
missing, so a single config file can carry the union of every script's
parameters and still be usable by each of them individually.

Configs are parsed as JSON5, so they may contain // and /* */ comments and
trailing commas. Plain JSON is a subset of JSON5 and loads unchanged.
"""

import argparse
import functools
import sys
from pathlib import Path
from typing import Any, Callable

import json5

from . import module_globals as g

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
# .json5 rather than .json so editors don't flag the comments as syntax errors.
# Either extension loads; the parser is the same.
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.json5"


class ConfigError(Exception):
    """A config file is missing, malformed, or missing a key a script needs."""


class Config:
    """Parameters loaded from a JSON5 config file.

    Access is by explicit getter so that a missing key is a hard error naming
    both the key and the file it should have been in, rather than a KeyError or
    a silent default.
    """

    def __init__(self, data: dict[str, Any], path: Path):
        self.data = data
        self.path = path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        try:
            # JSON5, so configs can carry // and /* */ comments and trailing
            # commas. Plain JSON is a subset, so existing configs still load.
            data = json5.loads(path.read_text())
        except ValueError as e:
            raise ConfigError(f"config file {path} is not valid JSON5: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"config file {path} must contain a JSON object")
        return cls(data, path)

    def _require(self, key: str) -> Any:
        if key not in self.data:
            raise ConfigError(f"missing required parameter '{key}' in {self.path}")
        return self.data[key]

    def get_int(self, key: str) -> int:
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be an integer, got {value!r}"
            )
        return value

    def get_bool(self, key: str) -> bool:
        value = self._require(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a boolean, got {value!r}"
            )
        return value

    def get_str(self, key: str) -> str:
        value = self._require(key)
        if not isinstance(value, str):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a string, got {value!r}"
            )
        return value

    def get_int_list(self, key: str) -> list[int]:
        value = self._require(key)
        if (
            not isinstance(value, list)
            or not value
            or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
        ):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a non-empty list of "
                f"integers, got {value!r}"
            )
        return value

    def get_str_list(self, key: str) -> list[str]:
        value = self._require(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(v, str) for v in value)
        ):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a non-empty list of "
                f"strings, got {value!r}"
            )
        return value

    def get_bool_list(self, key: str) -> list[bool]:
        value = self._require(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(v, bool) for v in value)
        ):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a non-empty list of "
                f"booleans, got {value!r}"
            )
        return value

    def get_cities(self) -> list[str]:
        """The 'cities' list, validated against the known Micropolis city names."""
        cities = self.get_str_list("cities")
        unknown = [c for c in cities if c not in g.CITY_CHOICES]
        if unknown:
            raise ConfigError(
                f"parameter 'cities' in {self.path} contains unknown "
                f"{'city' if len(unknown) == 1 else 'cities'}: {', '.join(unknown)}"
            )
        return cities

    def get_seed(self, override: int | None = None) -> int:
        """The 'seed', or override when one was passed on the command line."""
        return override if override is not None else self.get_int("seed")


def add_config_args(ap: argparse.ArgumentParser) -> None:
    """Add the config-file and seed-override arguments shared by every script."""
    ap.add_argument(
        "config",
        nargs="?",
        default=None,
        help=f"JSON5 config file (default: {DEFAULT_CONFIG_PATH})",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the 'seed' parameter from the config file",
    )


def load_config(args: argparse.Namespace) -> Config:
    """Load the config named by parsed args, exiting with a message on failure."""
    try:
        return Config.load(args.config)
    except ConfigError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def main_with_config(main: Callable[[], None]) -> Callable[[], None]:
    """Wrap a script's main() so a bad parameter exits with a message.

    Parameters are read lazily as the script needs them, so a missing or
    ill-typed key surfaces well after load_config() returned; without this the
    script would die with a traceback instead of a one-line error.
    """

    @functools.wraps(main)
    def wrapper() -> None:
        try:
            main()
        except ConfigError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)

    return wrapper


def scenarios_from(cfg: Config) -> list[tuple[str, bool]]:
    """The (city, disasters) pairs a run covers: cities x disasters."""
    return [
        (city, disasters)
        for city in cfg.get_cities()
        for disasters in cfg.get_bool_list("disasters")
    ]
