"""Warnings and errors: red, on stderr.

A long prompting run scrolls hundreds of progress lines, so a failed call has
to stand out and has to be separable from the report — hence stderr rather
than stdout, and red rather than plain. Everything that is not a warning or an
error stays on stdout as ordinary output.

Color is dropped when stderr is not a terminal, so a redirected log holds no
escape sequences, and when NO_COLOR is set (https://no-color.org).
"""

import os
import sys

RED = "\033[31m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"


def color_enabled(stream=None) -> bool:
    """Whether to emit ANSI codes on `stream` (default stderr)."""
    stream = stream or sys.stderr
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _emit(text: str, prefix: str, color: str) -> None:
    body = f"{prefix}{text}"
    if color_enabled():
        body = f"{color}{body}{RESET}"
    print(body, file=sys.stderr, flush=True)


def warn(text: str) -> None:
    """One warning line on stderr, red: something is off but the run goes on."""
    _emit(text, "[warning] ", RED)


def error(text: str) -> None:
    """One error line on stderr, bold red: something did not happen at all."""
    _emit(text, "[error] ", BOLD_RED)


def plain(text: str) -> None:
    """A continuation line on stderr, red but unprefixed.

    For the detail lines under a warning or error — a per-failure entry in a
    summary — which read as part of it rather than as errors of their own.
    """
    _emit(text, "", RED)
