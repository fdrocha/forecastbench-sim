"""Shared helpers for script-level unit tests.

The analysis scripts are standalone files (not packages), so tests load them
by path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(rel_path: str):
    """Import a repo script (e.g. 'scripts/uplift_v2/natcond_cells_v2.py')."""
    name = "civbench_test_" + Path(rel_path).stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
