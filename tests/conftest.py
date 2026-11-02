"""Shared test setup.

All three projects expose their code as a package called `src`, so importing two of them
in one pytest process would resolve to whichever directory happened to be first on
`sys.path`. `use_project()` makes the choice explicit: it removes every project directory
from the path, inserts the requested one, and evicts any previously imported `src.*`
module so the next import genuinely re-resolves.

Call it at the top of a test (before importing anything from `src`), not at module import
time, so the ordering is visible in the test body.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# `absolute()` rather than `resolve()` on purpose: resolve() follows reparse points, and
# on Windows that can rewrite the repository path into a virtualised view of itself where
# the package's modules are not visible, leaving `src` as an empty namespace package.
ROOT = Path(__file__).absolute().parent.parent

PROJECTS = {
    1: "project1_customer_churn",
    2: "project2_predictive_maintenance",
    3: "project3_recommendation",
}


def use_project(number: int) -> Path:
    """Point `import src.…` at the given project. Returns that project's directory."""
    for name in PROJECTS.values():
        path = str(ROOT / name)
        while path in sys.path:
            sys.path.remove(path)

    for module in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[module]

    project_dir = ROOT / PROJECTS[number]
    sys.path.insert(0, str(project_dir))
    if str(ROOT) not in sys.path:          # so `common.…` stays importable
        sys.path.append(str(ROOT))

    # A new directory on sys.path needs the finder caches dropped, or a previously
    # cached negative lookup for `src` is reused.
    importlib.invalidate_caches()
    return project_dir
