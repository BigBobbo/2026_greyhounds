"""Test environment setup.

Points DATABASE_URL at a migrated temp database BEFORE any app module is
imported, so tests exercise the real alembic-built schema instead of the
developer's ./data/greyhound.db. Importing this also serves as a regression
test for the fresh-database migration path (audit task B1).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

_tmpdir = tempfile.mkdtemp(prefix="greyhound-test-")
_db_path = os.path.join(_tmpdir, "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

_result = subprocess.run(
    [sys.executable, "-m", "alembic", "upgrade", "head"],
    cwd=BACKEND_DIR,
    env={**os.environ},
    capture_output=True,
    text=True,
)
if _result.returncode != 0:
    raise RuntimeError(
        f"alembic upgrade head failed on a fresh database:\n{_result.stderr}"
    )
