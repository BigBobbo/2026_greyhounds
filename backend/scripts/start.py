"""Minimal startup script with maximum error logging."""
import os
import sys
import traceback

# Ensure the project root (/app) is on sys.path so that
# `from scripts.X import ...` and `from app.X import ...` both work
# when this file is invoked as `python scripts/start.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    port = os.environ.get("PORT", "8000")
    print(f"=== START.PY: Starting on port {port} ===", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"CWD: {os.getcwd()}", flush=True)
    print(f"Files: {os.listdir('.')}", flush=True)

    # Ensure data directory
    os.makedirs("data/models", exist_ok=True)
    print(f"Data dir exists: {os.path.isdir('data')}", flush=True)

    # Run alembic (non-fatal)
    try:
        import subprocess
        r = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
        print(f"Alembic stdout: {r.stdout}", flush=True)
        if r.returncode != 0:
            print(f"Alembic stderr: {r.stderr}", flush=True)
    except Exception as e:
        print(f"Alembic failed: {e}", flush=True)

    # Seed tracks (non-fatal)
    try:
        from scripts.seed_tracks import seed
        seed()
    except Exception as e:
        print(f"Seed tracks failed: {e}", flush=True)
        traceback.print_exc()

    # Seed features (non-fatal)
    try:
        from scripts.seed_features import seed as seed_features
        seed_features()
    except Exception as e:
        print(f"Seed features failed: {e}", flush=True)
        traceback.print_exc()

    # Start uvicorn
    print(f"=== Launching uvicorn on port {port} ===", flush=True)
    os.execvp("uvicorn", [
        "uvicorn", "app.main:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--log-level", "info",
    ])


if __name__ == "__main__":
    main()
