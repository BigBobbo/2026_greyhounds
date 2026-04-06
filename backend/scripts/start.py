"""Startup script: runs migrations, seeds, then starts uvicorn."""
import os
import sys
import subprocess


def run(cmd: str) -> bool:
    print(f">>> {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"WARNING: '{cmd}' exited with code {result.returncode}", flush=True)
        return False
    return True


def main():
    # Ensure data directory exists
    os.makedirs("data/models", exist_ok=True)
    print("Data directory ready", flush=True)

    # Run migrations (non-fatal if they fail)
    run("alembic upgrade head")

    # Seed data (non-fatal)
    run("python scripts/seed_tracks.py")
    run("python scripts/seed_features.py")

    # Start the server
    port = os.environ.get("PORT", "8000")
    print(f"Starting uvicorn on port {port}", flush=True)
    os.execvp("uvicorn", [
        "uvicorn", "app.main:app",
        "--host", "0.0.0.0",
        "--port", port,
    ])


if __name__ == "__main__":
    main()
