"""
main.py — Launch backend (FastAPI) and frontend (Streamlit) together.

Usage:
    python main.py

Then open:
    Frontend  → http://localhost:8501
    API docs  → http://localhost:8000/docs
    Health    → http://localhost:8000/health
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

BACKEND_PORT  = 8000
FRONTEND_PORT = 8501


def _check_artefacts() -> None:
    required = ["model_cv_best.json", "feature_cols.json", "assignment-dataset.json"]
    missing  = [f for f in required if not (BASE_DIR / f).exists()]
    if missing:
        print("ERROR: required artefacts not found:", ", ".join(missing))
        print("Run all cells of train.ipynb first, then re-run main.py.")
        sys.exit(1)


def _pip_install(packages: list[str]) -> None:
    """Install packages that are not yet importable."""
    import importlib.util
    need = [p for p in packages if importlib.util.find_spec(p.split("[")[0].replace("-", "_")) is None]
    if need:
        print(f"Installing missing packages: {need}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *need]
        )


def main() -> None:
    _check_artefacts()

    # Ensure runtime deps are available (graceful first-run install)
    _pip_install([
        "fastapi",
        "uvicorn[standard]",
        "streamlit",
        "requests",
        "sentence-transformers",
        "shap",
    ])

    print("=" * 60)
    print("  Social Media Performance Predictor")
    print("=" * 60)

    # ── Start FastAPI backend ─────────────────────────────────────────────────
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend:app",
        "--host", "0.0.0.0",
        "--port", str(BACKEND_PORT),
    ]
    backend = subprocess.Popen(backend_cmd, cwd=str(BASE_DIR))
    print(f"[backend]  starting on http://localhost:{BACKEND_PORT} …")

    # Give the backend a moment to bind the port before Streamlit opens
    time.sleep(3)

    # ── Start Streamlit frontend ──────────────────────────────────────────────
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(BASE_DIR / "frontend.py"),
        "--server.port", str(FRONTEND_PORT),
        "--server.headless", "true",
    ]
    frontend = subprocess.Popen(frontend_cmd, cwd=str(BASE_DIR))

    print(f"[frontend] starting on http://localhost:{FRONTEND_PORT} …")
    print()
    print(f"  🌐  App     → http://localhost:{FRONTEND_PORT}")
    print(f"  📖  API docs → http://localhost:{BACKEND_PORT}/docs")
    print(f"  ❤️   Health  → http://localhost:{BACKEND_PORT}/health")
    print()
    print("Press Ctrl+C to stop both services.")
    print("=" * 60)

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("Stopped.")


if __name__ == "__main__":
    main()
