"""Run the shared-server RULER concurrency acceptance check.

Arguments are --server-config, --data and --output. Each cache mode is a
separate server run; see README.md.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[5] / "scripts/validate_ruler_concurrency.py"
    runpy.run_path(str(script), run_name="__main__")
