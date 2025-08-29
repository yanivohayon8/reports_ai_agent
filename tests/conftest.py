import sys
import os
from pathlib import Path

# Add backend directory to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT_DIR / "backend"

if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))
