import sys
from pathlib import Path

# Make the `app` package importable when pytest runs from the repo root or orchestrator/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
