import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Make root importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make scripts importable as top-level (so "import model_train" works)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))