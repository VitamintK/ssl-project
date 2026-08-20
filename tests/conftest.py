import sys
from pathlib import Path

# Add the repo root to sys.path so tests can import config and other modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
