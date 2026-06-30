from pathlib import Path

REALTIME_DIR = Path(__file__).resolve().parent
REPO_ROOT = REALTIME_DIR.parents[2]
LOOKUP_TABLE_PATH = REPO_ROOT / "Data_Creation" / "lookup_table.pkl"
