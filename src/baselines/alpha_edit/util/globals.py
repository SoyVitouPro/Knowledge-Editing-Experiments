from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
STATS_DIR = ROOT / "cache" / "stats"
HPARAMS_DIR = ROOT / "src" / "baselines" / "alpha_edit" / "hparams"
KV_DIR = ROOT / "cache" / "alpha_edit" / "kvs"
REMOTE_ROOT_URL = ""
