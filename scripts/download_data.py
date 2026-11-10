"""Download the Kenneth R. French daily files into data/raw/ and record their SHA-256.

Run from the package root:  python scripts/download_data.py
The raw files are not committed (see .gitignore); the hashes are, so a rerun can be checked.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ruin.data import FILES, download, parse_daily  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

if __name__ == "__main__":
    for name in FILES:
        path = download(name, RAW)
        df = parse_daily(path)
        print(f"{name}: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
