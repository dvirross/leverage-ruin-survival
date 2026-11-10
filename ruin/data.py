"""Download and parse the Kenneth R. French Data Library daily files.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Files are in percent; the market return is Mkt-RF + RF (value-weighted CRSP market including dividends).
The raw files are not redistributed here; run ``scripts/download_data.py``.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "us_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "japan_daily": "Japan_3_Factors_Daily_CSV.zip",
}


def download(name: str, dest: Path) -> Path:
    url = BASE + FILES[name]
    req = Request(url, headers={"User-Agent": "research-download"})
    data = urlopen(req, timeout=120).read()
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / FILES[name]
    out.write_bytes(data)
    (dest / (FILES[name] + ".sha256")).write_text(hashlib.sha256(data).hexdigest() + "\n")
    return out


def parse_daily(zip_path: Path) -> pd.DataFrame:
    """Parse a French daily factor zip file into a DataFrame of decimal returns indexed by date."""
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_csv_text(text)


def parse_csv_text(text: str) -> pd.DataFrame:
    """Parse the CSV text of a French daily factor file.

    The file has free-text header lines, then a header row starting with a comma, then rows ``YYYYMMDD,...``,
    and sometimes a copyright footer. Only rows whose first field is an 8-digit date are kept, up to the
    first blank line after the data (later blocks, if any, are other tables).
    """
    lines = text.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith(",") and "Mkt-RF" in l)
    cols = ["date"] + [c.strip() for c in lines[header_idx].split(",")[1:]]
    rows = []
    for l in lines[header_idx + 1:]:
        parts = [x.strip() for x in l.split(",")]
        if len(parts) == len(cols) and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
        elif rows:
            break
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0
    df["MKT"] = df["Mkt-RF"] + df["RF"]
    return df
