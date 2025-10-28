# src/ingest.py
"""
Ingest script for Agriculture Production data.
- Reads all CSV and Excel files from data/raw
- Standardizes columns and reshapes wide year columns into long format
- Extracts production from:
    1) production_YYYY_YY columns
    2) area_YYYY_YY + yield_YYYY_YY (production = area * yield/10)
    3) generic FY columns (2004_05, 2005_06, ...) when "particulars" contains "production"
    4) simple Crop x FY table (datafile.csv-like)
    5) produce.csv with 'Particulars' + columns like ' 3-1993'.. ' 3-2014' (Tonnes / Thousand Tonnes)
- Normalizes units and saves a single CSV to data/interim/agri_combined.csv
"""

import sys, os, re, glob
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]     # project root (parent of src/)
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "interim" / "agri_combined.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[ingest] {msg}")

# ---------- Helpers ----------
def std_cols(cols):
    return (pd.Index(cols)
              .str.strip()
              .str.lower()
              .str.replace(r'[^a-z0-9]+', '_', regex=True)
              .str.strip('_'))

# Map common variants to canonical names
RENAME_MAP = {
    'state_ut': 'state', 'state_name': 'state', 'state_ut_name': 'state',
    'crop_name': 'crop',
    'crop_year': 'year', 'year_': 'year', 'season_year': 'year',
    'prod': 'production', 'produce': 'production',
    'production_tonnes': 'production', 'production_tons': 'production',
    'production_ton': 'production', 'production_in_tons': 'production',
    'production_in_tonnes': 'production',
    'variety_name': 'variety',
    'units': 'unit',
    'recommendedzone': 'recommended_zone',
    'cost_of_cultivation': 'cost', 'total_cost': 'cost',
    'area': 'quantity', 'area_ha': 'quantity', 'quantity_ha': 'quantity',
    'yield_quintal_hectare': 'yield_q_ha',
    # add more if needed based on your raw headers
}

CANONICAL_COLS = [
    'crop','variety','state','quantity','production',
    'season','unit','cost','recommended_zone','year'
]

def extract_year_from_text(text):
    if not isinstance(text, str): return None
    m = re.search(r'(20\d{2})', text)
    return int(m.group(1)) if m else None

def canonicalize(df):
    df = df.copy()
    df.columns = std_cols(df.columns)
    # rename known variants
    rename_use = {k:v for k,v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_use)
    # ensure expected columns exist
    for col in CANONICAL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    # numeric types (only if present)
    for col in ['quantity','production','cost','year','yield_q_ha']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # text tidy
    for col in ['crop','variety','state','season','unit','recommended_zone','particulars','frequency']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(['nan','None','NaN']), col] = np.nan
    return df

def read_all_files(raw_dir: Path, header=0, skiprows=None):
    paths_csv = sorted([str(p) for p in (raw_dir.glob("*.csv"))])
    paths_xls = sorted([str(p) for p in (raw_dir.glob("*.xls*"))])
    paths = paths_csv + paths_xls

    log(f"Project root: {ROOT}")
    log(f"Working dir: {Path.cwd()}")
    log(f"RAW_DIR: {raw_dir}")
    log(f"Found files: {len(paths)}")
    for p in paths:
        log(f" - {Path(p).name}")

    if not paths:
        raise FileNotFoundError(f"No CSV or Excel files found in {raw_dir}")

    frames = []
    for p in paths:
        ext = Path(p).suffix.lower()

        if ext == ".csv":
            try:
                df = pd.read_csv(p)
            except UnicodeDecodeError:
                df = pd.read_csv(p, encoding="latin1")
            if df is None or df.empty:
                log(f"  Skip empty CSV: {Path(p).name}")
                continue
            df = canonicalize(df)

            if 'year' in df.columns and df['year'].isna().all():
                yr = extract_year_from_text(Path(p).name)
                if yr is not None:
                    df['year'] = yr

            df['source_file'] = Path(p).name
            df['source_sheet'] = 'csv'
            frames.append(df)
            continue

        # Excel handling
        if ext == ".xlsx":
            engine = "openpyxl"
        elif ext == ".xls":
            engine = "xlrd"
        elif ext == ".xlsb":
            try:
                import pyxlsb  # pip install pyxlsb
            except ImportError:
                raise RuntimeError("Found .xlsb file. Install: pip install pyxlsb")
            engine = "pyxlsb"
        else:
            log(f"  Skip unknown extension: {Path(p).name}")
            continue

        log(f"Reading Excel: {Path(p).name} (engine={engine})")
        try:
            xls = pd.ExcelFile(p, engine=engine)
        except Exception as e:
            log(f"  Cannot open workbook {Path(p).name}: {e}")
            continue

        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(p, sheet_name=sheet, engine=engine,
                                   header=header, skiprows=skiprows)
            except Exception as e:
                log(f"  Sheet skip {sheet} in {Path(p).name}: {e}")
                continue
            if df is None or df.empty:
                continue
            df = canonicalize(df)

            if 'year' in df.columns and df['year'].isna().all():
                yr = extract_year_from_text(Path(p).name) or extract_year_from_text(sheet)
                if yr is not None:
                    df['year'] = yr

            df['source_file'] = Path(p).name
            df['source_sheet'] = sheet
            frames.append(df)

    if not frames:
        raise RuntimeError("No usable data read from CSV/Excel files.")
    return pd.concat(frames, ignore_index=True, sort=False)

# ---------- Reshape wide -> long ----------
# Support 2006_07, 2006-07, 2006–07, etc.
FY_PAT = re.compile(r'(\d{4})\s*[_\-–—]\s*(\d{2})$')
# Generic "any 4-digit year" pattern (e.g., column named " 3-1993")
YEAR_ONLY_PAT = re.compile(r'(\d{4})')

def fy_to_end_year(text):
    """Convert '2006_07'/'2006-07' -> 2007 (end year of the fiscal/agri year)."""
    if not isinstance(text, str): return np.nan
    m = FY_PAT.search(text)
    if not m: return np.nan
    start = int(m.group(1))  # 2006
    end2 = int(m.group(2))   # 07
    century = (start // 100) * 100
    end_year = century + end2
    if end_year < start:
        end_year += 100
    return end_year

def col_to_year_any(text):
    """Extract a plain 4-digit year from a column like ' 3-1993' or 'Year_2011'."""
    if not isinstance(text, str): return np.nan
    m = YEAR_ONLY_PAT.search(text)
    return int(m.group(1)) if m else np.nan

def melt_prefix(df, prefix, value_name, id_cols):
    """Melt columns like 'production_2006_07' -> long with 'year'."""
    cols = [c for c in df.columns if c.startswith(prefix + "_") and FY_PAT.search(c)]
    if not cols:
        return None
    tmp = df[id_cols + cols].copy()
    tmp = tmp.melt(id_vars=id_cols, value_vars=cols,
                   var_name='tmp_col', value_name=value_name)
    tmp['year'] = tmp['tmp_col'].apply(fy_to_end_year).astype('Int64')
    tmp = tmp.drop(columns=['tmp_col'])
    return tmp

def reshape_from_simple_crop_fy(combined):
    """
    Handle files like datafile.csv that have: Crop + FY columns (2004_05, 2005_06, ...),
    with no 'particulars'. Assume values are production.
    """
    if 'crop' not in combined.columns:
        return None

    # Pick FY columns like 2004_05, 2005_06 ...
    fy_cols = [c for c in combined.columns
               if FY_PAT.search(c) and not any(c.startswith(p) for p in ['production_','area_','yield_'])]

    if not fy_cols:
        return None

    dfc = combined.copy()
    # Prefer rows from datafile.csv if present
    if 'source_file' in dfc.columns:
        mask = dfc['source_file'].str.lower() == 'datafile.csv'
        if mask.any():
            dfc = dfc[mask].copy()

    if dfc.empty:
        return None

    id_cols = [c for c in ['crop','variety','state','season','unit','recommended_zone','cost',
                           'source_file','source_sheet'] if c in dfc.columns]

    keep_cols = id_cols + [c for c in fy_cols if c in dfc.columns]
    dfc = dfc[keep_cols].copy()
    if dfc.empty:
        return None

    melted = dfc.melt(id_vars=id_cols, value_vars=fy_cols, var_name='fy', value_name='production')
    melted['year'] = melted['fy'].apply(fy_to_end_year).astype('Int64')
    melted = melted.drop(columns=['fy'])

    # Ensure numeric
    melted['production'] = pd.to_numeric(melted['production'], errors='coerce')
    melted = melted[~melted['production'].isna()]

    # Unit default
    if 'unit' in melted.columns:
        melted['unit'] = melted['unit'].fillna('Tons').astype(str)
    else:
        melted['unit'] = 'Tons'

    # Convenience
    if 'quantity' not in melted.columns:
        melted['quantity'] = np.nan

    preferred = ['crop','variety','state','season','year','quantity','production','unit','cost','recommended_zone',
                 'source_file','source_sheet']
    cols = [c for c in preferred if c in melted.columns] + [c for c in melted.columns if c not in preferred]
    return melted[cols]

def reshape_from_produce(combined):
    """
    Handle produce.csv which has 'Particulars', 'Unit' and year columns like ' 3-1993'..' 3-2014'.
    We extract rows where 'Particulars' starts with 'Agricultural Production ' and parse crop/season.
    Keep Tonnes/Thousand Tonnes and normalize to Tons.
    """
    if 'source_file' not in combined.columns:
        return None

    dfp = combined[combined['source_file'].str.lower() == 'produce.csv'].copy()
    if dfp.empty:
        return None

    # Find columns with a 4-digit year anywhere
    year_cols = [c for c in dfp.columns if YEAR_ONLY_PAT.search(str(c))]
    if not year_cols:
        return None

    id_cols = [c for c in ['particulars','unit','source_file','source_sheet'] if c in dfp.columns]
    keep = id_cols + year_cols
    dfp = dfp[keep].copy()

    # Filter relevant series
    if 'particulars' not in dfp.columns:
        return None
    mask = dfp['particulars'].astype(str).str.strip().str.lower().str.startswith('agricultural production')
    dfp = dfp[mask].copy()
    if dfp.empty:
        return None

    # Melt
    tmp = dfp.melt(id_vars=id_cols, value_vars=year_cols, var_name='col_year', value_name='production')
    tmp['year'] = tmp['col_year'].apply(col_to_year_any).astype('Int64')
    tmp = tmp.drop(columns=['col_year'])

    # Unit normalization
    if 'unit' in tmp.columns:
        u = tmp['unit'].astype(str).str.lower()
        mask_thousand = u.str.contains('thousand', na=False) & u.str.contains('tonne|ton', na=False)
        mask_ton = (~mask_thousand) & u.str.contains('tonne|ton', na=False)
        # Convert thousand tonnes -> tons
        tmp.loc[mask_thousand, 'production'] = pd.to_numeric(tmp.loc[mask_thousand, 'production'], errors='coerce') * 1000.0
        tmp = tmp[mask_thousand | mask_ton].copy()
        tmp['unit'] = 'Tons'
    else:
        tmp['unit'] = 'Tons'

    # Parse crop and season from 'Particulars'
    def parse_particulars(s):
        s = str(s).strip()
        # Remove prefix
        prefix = 'agricultural production'
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
        tokens = s.split()
        season = None
        if tokens and tokens[-1].lower() in ['kharif', 'rabi']:
            season = tokens[-1].capitalize()
            tokens = tokens[:-1]
        crop = tokens[-1] if tokens else None
        # Remove generic categories when they appear alone
        if crop and crop.lower() in ['foodgrains','oilseeds','cereals','pulses','horticulture']:
            # If only generic word remains, drop
            if len(tokens) == 1:
                crop = None
        return crop, season

    parsed = tmp['particulars'].apply(parse_particulars)
    tmp['crop'] = parsed.apply(lambda x: x[0])
    tmp['season'] = parsed.apply(lambda x: x[1])
    tmp = tmp[~tmp['crop'].isna()].copy()

    # Fill missing convenience cols
    tmp['quantity'] = np.nan
    tmp['recommended_zone'] = np.nan
    tmp['variety'] = np.nan
    tmp['state'] = np.nan
    tmp['cost'] = np.nan

    preferred = ['crop','variety','state','season','year','quantity','production','unit','cost','recommended_zone',
                 'source_file','source_sheet']
    cols = [c for c in preferred if c in tmp.columns] + [c for c in tmp.columns if c not in preferred]
    return tmp[cols]

def reshape_wide_to_long(combined):
    """
    Final long builder that merges multiple paths:
    1) production_YYYY_YY (explicit)
    2) area_YYYY_YY + yield_YYYY_YY -> compute production
    3) particulars + generic FY columns (e.g., 2004_05) -> production rows
    4) simple Crop x FY table (e.g., datafile.csv) -> assume values are production
    5) produce.csv: 'Particulars' + year columns like ' 3-1993'..' 3-2014'
    """
    id_cols = [c for c in ['crop','variety','state','season','unit','recommended_zone','cost',
                           'source_file','source_sheet'] if c in combined.columns]

    # Path 1: explicit production_YYYY_YY
    prod_long = melt_prefix(combined, 'production', 'production', id_cols)
    # Path 2: area_YYYY_YY + yield_YYYY_YY
    area_long = melt_prefix(combined, 'area', 'area_ha', id_cols)
    yld_long  = melt_prefix(combined, 'yield', 'yield_q_ha', id_cols)

    out1 = None
    if prod_long is not None:
        out1 = prod_long
        if area_long is not None:
            out1 = out1.merge(area_long, on=id_cols+['year'], how='left')
        if yld_long is not None:
            out1 = out1.merge(yld_long, on=id_cols+['year'], how='left')
        log(f"Path1: explicit production_* rows = {len(out1)}")
    elif area_long is not None and yld_long is not None:
        tmp = area_long.merge(yld_long, on=id_cols+['year'], how='outer')
        tmp['production'] = (pd.to_numeric(tmp['area_ha'], errors='coerce') *
                             (pd.to_numeric(tmp['yield_q_ha'], errors='coerce') / 10.0))
        out1 = tmp
        log(f"Path2: built from area_* + yield_* rows = {len(out1)}")

    # Path 3: “particulars” + generic FY columns (2004_05, ...)
    out2 = None
    year_cols_generic = [c for c in combined.columns
                         if FY_PAT.search(c) and not any(c.startswith(p) for p in ['production_','area_','yield_'])]
    if 'particulars' in combined.columns and year_cols_generic:
        mask = combined['particulars'].astype(str).str.contains('production', case=False, na=False)
        dfp = combined[mask].copy()
        if not dfp.empty:
            ids = [c for c in ['crop','variety','state','season','unit','recommended_zone','cost',
                               'source_file','source_sheet'] if c in dfp.columns]
            tmp = dfp[ids + year_cols_generic].melt(id_vars=ids, value_vars=year_cols_generic,
                                                    var_name='fy', value_name='production')
            tmp['year'] = tmp['fy'].apply(fy_to_end_year).astype('Int64')
            tmp = tmp.drop(columns=['fy'])
            out2 = tmp
            log(f"Path3: particulars + FY columns rows = {len(out2)}")

    # Path 4: simple Crop x FY (datafile.csv)
    out3 = reshape_from_simple_crop_fy(combined)
    if out3 is not None:
        log(f"Path4: simple Crop x FY rows = {len(out3)}")

    # Path 5: produce.csv (Particulars + '3-1993' type years)
    out4 = reshape_from_produce(combined)
    if out4 is not None:
        log(f"Path5: produce.csv parsed rows = {len(out4)}")

    frames = [f for f in [out1, out2, out3, out4] if f is not None]
    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure numeric and units
    out['production'] = pd.to_numeric(out['production'], errors='coerce')
    if 'unit' in out.columns:
        out['unit'] = out['unit'].fillna('Tons').astype(str)
    else:
        out['unit'] = 'Tons'
    if 'quantity' not in out.columns and 'area_ha' in out.columns:
        out['quantity'] = out['area_ha']

    # Reorder
    preferred = ['crop','variety','state','season','year','quantity','production','unit','cost','recommended_zone',
                 'source_file','source_sheet']
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]

def normalize_units(df):
    df = df.copy()
    df['unit'] = df['unit'].fillna('Tons').astype(str)
    # Convert if any unit mentions quintal
    mask_quintal = df['unit'].str.contains('quintal', case=False, na=False)
    if mask_quintal.any():
        df.loc[mask_quintal, 'production'] = pd.to_numeric(df.loc[mask_quintal, 'production'], errors='coerce') / 10.0
        df.loc[mask_quintal, 'unit'] = 'Tons'
    return df

# ---------- Main ----------
def main():
    log(f"Python {sys.version.split()[0]}")
    wide = read_all_files(RAW_DIR)
    log(f"Initial combined shape (wide): {wide.shape}")
    log(f"Columns sample: {list(wide.columns)[:20]} ...")

    long = reshape_wide_to_long(wide)
    if long is None:
        log("ERROR: Could not build 'production' from wide/year columns. Please share sample headers.")
        # Fallback: continue with wide (will likely drop to zero rows)
        long = wide.copy()
        if 'production' not in long.columns:
            long['production'] = np.nan

    log(f"Rows before dropna(production): {long.shape}")
    if long['production'].isna().all():
        log("WARNING: 'production' is empty in all rows even after reshaping.")
        log("Check your raw column names; you may need to add mappings to RENAME_MAP.")
        log(f"Columns now: {list(long.columns)[:30]} ...")

    # Keep only rows with production
    long = long[~long['production'].isna()]
    log(f"Rows after dropna(production): {long.shape}")

    long = normalize_units(long)

    # De-duplicate
    dedup_keys = [k for k in ['state','crop','variety','season','year','quantity','production','cost'] if k in long.columns]
    long = long.drop_duplicates(subset=dedup_keys)

    long.to_csv(OUT_PATH, index=False)
    log(f"Saved {OUT_PATH} with shape {long.shape}")

    if 'source_file' in long.columns and not long.empty:
        by_file = (long.groupby('source_file')['production']
                   .size().sort_values(ascending=False))
        log("Rows per file:")
        log(str(by_file))

if __name__ == "__main__":
    main()