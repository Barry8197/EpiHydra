#!/usr/bin/env python3
"""
build_methylation_pkls.py

Builds methylation.pkl for each GSE using only files already on disk.

Strategy per GSE:
  - If a parquet exists → use it for CpG betas
  - If only a series matrix exists → extract betas from it directly
  - Age metadata always comes from the series matrix
  - Samples missing age are dropped from both meta and CpG
  - No imputation is performed
  - CpG rows that are all-NaN or all-zero across all samples are dropped

Output format:
    {"meta": pd.DataFrame, "CpG": pd.DataFrame}

    meta columns : ID, Organism, platform, age, [gender, tissue if present]
    CpG          : probes x samples (columns = GSM IDs matching meta["ID"])
"""

import re
import gzip
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

RAW_DIR       = Path("/work/gr-fe/bryan/EpiHydra/data/01_raw")
PROCESSED_DIR = Path("/work/gr-fe/bryan/EpiHydra/data/02_processed")

GSE_LIST = [
    "GSE111629",
    "GSE157131",
    "GSE169156",
    "GSE179325",
    "GSE19711",
    "GSE210255",
    "GSE40279",
    "GSE41037",
    "GSE42861",
    "GSE55763",
    "GSE84727",
]

# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_parquet(raw_dir: Path) -> Path | None:
    parquets = [f for f in raw_dir.iterdir() if f.suffix == ".parquet"]
    return parquets[0] if parquets else None


def find_series_matrix(raw_dir: Path) -> Path | None:
    candidates = [
        f for f in raw_dir.iterdir()
        if "series_matrix" in f.name and f.name.endswith(".gz")
    ]
    return candidates[0] if candidates else None



# ──────────────────────────────────────────────────────────────────────────────
# Metadata parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_metadata(filepath: Path) -> dict:
    metadata = {}
    with gzip.open(filepath, "rt") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!"):
                continue
            line = line.strip().lstrip("!")
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            key, value = parts

            if key == "Sample_characteristics_ch1":
                if "age" in parts[1].lower():
                    key = "age"
                    cleaned = []
                    pattern = re.compile(
                        r'(age[^:]*):\s*(\d+(?:\.\d+)?)', re.IGNORECASE
                    )
                    for token in parts[1].split("\t"):
                        m = pattern.search(token)
                        if m:
                            key_part = m.group(1)
                            val = float(m.group(2))
                            if re.search(r'\bmonth', key_part, re.IGNORECASE):
                                val = round(val / 12, 3)
                            elif re.search(r'\bweek', key_part, re.IGNORECASE):
                                val = round(val / 52, 3)
                            elif re.search(r'\bday', key_part, re.IGNORECASE):
                                val = round(val / 365, 3)
                            cleaned.append(str(val))
                        else:
                            cleaned.append("nan")
                    value = "  ".join(cleaned)
                else:
                    key = (
                        parts[1].split("\t")[0]
                        .split(":")[0]
                        .split(" ")[0]
                        .strip()
                        .replace('"', "")
                    )
                    cleaned = []
                    for token in parts[1].split("\t"):
                        piece = token.split(":", 1)
                        v = piece[1] if len(piece) == 2 else piece[0]
                        cleaned.append(v.strip().replace('"', ""))
                    value = "  ".join(cleaned)

            if key not in metadata:
                metadata[key] = value.strip('"')

    return metadata


def build_meta_df(metadata: dict) -> pd.DataFrame:
    idx      = [i.replace('"', "") for i in metadata["Sample_geo_accession"].split("\t")]
    organism = [i.replace('"', "") for i in metadata["Sample_organism_ch1"].split("\t")]
    platform = [i.replace('"', "") for i in metadata["Sample_platform_id"].split("\t")]

    data = {"ID": idx, "Organism": organism, "platform": platform}

    for field, caster in (("age", np.float32), ("gender", str), ("tissue", str)):
        if field in metadata:
            values = metadata[field].split("  ")
            try:
                data[field] = [caster(v) for v in values]
            except ValueError:
                data[field] = values
        else:
            if field == "age":
                print(f"  WARNING: 'age' not found in metadata.")
            else:
                print(f"  note: field '{field}' not found, skipping.")

    return pd.DataFrame(data)


# ──────────────────────────────────────────────────────────────────────────────
# CpG loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_parquet_as_cpg(parquet_path: Path, sample_ids: list) -> pd.DataFrame:
    print(f"  Loading parquet: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    # Set index to probe IDs if first column looks like CpG names
    first_col = df.columns[0]
    if df[first_col].dtype == object:
        df = df.set_index(first_col)

    df.index   = df.index.astype(str)
    df.columns = df.columns.astype(str)
    sample_ids = [str(s) for s in sample_ids]

    col_matches = sum(s in df.columns for s in sample_ids)
    idx_matches = sum(s in df.index   for s in sample_ids)

    if col_matches == 0 and idx_matches == 0:
        raise ValueError(
            f"Could not match GSM IDs in parquet. "
            f"First cols: {list(df.columns[:5])}"
        )

    expr = df if col_matches >= idx_matches else df.T

    missing = [s for s in sample_ids if s not in expr.columns]
    if missing:
        print(f"  WARNING: {len(missing)} samples not in parquet — dropping from meta.")
        sample_ids = [s for s in sample_ids if s in expr.columns]

    expr = expr.loc[:, sample_ids]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    print(f"  Loaded: {expr.shape[0]:,} CpGs × {expr.shape[1]} samples")
    return expr


def load_series_matrix_as_cpg(filepath: Path, sample_ids: list) -> pd.DataFrame:
    print(f"  Parsing betas from series matrix: {filepath.name}")
    df = pd.read_csv(
        filepath, sep="\t", comment="!", index_col=0, compression="gzip"
    )
    df.index   = df.index.astype(str)
    df.columns = df.columns.astype(str)
    sample_ids = [str(s) for s in sample_ids]

    available = [s for s in sample_ids if s in df.columns]
    missing   = [s for s in sample_ids if s not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} samples not in matrix — dropping from meta.")

    expr = df[available].apply(pd.to_numeric, errors="coerce")
    print(f"  Loaded: {expr.shape[0]:,} CpGs × {expr.shape[1]} samples")
    return expr



# ──────────────────────────────────────────────────────────────────────────────
# CpG cleaning
# ──────────────────────────────────────────────────────────────────────────────

def drop_uninformative_cpgs(expr: pd.DataFrame) -> pd.DataFrame:
    n_before = len(expr)
    all_nan  = expr.isna().all(axis=1)
    expr     = expr.loc[~all_nan]
    all_zero = (expr.fillna(0) == 0).all(axis=1)
    expr     = expr.loc[~all_zero]
    n_dropped = n_before - len(expr)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} all-NaN or all-zero CpG rows.")
    return expr


# ──────────────────────────────────────────────────────────────────────────────
# Per-GSE processing
# ──────────────────────────────────────────────────────────────────────────────

def process_gse(gse: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {gse}")
    print(f"{'='*60}")

    raw_dir        = RAW_DIR / gse
    processed_path = PROCESSED_DIR / gse / "methylation.pkl"

    if processed_path.exists():
        print(f"  methylation.pkl already exists, skipping.")
        return

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    # ── 1. Metadata ───────────────────────────────────────────────────────────
    series_matrix = find_series_matrix(raw_dir)
    if series_matrix is None:
        raise FileNotFoundError(f"No series matrix found in {raw_dir}")

    print(f"  Parsing metadata from {series_matrix.name}")
    metadata = parse_metadata(series_matrix)
    meta     = build_meta_df(metadata)
    print(f"  Samples in meta: {len(meta)}")

    # ── 2. Drop samples missing age ───────────────────────────────────────────
    if "age" not in meta.columns:
        raise ValueError(f"{gse}: no age column parsed from series matrix.")

    before = len(meta)
    meta   = meta.dropna(subset=["age"]).reset_index(drop=True)
    if len(meta) < before:
        print(f"  Dropped {before - len(meta)} samples with missing age.")

    sample_ids = meta["ID"].tolist()
    print(f"  Samples with age: {len(sample_ids)}")

    # ── 3. Load CpG betas ─────────────────────────────────────────────────────
    parquet = find_parquet(raw_dir)

    if parquet is not None:
        print(f"  Parquet found — using for CpG data.")
        expr = load_parquet_as_cpg(parquet, sample_ids)
    else:
        matrix_mb = series_matrix.stat().st_size / (1024 ** 2)
        print(f"  No parquet — parsing betas from series matrix ({matrix_mb:.1f} MB).")
        expr = load_series_matrix_as_cpg(series_matrix, sample_ids)

    # ── 4. Align meta and CpG ─────────────────────────────────────────────────
    available  = [s for s in sample_ids if s in expr.columns]
    if len(available) < len(sample_ids):
        print(f"  Restricting meta to {len(available)} samples present in CpG.")
        meta       = meta[meta["ID"].isin(available)].reset_index(drop=True)
        sample_ids = available

    expr = expr[sample_ids]  # enforce column order = meta order

    # ── 5. Drop uninformative CpGs ────────────────────────────────────────────
    expr = drop_uninformative_cpgs(expr)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump({"meta": meta, "CpG": expr}, f)

    print(
        f"  Saved → {processed_path}\n"
        f"  Shape : {expr.shape[0]:,} CpGs × {expr.shape[1]} samples | "
        f"Age range: {meta['age'].min():.1f}–{meta['age'].max():.1f} yrs"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    succeeded, failed = [], []

    for gse in GSE_LIST:
        try:
            process_gse(gse)
            succeeded.append(gse)
        except Exception as e:
            print(f"\n  {gse}: FAILED — {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(gse)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Succeeded ({len(succeeded)}): {succeeded}")
    if failed:
        print(f"Failed    ({len(failed)}): {failed}")
        print("Re-run to retry failed datasets (completed ones are skipped).")


if __name__ == "__main__":
    main()