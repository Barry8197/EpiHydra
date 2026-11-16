import os
import polars as pl
import subprocess
from pathlib import Path

DATADIR = Path('/work/gr-fe/bryan/EpiHydra/data')
GSE = 'GSE179325'

# Find the gz file
gz_file = next(f for f in os.listdir(DATADIR / '01_raw' / GSE) if f.endswith('.gz') and 'series_matrix' not in f)
gz_path  = DATADIR / '01_raw' / GSE / gz_file
txt_path = gz_path.with_suffix('')  # strips .gz
parquet_path = DATADIR / '01_raw' / GSE / gz_file.replace('.csv.gz', '.parquet').replace('.txt.gz', '.parquet')

# Determine separator 
separator_overrides = {
    'GSE157131': ' ',
}
separator = separator_overrides.get(GSE, ',' if '.csv' in gz_file else '\t')

# Decompress
if not txt_path.exists():
    print(f'Decompressing {gz_file}...')
    subprocess.run(['gunzip', '-k', str(gz_path)], check=True)
else:
    print(f'Already decompressed, skipping...')

# Read with polars lazy
print(f'Reading with polars (separator={repr(separator)})...')
betas = pl.scan_csv(
    txt_path, 
    separator=separator, 
    has_header=True, 
    null_values=["NA", ""],
    infer_schema_length=0,  # read all as strings, avoid type inference errors
)

# Drop Detection Pval columns and cast remaining to float
beta_cols = [c for c in betas.columns if "detection" not in c.lower() and "pval" not in c.lower()]
betas = betas.select(beta_cols).with_columns(
    [pl.col(c).cast(pl.Float64, strict=False) for c in beta_cols[1:]]  # skip probe ID column
)

# Save as parquet
print('Saving as parquet...')
parquet_path.parent.mkdir(parents=True, exist_ok=True)
betas.sink_parquet(parquet_path)

# Delete uncompressed
print('Cleaning up...')
txt_path.unlink()

print(f'Done → {parquet_path}')