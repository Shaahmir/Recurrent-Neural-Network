from pathlib import Path
import pandas as pd
from config import cfg

# Fetching the parquet files

parquet_files = sorted((cfg.RAW_DATA).glob("*.parquet"))

print(f"Found {len(parquet_files)} parquet files!")

# Reading the parquet file

df = pd.read_parquet(parquet_files[0])

print(df.columns.tolist())
print()
print(df.head())
print()
print(df.dtypes)