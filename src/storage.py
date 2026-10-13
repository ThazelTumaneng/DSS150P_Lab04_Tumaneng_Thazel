# src/storage.py
import hashlib
import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensures that a directory exists, creating all parent folders if necessary."""
    target_path = Path(path)
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def compute_file_hash(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Computes the SHA-256 checksum of a file to ensure reproducible data lineage."""
    src = Path(filepath)
    if not src.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {src}")

    sha256 = hashlib.sha256()
    with open(src, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    file_format: Optional[str] = None,
    index: bool = False,
    **kwargs,
) -> Path:
    """Saves a pandas DataFrame to disk in Parquet or CSV format."""
    dest = Path(filepath)
    ensure_dir(dest.parent)

    fmt = file_format or dest.suffix.lstrip(".").lower()
    if fmt == "parquet":
        df.to_parquet(dest, index=index, **kwargs)
    elif fmt == "csv":
        df.to_csv(dest, index=index, **kwargs)
    else:
        raise ValueError(f"Unsupported file format '{fmt}'. Choose 'parquet' or 'csv'.")

    return dest


def load_dataframe(
    filepath: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Loads a pandas DataFrame from a Parquet or CSV file."""
    src = Path(filepath)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    fmt = file_format or src.suffix.lstrip(".").lower()
    if fmt == "parquet":
        return pd.read_parquet(src, **kwargs)
    elif fmt == "csv":
        return pd.read_csv(src, **kwargs)
    else:
        raise ValueError(f"Unable to read file with extension: {src.suffix}")


def get_latest_file(directory: Union[str, Path], pattern: str = "*.parquet") -> Path:
    """Finds the most recently modified file matching the specified pattern."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = list(dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {dir_path}")

    return max(files, key=os.path.getmtime)


def export_multiformat(df: pd.DataFrame, base_path: Union[str, Path]):
    """Exports a DataFrame to JSONL and age-partitioned Parquet directories.
    
    Addresses storage format variety requirements (JSON Lines & partition-pruning).
    """
    path = Path(base_path)
    ensure_dir(path.parent)

    # 1. Export JSON Lines (JSONL)
    jsonl_path = path.with_suffix(".jsonl")
    df.to_json(jsonl_path, orient="records", lines=True)

    # 2. Export Partitioned Parquet (Hive-style partitioned by age decade)
    if "age" in df.columns:
        partition_dir = path.parent / f"{path.stem}_partitioned_age"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        df_part = df.copy()
        df_part["age_decade"] = (df_part["age"] // 10) * 10
        df_part.to_parquet(partition_dir, partition_cols=["age_decade"], index=False)