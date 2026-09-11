from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
FINAL = ROOT / "data" / "final"


def load_orders(name="orders_train"):
    """Read a raw DMC 2016 order-line file (``orders_train`` has ``returnQuantity``; ``orders_class`` is the unlabeled test set)."""
    return pd.read_csv(RAW / f"{name}.txt", sep=";", parse_dates=["orderDate"], dtype={"sizeCode": str})


def save_features(df, name, keys=("orderID", "orderDate", "articleID", "customerID"), target="returnQuantity"):
    """Clean column names, downcast feature columns to float32, and write a parquet block to ``data/interim``."""
    df = df.copy()
    if df.index.name is not None and df.index.name not in df.columns:
        df = df.reset_index()
    df.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)) for c in df.columns]
    feat = [c for c in df.columns if c not in keys and c != target and pd.api.types.is_numeric_dtype(df[c])]
    df[feat] = df[feat].astype("float32")
    df.to_parquet(INTERIM / f"{name}.parquet", index=False)
    return df.shape
