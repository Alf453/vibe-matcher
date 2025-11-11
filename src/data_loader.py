"""
Data loading helpers for the Vibe Matcher prototype.
- load_products_json(path): JSON -> DataFrame with required columns
- validate_schema(df): ensures required columns exist
"""
from __future__ import annotations
from typing import List, Dict
import json
import pandas as pd

REQUIRED_COLS = ["name", "desc", "vibes"]


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_products_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return validate_schema(df)


def load_products_list(items: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    return validate_schema(df)
