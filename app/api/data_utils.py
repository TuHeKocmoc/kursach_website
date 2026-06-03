from datetime import datetime, timezone
from typing import Any, Optional, cast

import pandas as pd


def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_download_df(df: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    
    cols = cast(pd.MultiIndex, df.columns)
    
    if isinstance(df.columns, pd.MultiIndex):
        cols = cast(pd.MultiIndex, df.columns)
        for level in range(cols.nlevels):
            values = {str(v).lower() for v in cols.get_level_values(level)}
            if {"open", "high", "low", "close"}.issubset(values):
                out = df.copy()
                out.columns = cols.get_level_values(level)
                return out
        lvl0 = cols.get_level_values(0)
        lvl1 = cols.get_level_values(1)
        if symbol in lvl0:
            out = df.loc[:, pd.IndexSlice[symbol, :]]
            out.columns = out.columns.droplevel(0)
            return out
        if symbol in lvl1:
            out = df.loc[:, pd.IndexSlice[:, symbol]]
            out.columns = out.columns.droplevel(1)
            return out
    return df.copy()


def ts_to_datetime(value: Any) -> Optional[datetime]:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    t = ts.to_pydatetime()
    if not isinstance(t, datetime):
        return None
    return as_utc(t)
