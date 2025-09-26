# app/preprocessing.py
import pandas as pd
import numpy as np

COMMON_COLS = {
    'temp': ['temp','temperature','t','air_temp','field1','field2'],
    'pressure': ['pressure','press','P'],
    'humidity': ['humidity','hum','h'],
    'ozone': ['ozone','mq131','o3'],
    'air_quality': ['mq135','air_quality','aqi'],
    'uv_index': ['uv','uv_index','veml_uv','uvindex'],
    'lat': ['lat','latitude','gps_lat'],
    'lon': ['lon','longitude','gps_lon'],
    'timestamp': ['timestamp','time','created_at','date']
}

def normalize_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns.astype(str)]
    colmap = {}
    cols = set(df.columns)
    for standard, variants in COMMON_COLS.items():
        for v in variants:
            if v in cols and v not in colmap:
                colmap[v] = standard
    df = df.rename(columns=colmap)

    # remove duplicate columns by keeping first occurrence
    df = df.loc[:,~df.columns.duplicated()]

    # ensure timestamp column exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        try:
            ts_index = pd.to_datetime(df.index)
            if not ts_index.isnull().all():
                df['timestamp'] = ts_index
            else:
                df['timestamp'] = pd.NaT
        except Exception:
            df['timestamp'] = pd.NaT
    return df

def calibrate_mq(raw_val, sensor='mq131'):
    if pd.isna(raw_val):
        return np.nan
    if sensor == 'mq131':
        return raw_val * 0.05
    elif sensor == 'mq135':
        return raw_val * 0.03
    else:
        return raw_val

def fill_and_resample(df: pd.DataFrame, freq='1min'):
    df = df.copy()
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.NaT
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if df['timestamp'].isna().all():
        df = df.reset_index(drop=True)
        df['timestamp'] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq=freq)

    df = df.sort_values('timestamp').set_index('timestamp')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_resampled = df[numeric_cols].resample(freq).mean()
    df_resampled = df_resampled.ffill().bfill()
    df_resampled = df_resampled.reset_index()
    return df_resampled

def create_lag_features(df: pd.DataFrame, target_col, lags=[1,2,3,6,12]):
    df = df.copy()
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataframe")
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    df = df.dropna()
    return df
