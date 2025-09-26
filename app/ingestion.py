# app/ingestion.py
import pandas as pd
import requests
from io import StringIO
from pymongo import MongoClient

def load_csv_file(file_obj):
    """
    Accepts a file-like object from Streamlit upload or open() and returns DataFrame.
    """
    try:
        # try direct read (works if stream supports .read())
        df = pd.read_csv(file_obj)
    except Exception:
        # fallback: read bytes and decode
        file_obj.seek(0)
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
        df = pd.read_csv(StringIO(raw))
    return df

def parse_text_input(text):
    """
    Parse pasted text as CSV if possible, otherwise attempt simple key:value parsing.
    """
    text = text.strip()
    if not text:
        return pd.DataFrame()
    # try CSV parsing
    try:
        df = pd.read_csv(StringIO(text))
        return df
    except Exception:
        pass

    # try JSON lines
    try:
        import json
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass

    # fallback simple key:value or comma-separated values
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ',' in line and ':' not in line:
            parts = [p.strip() for p in line.split(',')]
            rows.append(parts)
        else:
            kv = {}
            for seg in line.split(','):
                if ':' in seg:
                    k, v = seg.split(':', 1)
                    kv[k.strip()] = v.strip()
            if kv:
                rows.append(kv)

    if not rows:
        return pd.DataFrame()
    # if rows are lists, try to infer header
    if isinstance(rows[0], list):
        maxlen = max(len(r) for r in rows)
        cols = [f'col{i+1}' for i in range(maxlen)]
        rows2 = [r + ['']*(maxlen-len(r)) for r in rows]
        return pd.DataFrame(rows2, columns=cols)
    else:
        return pd.DataFrame(rows)

def fetch_thingspeak(channel_id, read_api_key=None, results=8000, fields=None):
    """
    Fetch ThingSpeak channel feed and return DataFrame.
    If `fields` is provided, only return those fields + timestamp.
    """
    base = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {"results": results}
    if read_api_key:
        params["api_key"] = read_api_key

    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    feeds = j.get('feeds', [])
    if not feeds:
        return pd.DataFrame()

    # build rows: keep created_at + field1..field8
    rows = []
    for f in feeds:
        row = {"created_at": f.get("created_at")}
        for i in range(1, 9):
            key = f"field{i}"
            row[key] = f.get(key)
        rows.append(row)

    df = pd.DataFrame(rows)

    # add parsed timestamp
    if 'created_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_at'], errors='coerce')

    # filter only requested fields
    if fields:
        keep_cols = ['timestamp'] + [f for f in fields if f in df.columns]
        df = df[keep_cols]

    return df

def fetch_mongodb(uri, db_name, collection_name, query=None, limit=10000):
    """
    Fetch documents from MongoDB into a DataFrame. Keep _id as string.
    """
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]
    if query is None:
        query = {}
    cursor = coll.find(query).limit(limit)
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)
    return df
