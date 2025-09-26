# app/utils.py
import base64
import pandas as pd

def df_to_csv_download_link(df: pd.DataFrame, name="results.csv"):
    """
    Return data URI that Streamlit can use to download CSV.
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href
