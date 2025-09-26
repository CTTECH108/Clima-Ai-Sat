# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from io import BytesIO
from app import ingestion, preprocessing, ndvi, models, utils

st.set_page_config(layout="wide", page_title="ClimaAI-MicroSat ML Module (Updated)")

st.title("ClimaAI-MicroSat — ML Module & Visualization (Updated and Robust)")

# ---------------- Session State Handling ----------------
if "df" not in st.session_state:
    st.session_state["df"] = None

# Sidebar: data source selection
st.sidebar.header("Data Input")
data_source = st.sidebar.selectbox(
    "Choose input source",
    ["Upload CSV", "Paste/Text", "ThingSpeak API", "MongoDB"]
)

df = st.session_state["df"]

# ---------------- Data Input ----------------
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded:
        try:
            df = ingestion.load_csv_file(uploaded)
            st.session_state["df"] = df
            st.sidebar.success("CSV loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to load CSV: {e}")

elif data_source == "Paste/Text":
    txt = st.sidebar.text_area("Paste CSV or key:value lines")
    if st.sidebar.button("Parse Text"):
        try:
            df = ingestion.parse_text_input(txt)
            st.session_state["df"] = df
            st.sidebar.success("Text parsed")
        except Exception as e:
            st.sidebar.error(f"Parse error: {e}")

elif data_source == "ThingSpeak API":
    channel_id = st.sidebar.text_input("ThingSpeak Channel ID")
    read_key = st.sidebar.text_input("Read API Key (optional)")
    fields_txt = st.sidebar.text_input("Fields (comma e.g. 1,2,3) or leave blank")
    if st.sidebar.button("Fetch ThingSpeak"):
        try:
            fields = [int(x.strip()) for x in fields_txt.split(',')] if fields_txt else None
            df = ingestion.fetch_thingspeak(channel_id, read_api_key=read_key, fields=fields)
            st.session_state["df"] = df
            st.sidebar.success("Fetched ThingSpeak")
        except Exception as e:
            st.sidebar.error(f"Error fetching ThingSpeak: {e}")

elif data_source == "MongoDB":
    uri = st.sidebar.text_input("MongoDB URI (mongodb+srv://...)")
    dbname = st.sidebar.text_input("DB name")
    coll = st.sidebar.text_input("Collection name")
    if st.sidebar.button("Fetch MongoDB"):
        try:
            df = ingestion.fetch_mongodb(uri, dbname, coll)
            st.session_state["df"] = df
            st.sidebar.success("Fetched MongoDB")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# ---------------- Data Validation ----------------
if df is None:
    st.info("No data loaded yet. Upload CSV or fetch data.")
    st.stop()

# Remove duplicate columns immediately
df = df.loc[:, ~df.columns.duplicated()]

st.subheader("Raw Data Preview")
st.write(df.head(30))

# ---------------- Preprocessing ----------------
st.subheader("Preprocessing")
try:
    df_norm = preprocessing.normalize_columns(df)
    st.write("Normalized columns:", df_norm.columns.tolist())
except Exception as e:
    st.error(f"Normalization failed: {e}")
    st.stop()

# Resample & fill
if st.button("Resample & Fill (1 min)"):
    try:
        df_filled = preprocessing.fill_and_resample(df_norm, freq='1min')
        st.success("Resampled & filled")
    except Exception as e:
        st.error(f"Resample failed: {e}")
        df_filled = df_norm.copy()
else:
    try:
        df_filled = preprocessing.fill_and_resample(df_norm, freq='1min')
    except Exception:
        df_filled = df_norm.copy()
        if 'timestamp' not in df_filled.columns:
            df_filled['timestamp'] = pd.to_datetime(
                df_filled.get('created_at', pd.Series([pd.NaT] * len(df_filled))),
                errors='coerce'
            )

# Ensure timestamp column exists
if 'timestamp' not in df_filled.columns:
    if df_filled.index.dtype == 'datetime64[ns]':
        df_filled = df_filled.reset_index().rename(columns={'index': 'timestamp'})
    else:
        if 'created_at' in df_filled.columns:
            df_filled['timestamp'] = pd.to_datetime(df_filled['created_at'], errors='coerce')
        else:
            df_filled['timestamp'] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(df_filled), freq='T'
            )

# Remove duplicates again
df_filled = df_filled.loc[:, ~df_filled.columns.duplicated()]

# Save back processed df to session
st.session_state["df"] = df_filled

# Show processed
st.write("Processed (head):")
st.write(df_filled.head(20))

# ---------------- Identify numeric columns ----------------
numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
st.write("Numeric columns detected:", numeric_cols)

# ---------------- Modeling / Analysis ----------------
st.subheader("Modeling / Analysis")

possible_features = numeric_cols.copy()
feature_cols = st.multiselect(
    "Select features for anomaly detection",
    possible_features,
    default=possible_features[:4]
)

if st.button("Train Anomaly Detector"):
    if not feature_cols:
        st.error("Select at least one numeric feature for anomaly detection.")
    else:
        try:
            iso, df_anom = models.train_anomaly_detector(df_filled, feature_cols)
            st.session_state['anomaly_model'] = iso
            st.write(df_anom.head())
            st.success("Anomaly model trained.")
        except Exception as e:
            st.error(f"Error training anomaly detector: {e}")

# Predictor training
preferred_targets = ['temp', 'temperature', 't', 'field1', 'field2']
available_targets = [c for c in df_filled.columns if c in preferred_targets]
auto_target = available_targets[0] if available_targets else (numeric_cols[0] if numeric_cols else None)

target_col = st.text_input(
    "Predict target column for next-step (leave empty to auto-select)",
    value=str(auto_target) if auto_target else ""
)
lags_text = st.text_input("Lags (comma) e.g. 1,2,3,6", value="1,2,3")

if st.button("Train Predictor (RandomForest)"):

    if not target_col:
        if numeric_cols:
            selected_target = numeric_cols[0]
            st.info(f"No target specified — using first numeric column: {selected_target}")
        else:
            st.error("No numeric columns to use as target.")
            selected_target = None
    else:
        selected_target = target_col.strip()
        if selected_target not in df_filled.columns:
            st.warning(f"Requested target '{selected_target}' not present. Using first numeric column instead.")
            selected_target = numeric_cols[0] if numeric_cols else None

    if selected_target:
        try:
            lags_list = [int(x.strip()) for x in lags_text.split(',') if x.strip().isdigit()]
        except Exception:
            lags_list = [1, 2, 3]

        try:
            df_for_model = df_filled.copy()
            df_for_model = df_for_model.rename(columns={selected_target: 'temp'})
            model_rf, mse, used_lags = models.train_predictor_with_lags(
                df_for_model, target_col='temp', lags=lags_list
            )
            st.session_state['predictor'] = model_rf
            st.session_state['predictor_lags'] = used_lags
            st.success(f"Predictor trained. MSE: {mse:.4f}. Used lags: {used_lags}")
        except Exception as e:
            st.error(f"Error training predictor: {e}")

# ---------------- Run analysis ----------------
st.subheader("Run analysis (single latest row)")
use_thresholds = st.checkbox("Use threshold rules", value=False)
threshold_rules = {}
if use_thresholds:
    t_temp = st.number_input("Temp threshold (°C)", value=40.0)
    t_oz = st.number_input("Ozone low threshold", value=0.05)
    threshold_rules = {'temp': t_temp, 'ozone': t_oz}

if st.button("Analyze latest row"):
    latest = df_filled.iloc[-1]
    predictor = st.session_state.get('predictor', None)
    predictor_lags = st.session_state.get('predictor_lags', None)
    anomaly_model = st.session_state.get('anomaly_model', None)
    comment = models.ensemble_reasoning(
        latest,
        anomaly_model=anomaly_model,
        predictor=predictor,
        predictor_lags=predictor_lags,
        threshold_rules=threshold_rules
    )
    st.markdown("### Analysis Comments")
    st.write(comment)
    st.markdown("### Latest telemetry")
    st.write(latest)

# ---------------- NDVI processing ----------------
st.subheader("NDVI Image Processing (OpenCV)")
col1, col2 = st.columns(2)
with col1:
    rgb_file = st.file_uploader("Upload RGB image (normal camera)", type=['jpg','jpeg','png'], key='rgb')
with col2:
    nir_file = st.file_uploader("Upload NIR image (NDVI camera)", type=['jpg','jpeg','png'], key='nir')

if st.button("Compute NDVI"):
    if rgb_file is None or nir_file is None:
        st.error("Please upload both RGB and NIR images.")
    else:
        try:
            rgb_bytes = np.frombuffer(rgb_file.read(), dtype=np.uint8)
            nir_bytes = np.frombuffer(nir_file.read(), dtype=np.uint8)
            rgb_img = cv2.imdecode(rgb_bytes, cv2.IMREAD_COLOR)
            nir_img = cv2.imdecode(nir_bytes, cv2.IMREAD_COLOR)
            aligned_rgb = ndvi.align_images(nir_img, rgb_img)
            ndvi_map, ndvi_col = ndvi.compute_ndvi(nir_img, aligned_rgb)
            st.image(
                cv2.cvtColor(ndvi_col, cv2.COLOR_BGR2RGB),
                caption="NDVI Color Map",
                use_column_width=True
            )
            st.write(f"NDVI stats: min {ndvi_map.min():.3f}, max {ndvi_map.max():.3f}, mean {ndvi_map.mean():.3f}")
        except Exception as e:
            st.error(f"NDVI processing failed: {e}")

# ---------------- Visualization ----------------
st.subheader("Visualization")
if 'anomaly_model' in st.session_state:
    try:
        iso = st.session_state['anomaly_model']
        if feature_cols:
            X = df_filled[feature_cols].dropna().copy()
            timestamps = pd.to_datetime(df_filled['timestamp'], errors='coerce')
            df_vis = X.copy()
            df_vis['anomaly_score'] = iso.decision_function(X)
            df_vis['anomaly_label'] = iso.predict(X)
            ts = timestamps[-len(df_vis):].reset_index(drop=True)
            df_vis = df_vis.reset_index(drop=True)
            df_vis['timestamp'] = ts
            import plotly.express as px
            st.plotly_chart(px.line(df_vis, x='timestamp', y='anomaly_score', title='Anomaly Score Over Time'))
        else:
            st.info("Select feature columns to visualize anomaly score.")
    except Exception as e:
        st.warning(f"Could not visualize anomaly model: {e}")
else:
    st.info("No anomaly model in session. Train one to visualize.")

# ---------------- Export ----------------
st.markdown("---")
st.write("Export processed dataset")
if st.button("Download processed CSV"):
    csv_link = utils.df_to_csv_download_link(df_filled, name="processed.csv")
    st.markdown(f"[Download processed CSV]({csv_link})")
