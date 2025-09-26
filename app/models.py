# app/models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump, load

def train_anomaly_detector(df, feature_cols, contamination=0.01, random_state=42):
    """
    Train and return an IsolationForest on given features.
    Returns: (model, scored_df)
    """
    X = df[feature_cols].dropna()
    if X.shape[0] < 5:
        raise ValueError("Not enough rows to train anomaly detector (need >=5 rows)")
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    scores = iso.decision_function(X)
    labels = iso.predict(X)  # -1 anomaly, 1 normal
    out = X.copy()
    out['anomaly_label'] = labels
    out['anomaly_score'] = scores
    return iso, out

def train_predictor_with_lags(df, target_col='temp', lags=[1,2,3,6,12], test_size=0.2, random_state=42):
    """
    Tries to create lag features and train a RandomForestRegressor.
    If after applying lags there are too few rows, tries progressively smaller lag sets.
    Returns (model, mse, used_lags)
    """
    df2 = df.copy()
    if target_col not in df2.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    # try progressively smaller lag sets if necessary
    lags_sorted = sorted(set([int(x) for x in lags if int(x) > 0]))
    for n in range(len(lags_sorted), 0, -1):
        try_lags = lags_sorted[:n]
        tmp = df2.copy()
        for lag in try_lags:
            tmp[f'{target_col}_lag_{lag}'] = tmp[target_col].shift(lag)
        tmp = tmp.dropna()
        if tmp.shape[0] < max(20, len(try_lags) * 5):  # require at least some data
            continue
        X = tmp[[c for c in tmp.columns if c.startswith(f'{target_col}_lag_')]]
        y = tmp[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return rf, mse, try_lags
    # if we exit loop, not enough data
    raise ValueError("Not enough rows to train predictor with provided lags. Try fewer/smaller lags or provide more data.")

def predict_next(model, recent_values_dict, target_col='temp', lags=[1,2,3,6,12]):
    """
    Predict next value using model given a dict of lagged recent values.
    """
    x = []
    for lag in lags:
        key = f'{target_col}_lag_{lag}'
        x.append(recent_values_dict.get(key, np.nan))
    x = np.array(x).reshape(1, -1)
    return model.predict(x)[0]

def ensemble_reasoning(df_current_row, anomaly_model=None, predictor=None, predictor_lags=None, threshold_rules=None):
    """
    Combine rules + anomaly + predictor into human-readable commentary.
    df_current_row: Pandas Series with features (including lag columns if predictor is used)
    predictor_lags: list of lags used for predictor (so we can pick lag cols)
    """
    comments = []
    # Rule-based thresholds
    if threshold_rules:
        for k, thresh in threshold_rules.items():
            if k in df_current_row.index:
                try:
                    val = float(df_current_row[k])
                    if k == 'temp' and val >= thresh:
                        comments.append(f"High temperature: {val:.1f}°C ≥ threshold {thresh}°C")
                    elif k == 'ozone' and val <= thresh:
                        comments.append(f"Low ozone: {val:.3f} ≤ threshold {thresh}")
                except Exception:
                    pass
    # Anomaly model
    if anomaly_model is not None:
        # choose intersection of model features and current row
        try:
            # anomaly model expects same features it was trained on
            # we try to create a DataFrame with model input columns
            # If the model has no attribute feature_names_in_, we'll pass all numerics
            try:
                model_features = list(anomaly_model.feature_names_in_)
            except Exception:
                model_features = [c for c in df_current_row.index if pd.api.types.is_numeric_dtype(type(df_current_row[c]))]
            feats = {f: df_current_row.get(f, np.nan) for f in model_features}
            feats_df = pd.DataFrame([feats])
            pred = anomaly_model.predict(feats_df)[0]
            score = anomaly_model.decision_function(feats_df)[0]
            if pred == -1:
                comments.append(f"Anomaly detected (score={score:.3f}).")
            else:
                comments.append(f"No anomaly detected by model (score={score:.3f}).")
        except Exception:
            comments.append("Anomaly model could not be evaluated for the current row.")
    # Predictor
    if predictor is not None and predictor_lags:
        try:
            lag_keys = [f"temp_lag_{lag}" for lag in predictor_lags]
            x = [float(df_current_row.get(k, np.nan)) for k in lag_keys]
            if any(np.isnan(x)):
                comments.append("Not enough lag values present to run predictor for current row.")
            else:
                pred_val = predictor.predict(np.array(x).reshape(1, -1))[0]
                comments.append(f"Predictor forecast next temp ≈ {pred_val:.2f}")
        except Exception:
            comments.append("Predictor could not produce a forecast for the current row.")
    if not comments:
        comments.append("No notable events detected.")
    return "\n".join(comments)
