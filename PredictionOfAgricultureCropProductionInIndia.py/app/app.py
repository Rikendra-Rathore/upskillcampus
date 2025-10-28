# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "production_predictor.joblib"
META_PATH  = ROOT / "models" / "meta.json"
DATA_PATH  = ROOT / "data" / "interim" / "agri_combined.csv"

st.set_page_config(page_title="Crop Production Forecast (India)", page_icon="🌾", layout="wide")
st.title("🌾 Crop Production Forecast (India)")

# ---------------- Loaders ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    default = {"feature_cols": ['year','prod_prev1','prod_prev2','prod_ma2','prod_delta','crop','season']}
    if META_PATH.exists():
        try:
            with open(META_PATH, "r") as f:
                m = json.load(f)
                if "feature_cols" in m and isinstance(m["feature_cols"], list):
                    return m
        except Exception:
            pass
    return default

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            # Normalize types
            for c in ['year','production']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            for c in ['crop','season']:
                if c in df.columns:
                    df[c] = df[c].astype(str).str.strip()
                    df.loc[df[c].isin(['nan','None','NaN']), c] = np.nan
            keep = [c for c in ['crop','season','year','production'] if c in df.columns]
            return df[keep].copy()
        except Exception:
            pass
    return pd.DataFrame()

def get_last_two(df_all, crop, season, year):
    """Return prod_prev1 (year-1), prod_prev2 (year-2) for crop/season."""
    if df_all.empty:
        return None, None
    dfc = df_all.copy()
    dfc = dfc[dfc['crop'].astype(str).str.lower() == str(crop).lower()]
    if 'season' in dfc.columns and season:
        dfc = dfc[dfc['season'].astype(str).str.lower() == season.lower()]
    y1, y2 = year - 1, year - 2
    p1 = dfc.loc[dfc['year'] == y1, 'production']
    p2 = dfc.loc[dfc['year'] == y2, 'production']
    v1 = float(p1.iloc[0]) if not p1.empty else None
    v2 = float(p2.iloc[0]) if not p2.empty else None
    return v1, v2

def make_input_row(crop, season, year, prev1, prev2, feature_cols):
    """Build a single-row DataFrame in the exact feature_cols order."""
    if prev1 is not None and prev2 is not None and prev1 > 0 and prev2 > 0:
        prod_ma2  = (prev1 + prev2) / 2.0
        prod_delta = (prev1 - prev2)
    else:
        prod_ma2  = prev1 if prev1 is not None else np.nan
        prod_delta = np.nan

    row = {
        "year": int(year),
        "prod_prev1": float(prev1) if prev1 is not None else np.nan,
        "prod_prev2": float(prev2) if prev2 is not None else np.nan,
        "prod_ma2": float(prod_ma2) if not pd.isna(prod_ma2) else np.nan,
        "prod_delta": float(prod_delta) if not pd.isna(prod_delta) else np.nan,
        "crop": crop,
        "season": season if season else np.nan,
    }
    X = pd.DataFrame([row])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]

# ---------------- Load resources ----------------
with st.sidebar:
    st.markdown("Paths (debug):")
    st.code(f"ROOT: {ROOT}\nMODEL: {MODEL_PATH.exists()}\nMETA: {META_PATH.exists()}\nDATA: {DATA_PATH.exists()}")

model_loaded = False
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model load failed: {e}\nTrain+save model from notebook first (models/production_predictor.joblib).")

meta = load_meta()
feature_cols = meta.get("feature_cols", ['year','prod_prev1','prod_prev2','prod_ma2','prod_delta','crop','season'])
df_all = load_data()

# Prepare options
if not df_all.empty and 'crop' in df_all.columns:
    crop_options = sorted(df_all['crop'].dropna().unique().tolist())
else:
    crop_options = ["Wheat", "Rice", "Maize", "Sugarcane"]

season_options = ["—", "Kharif", "Rabi"]
has_season = (not df_all.empty) and ('season' in df_all.columns) and (df_all['season'].notna().any())
if not has_season:
    season_options = ["—"]

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Single prediction", "Evaluate (backtest)", "Batch forecast"])

# Tab 1: Single prediction
with tab1:
    st.subheader("Single prediction")
    col1, col2, col3 = st.columns([1,1,1])
    crop = col1.selectbox("Crop", options=crop_options, index=0)
    season = col2.selectbox("Season (optional)", options=season_options, index=0)
    season_val = None if season == "—" else season

    if not df_all.empty and 'year' in df_all.columns:
        max_year = int(pd.to_numeric(df_all['year'], errors='coerce').dropna().max())
        default_year = max_year + 1
    else:
        default_year = 2012
    year = col3.number_input("Forecast Year", min_value=1900, max_value=2100, value=default_year, step=1)

    st.markdown("Provide last two years' production (Tons). You can autofill if present in dataset.")
    autofill = st.checkbox("Autofill last 2 years from dataset", value=True)

    if autofill:
        v1, v2 = get_last_two(df_all, crop, season_val, int(year))
        default_p1 = v1 if v1 is not None else 0.0
        default_p2 = v2 if v2 is not None else 0.0
    else:
        default_p1, default_p2 = 0.0, 0.0

    c1, c2 = st.columns(2)
    prod_prev1 = c1.number_input("Last year production (prod_prev1) [Tons]", min_value=0.0, value=float(default_p1), step=1.0)
    prod_prev2 = c2.number_input("2 years ago production (prod_prev2) [Tons] (optional)", min_value=0.0, value=float(default_p2), step=1.0)

    # Show actual if evaluating a known year
    actual = None
    if not df_all.empty and 'year' in df_all.columns and 'production' in df_all.columns:
        q = (df_all['crop'].astype(str).str.lower() == str(crop).lower()) & (df_all['year'] == int(year))
        if has_season and season_val:
            q = q & (df_all['season'].astype(str).str.lower() == season_val.lower())
        s = df_all.loc[q, 'production']
        if not s.empty:
            actual = float(s.iloc[0])

    if st.button("Predict", type="primary", use_container_width=True, key="predict_single"):
        if not model_loaded:
            st.warning("Model not loaded.")
        else:
            X = make_input_row(crop, season_val, int(year), prod_prev1, prod_prev2, feature_cols)
            try:
                pred = float(model.predict(X)[0])
                st.success(f"Estimated production: {pred:,.2f} Tons")
                if actual is not None:
                    st.info(f"Actual production (dataset): {actual:,.2f} Tons")
                with st.expander("Model input (debug)"):
                    st.dataframe(X)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Tab 2: Evaluate (backtest)
with tab2:
    st.subheader("Evaluate on historical years (backtest)")
    if df_all.empty:
        st.warning("Dataset not loaded. Keep agri_combined.csv under data/interim to use evaluation.")
    else:
        e1, e2 = st.columns(2)
        eval_crop = e1.selectbox("Crop", options=crop_options, index=0, key="eval_crop")
        eval_season = e2.selectbox("Season (optional)", options=season_options, index=0, key="eval_season")
        eval_season_val = None if eval_season == "—" else eval_season

        # Available years for this series
        dfc = df_all[df_all['crop'].astype(str).str.lower() == eval_crop.lower()].copy()
        if has_season and eval_season_val:
            dfc = dfc[dfc['season'].astype(str).str.lower() == eval_season_val.lower()]
        yrs = sorted(pd.to_numeric(dfc.get('year', pd.Series([])), errors='coerce').dropna().unique().tolist())
        if len(yrs) < 3:
            st.warning("Not enough years found for this crop/season to evaluate (need ≥ 3). Try another crop.")
        else:
            s_min, s_max = yrs[0], yrs[-1]
            st.caption(f"Available years in dataset for this series: {yrs}")
            r1, r2 = st.columns(2)
            start = r1.slider("Evaluation start year", min_value=int(s_min), max_value=int(s_max), value=int(max(s_min, s_max-4)))
            end   = r2.slider("Evaluation end year",   min_value=int(s_min), max_value=int(s_max), value=int(s_max))
            if start >= end:
                st.warning("Start year must be < end year.")
            else:
                if st.button("Run backtest", type="primary", use_container_width=True, key="run_backtest"):
                    rows = []
                    for y in range(int(start), int(end)+1):
                        p1, p2 = get_last_two(dfc, eval_crop, eval_season_val, y)
                        X = make_input_row(eval_crop, eval_season_val, y, p1, p2, feature_cols)
                        try:
                            yhat = float(model.predict(X)[0]) if model_loaded else np.nan
                        except Exception:
                            yhat = np.nan
                        actual_y = dfc.loc[dfc['year'] == y, 'production']
                        actual_y = float(actual_y.iloc[0]) if not actual_y.empty else np.nan
                        rows.append({"year": y, "actual": actual_y, "pred": yhat,
                                     "prod_prev1": p1, "prod_prev2": p2})
                    res = pd.DataFrame(rows).dropna(subset=['actual'])
                    if res.empty:
                        st.warning("No actual values found in selected range.")
                    else:
                        res['abs_err'] = (res['pred'] - res['actual']).abs()
                        res['ape_%'] = (res['abs_err'] / res['actual'].replace(0, np.nan)) * 100.0
                        mae  = res['abs_err'].mean()
                        rmse = np.sqrt(((res['pred'] - res['actual'])**2).mean())
                        mape = res['ape_%'].mean()
                        # R^2 (guard against constant target)
                        if res['actual'].nunique() > 1:
                            from sklearn.metrics import r2_score
                            r2 = r2_score(res['actual'], res['pred'])
                        else:
                            r2 = np.nan

                        st.metric("MAE (Tons)", f"{mae:,.2f}")
                        cA, cB, cC = st.columns(3)
                        cA.metric("RMSE (Tons)", f"{rmse:,.2f}")
                        cB.metric("MAPE (%)", f"{mape:,.2f}")
                        cC.metric("R²", f"{r2:.3f}" if not pd.isna(r2) else "NA")

                        st.line_chart(res.set_index('year')[['actual','pred']])
                        st.dataframe(res, use_container_width=True)

                        csv = res.to_csv(index=False).encode('utf-8')
                        st.download_button("Download results CSV", data=csv, file_name=f"backtest_{eval_crop}_{start}_{end}.csv", mime="text/csv")

# Tab 3: Batch forecast (all crops)
with tab3:
    st.subheader("Batch forecast for all crops")
    if df_all.empty:
        st.warning("Dataset not loaded. Keep agri_combined.csv under data/interim to use batch forecast.")
    else:
        if not df_all.empty and 'year' in df_all.columns:
            max_year = int(pd.to_numeric(df_all['year'], errors='coerce').dropna().max())
            default_target = max_year + 1
        else:
            default_target = 2012
        target_year = st.number_input("Target year to forecast", min_value=1900, max_value=2100, value=int(default_target), step=1)

        run = st.button("Run batch forecast", type="primary", use_container_width=True)
        if run:
            rows = []
            crops = sorted(df_all['crop'].dropna().unique().tolist())
            for c in crops:
                if has_season:
                    # If season exists, forecast both seasons separately
                    seasons = sorted(df_all.loc[df_all['crop'] == c, 'season'].dropna().unique().tolist())
                    if not seasons:
                        seasons = [None]
                else:
                    seasons = [None]

                for s in seasons:
                    p1, p2 = get_last_two(df_all, c, s, int(target_year))
                    X = make_input_row(c, s, int(target_year), p1, p2, feature_cols)
                    try:
                        yhat = float(model.predict(X)[0]) if model_loaded else np.nan
                    except Exception:
                        yhat = np.nan
                    rows.append({
                        "crop": c,
                        "season": s if s else None,
                        "year": int(target_year),
                        "prod_prev1": p1,
                        "prod_prev2": p2,
                        "prediction_tons": yhat
                    })
            out = pd.DataFrame(rows).sort_values(['crop','season']).reset_index(drop=True)
            st.dataframe(out, use_container_width=True)

            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download batch forecast CSV", data=csv, file_name=f"forecast_all_{target_year}.csv", mime="text/csv")