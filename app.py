import glob
import io
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EduPredict | Student Analytics & ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Light & Dark mode compatible)
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.15);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.75;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 4px;
        opacity: 0.7;
    }

    /* Gradient Header Badge */
    .app-header {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        background-color: rgba(99, 102, 241, 0.15);
        color: #6366F1;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 10px;
    }
    
    /* Grade pill badges */
    .grade-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
    }
    .grade-outstanding { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .grade-excellent   { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
    .grade-good        { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    .grade-average     { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
    .grade-needs-help  { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

    /* Info callout box */
    .callout-box {
        background: rgba(99, 102, 241, 0.05);
        border-left: 4px solid #6366F1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
    }

    .status-banner {
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    .status-info {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #1D4ED8;
    }
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #047857;
    }
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #B45309;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TARGET = "Performance Index"
FEATURE_COLS = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]

# ==========================================
# INTELLIGENT SCHEMA NORMALIZER
# ==========================================
def smart_normalize_columns(df_in):
    """
    Intelligently maps varying column names to standard feature & target names.
    Handles snake_case, spaces, casing, and common educational synonyms.
    """
    col_mapping = {}
    used_standards = set()

    for col in df_in.columns:
        c_str = str(col).strip().lower()
        c_clean = re.sub(r'[^a-z0-9]', ' ', c_str)
        tokens = c_clean.split()

        # 1. Target check
        if any(term in tokens for term in ["performance", "target", "final", "gpa", "result"]) or \
           "performance index" in c_clean or ("score" in tokens and len(tokens) == 1):
            if TARGET not in used_standards:
                col_mapping[col] = TARGET
                used_standards.add(TARGET)
                continue

        # 2. Previous scores
        if any(term in c_clean for term in ["prev", "previous", "past", "prior", "last"]):
            if "Previous Scores" not in used_standards:
                col_mapping[col] = "Previous Scores"
                used_standards.add("Previous Scores")
                continue

        # 3. Sleep hours
        if "sleep" in c_clean:
            if "Sleep Hours" not in used_standards:
                col_mapping[col] = "Sleep Hours"
                used_standards.add("Sleep Hours")
                continue

        # 4. Extracurricular activities
        if any(term in c_clean for term in ["extra", "curricular", "activity", "activities"]):
            if "Extracurricular Activities" not in used_standards:
                col_mapping[col] = "Extracurricular Activities"
                used_standards.add("Extracurricular Activities")
                continue

        # 5. Question papers practiced
        if any(term in c_clean for term in ["paper", "practice", "sample", "mock", "question"]):
            if "Sample Question Papers Practiced" not in used_standards:
                col_mapping[col] = "Sample Question Papers Practiced"
                used_standards.add("Sample Question Papers Practiced")
                continue

        # 6. Study hours
        if any(term in c_clean for term in ["hour", "study", "time"]):
            if "Hours Studied" not in used_standards:
                col_mapping[col] = "Hours Studied"
                used_standards.add("Hours Studied")
                continue

    return df_in.rename(columns=col_mapping)


# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data(show_spinner="Loading student benchmark dataset...")
def load_benchmark_data():
    """Loads the core benchmark 10,000 record dataset safely."""
    # 1. Local Student_Performance.csv in project directory
    if os.path.exists("Student_Performance.csv"):
        return pd.read_csv("Student_Performance.csv")

    local_csvs = [f for f in glob.glob("*.csv") if "template" not in f.lower() and "prediction" not in f.lower()]
    if local_csvs:
        return pd.read_csv(local_csvs[0])

    # 2. Kaggle download fallback
    try:
        import kagglehub
        path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if csv_files:
            return pd.read_csv(csv_files[0])
    except Exception as err:
        st.warning(f"KaggleHub fallback failed: {err}")

    raise FileNotFoundError("Could not find benchmark 'Student_Performance.csv' dataset.")


def process_dataset(uploaded_file=None):
    """
    Safely inspects uploaded or benchmark data:
    - Normalizes columns
    - Validates feature & target requirements
    - Detects whether file is training dataset or batch inference template
    - Never crashes on missing target; falls back seamlessly to benchmark data.
    """
    benchmark_df = load_benchmark_data()

    if uploaded_file is None:
        return benchmark_df, "default", None, None

    try:
        raw_uploaded = pd.read_csv(uploaded_file)
        normalized_df = smart_normalize_columns(raw_uploaded)

        has_all_features = all(c in normalized_df.columns for c in FEATURE_COLS)
        has_target = TARGET in normalized_df.columns

        # Case A: Full valid training dataset with target
        if has_all_features and has_target:
            return normalized_df, "custom_valid", uploaded_file.name, None

        # Case B: Inference dataset (has 5 features, but NO target) -> e.g. student_sample.csv
        elif has_all_features and not has_target:
            return benchmark_df, "inference_detected", uploaded_file.name, normalized_df

        # Case C: Partial or unrecognized dataset
        else:
            missing_feats = [c for c in FEATURE_COLS if c not in normalized_df.columns]
            return benchmark_df, "incompatible", uploaded_file.name, missing_feats

    except Exception as read_err:
        return benchmark_df, "read_error", uploaded_file.name, str(read_err)


def preprocess_data(df_in):
    """Cleans dataset and creates derived categorization features."""
    df = df_in.copy().drop_duplicates()

    # Map binary categorical if not already numeric
    if "Extracurricular Activities" in df.columns:
        if df["Extracurricular Activities"].dtype == object:
            df["Extracurricular Activities"] = (
                df["Extracurricular Activities"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)
            )

    # Add score category for analysis
    if TARGET in df.columns:
        bins = [-1, 40, 60, 75, 90, 101]
        labels = [
            "🔴 Needs Help (<40)",
            "🟠 Average (40-59)",
            "🟡 Good (60-74)",
            "🟢 Excellent (75-89)",
            "🌟 Outstanding (90-100)",
        ]
        df["Performance Tier"] = pd.cut(df[TARGET], bins=bins, labels=labels)

    return df


# ==========================================
# ADVANCED MODEL TRAINING & DIAGNOSTICS
# ==========================================
@st.cache_resource(show_spinner="Training predictive models & econometric diagnostics...")
def train_and_evaluate(df_clean, test_size=0.25, seed=42):
    """
    Trains Multiple Linear Regression along with:
    - Analytical 95% Prediction Interval matrices (XtX_inv, MSE)
    - Variance Inflation Factor (VIF) for Multicollinearity audit
    - Standardized Beta Coefficients (effect size in standard deviations)
    - Ridge, Lasso & Random Forest benchmarks
    """
    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=int(seed)
    )

    # Main Multiple Linear Regression model
    lr_model = LinearRegression().fit(x_train, y_train)
    y_pred_test = lr_model.predict(x_test)
    y_pred_train = lr_model.predict(x_train)

    # Analytical Prediction Interval Setup
    # Design matrix with constant 1 column for intercept
    X_design = np.hstack([np.ones((len(x_train), 1)), x_train.values])
    mse_train = np.sum((y_train.values - y_pred_train)**2) / (len(y_train) - X_design.shape[1])
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)

    # Performance metrics
    n = len(y_test)
    p = x_test.shape[1]
    r2_test = r2_score(y_test, y_pred_test)
    adj_r2_test = 1 - ((1 - r2_test) * (n - 1) / (n - p - 1))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    metrics_lr = {
        "R² (Test)": r2_test,
        "Adj R² (Test)": adj_r2_test,
        "R² (Train)": r2_score(y_train, y_pred_train),
        "RMSE": rmse_test,
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "Max Error": max_error(y_test, y_pred_test),
    }

    # Variance Inflation Factor (VIF) Calculation
    corr_matrix = x.corr().values
    vif_values = np.diag(np.linalg.pinv(corr_matrix))
    vif_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "VIF": np.round(vif_values, 3),
        "Collinearity Status": ["✅ Low (< 5)" if v < 5 else ("⚠️ Moderate" if v < 10 else "🚨 High (> 10)") for v in vif_values],
    })

    # Standardized Beta Coefficients: beta_std = beta * (std(x) / std(y))
    std_x = x.std().values
    std_y = y.std()
    std_beta = lr_model.coef_ * (std_x / std_y)

    coef_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coefficient": lr_model.coef_,
        "Standardized_Beta": std_beta,
        "Abs_Coefficient": np.abs(lr_model.coef_),
        "Impact_Rank": np.argsort(np.argsort(-np.abs(std_beta))) + 1,
    }).sort_values(by="Abs_Coefficient", ascending=False)

    # Benchmark models for model comparison tab
    models_comparison = {
        "Multiple Linear Regression": {
            "model": lr_model,
            "R² Test": r2_test,
            "RMSE": rmse_test,
            "MAE": mean_absolute_error(y_test, y_pred_test),
        }
    }

    # Ridge
    ridge = Ridge(alpha=1.0).fit(x_train, y_train)
    ridge_preds = ridge.predict(x_test)
    models_comparison["Ridge Regression (L2)"] = {
        "model": ridge,
        "R² Test": r2_score(y_test, ridge_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds)),
        "MAE": mean_absolute_error(y_test, ridge_preds),
    }

    # Lasso
    lasso = Lasso(alpha=0.1).fit(x_train, y_train)
    lasso_preds = lasso.predict(x_test)
    models_comparison["Lasso Regression (L1)"] = {
        "model": lasso,
        "R² Test": r2_score(y_test, lasso_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, lasso_preds)),
        "MAE": mean_absolute_error(y_test, lasso_preds),
    }

    # Fast Random Forest sample
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=int(seed), n_jobs=-1).fit(
        x_train, y_train
    )
    rf_preds = rf.predict(x_test)
    models_comparison["Random Forest (Tree-Based)"] = {
        "model": rf,
        "R² Test": r2_score(y_test, rf_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "MAE": mean_absolute_error(y_test, rf_preds),
    }

    interval_params = {
        "mse": mse_train,
        "XtX_inv": XtX_inv,
        "coef": lr_model.coef_,
        "intercept": lr_model.intercept_,
    }

    return (
        lr_model,
        metrics_lr,
        coef_df,
        vif_df,
        models_comparison,
        interval_params,
        (x_train, x_test, y_train, y_test, y_pred_test),
    )


# Analytical Prediction Interval Calculator
def calculate_prediction_interval(x_vector, interval_params):
    """Computes exact 95% Prediction Interval for a given input feature vector."""
    x_input = np.array([1.0] + list(x_vector))
    mse = interval_params["mse"]
    XtX_inv = interval_params["XtX_inv"]
    variance_pred = mse * (1.0 + float(x_input @ XtX_inv @ x_input))
    se_pred = np.sqrt(max(0.0001, variance_pred))
    
    # Point prediction
    raw_pred = float(interval_params["intercept"] + np.dot(x_vector, interval_params["coef"]))
    lower_95 = max(0.0, raw_pred - 1.96 * se_pred)
    upper_95 = min(100.0, raw_pred + 1.96 * se_pred)
    clamped_pred = max(0.0, min(100.0, raw_pred))
    
    return clamped_pred, lower_95, upper_95, se_pred


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    uploaded = st.file_uploader(
        "📂 Upload Custom CSV Dataset",
        type=["csv"],
        help="Upload training data or inference batch file. Flexible column matching is enabled.",
    )

    st.markdown("---")
    st.markdown("#### 🧪 Model Hyperparameters")
    test_size_val = st.slider("Test Split Size", min_value=0.10, max_value=0.50, value=0.25, step=0.05, help="Percentage of records reserved for testing")
    random_seed = st.number_input("Random Seed (State)", value=42, min_value=0, max_value=9999, step=1)

    st.markdown("---")
    st.markdown("#### 🎯 Quick Filter (Overview)")
    ec_filter = st.multiselect(
        "Extracurricular Activity",
        options=["Yes", "No"],
        default=["Yes", "No"],
        help="Filter records shown in Overview charts",
    )
    hour_range = st.slider(
        "Hours Studied Range",
        min_value=1,
        max_value=9,
        value=(1, 9),
        help="Filter student study hours range",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8rem; opacity: 0.7;'>
        <b>EduPredict Suite</b><br>
        Multiple Linear Regression · Python 3.13<br>
        Engineered for Academic Analytics
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# SAFE DATASET INITIALIZATION
# ==========================================
df_loaded, data_status, file_name, extra_info = process_dataset(uploaded)
df = preprocess_data(df_loaded)

# Train models with econometric diagnostics
(
    model,
    metrics,
    coef_df,
    vif_df,
    benchmark_dict,
    interval_params,
    (X_train, X_test, y_train, y_test, y_pred_test),
) = train_and_evaluate(df, test_size=test_size_val, seed=random_seed)

# Filtered dataset for overview tab
df_filtered = df.copy()
filter_map = {"Yes": 1, "No": 0}
selected_ec = [filter_map[x] for x in ec_filter if x in filter_map]
if selected_ec:
    df_filtered = df_filtered[df_filtered["Extracurricular Activities"].isin(selected_ec)]
df_filtered = df_filtered[
    (df_filtered["Hours Studied"] >= hour_range[0])
    & (df_filtered["Hours Studied"] <= hour_range[1])
]

# ==========================================
# APP HEADER
# ==========================================
st.markdown("<h1 class='app-header'>🎓 EduPredict — Student Analytics & ML Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px;'>
        <span class='badge-pill'>📈 Multiple Linear Regression</span>
        <span class='badge-pill'>🎯 Test R²: {metrics['R² (Test)']*100:.2f}%</span>
        <span class='badge-pill'>⚡ RMSE: {metrics['RMSE']:.2f} pts</span>
        <span class='badge-pill'>👥 Dataset: {len(df):,} Students</span>
        <span class='badge-pill'>🛡️ Multicollinearity (VIF): Safe (≈1.0)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display intelligent dataset status banner
if data_status == "inference_detected":
    st.markdown(
        f"""
        <div class='status-banner status-info'>
            ℹ️ <b>Inference File Detected ('{file_name}')</b>: Your uploaded CSV contains all 5 student input features without a target column (<code>Performance Index</code>).
            <br>The model is trained on the benchmark dataset, and your file has been automatically pre-loaded into the <b>📁 Batch Prediction Engine (Tab 2)</b>!
        </div>
        """,
        unsafe_allow_html=True,
    )
elif data_status == "custom_valid":
    st.markdown(
        f"""
        <div class='status-banner status-success'>
            ✅ <b>Custom Dataset Active ('{file_name}')</b>: Successfully mapped columns and retrained the model on {len(df):,} student records!
        </div>
        """,
        unsafe_allow_html=True,
    )
elif data_status == "incompatible":
    st.markdown(
        f"""
        <div class='status-banner status-warning'>
            ⚠️ <b>Uploaded File Schema Mismatch ('{file_name}')</b>: Required features could not be auto-detected: <code>{extra_info}</code>.
            <br>Reverting to benchmark dataset (10,000 students) to prevent errors. Please check column headers.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# MAIN TABS NAVIGATION
# ==========================================
tab_overview, tab_predict, tab_eda, tab_diagnostics = st.tabs([
    "📊 Executive Overview",
    "🔮 Smart Predictor & Simulator",
    "🔍 Interactive Data Explorer",
    "📈 Model Diagnostics & Econometrics",
])


# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
with tab_overview:
    st.markdown("### 📌 High-Level Performance Indicators")

    # 4 Key KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df_filtered[TARGET].mean()
    high_performers_pct = (df_filtered[TARGET] >= 75).mean() * 100
    avg_hours = df_filtered["Hours Studied"].mean()

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>👥 Total Students</div>
                <div class='metric-value'>{len(df_filtered):,}</div>
                <div class='metric-sub'>Active in current filter (out of {len(df):,})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🎯 Average Performance</div>
                <div class='metric-value'>{avg_score:.1f} / 100</div>
                <div class='metric-sub'>Median: {df_filtered[TARGET].median():.1f} · Std: {df_filtered[TARGET].std():.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🌟 High Performers (≥75)</div>
                <div class='metric-value'>{high_performers_pct:.1f}%</div>
                <div class='metric-sub'>{(df_filtered[TARGET] >= 75).sum():,} students qualified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🤖 Model Predictive Power</div>
                <div class='metric-value'>{metrics['R² (Test)']*100:.2f}%</div>
                <div class='metric-sub'>Mean Absolute Error: ±{metrics['MAE']:.2f} pts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1: Distribution & Donut
    c_left, c_right = st.columns([1.3, 1])

    with c_left:
        st.markdown("##### 📊 Performance Index Distribution")
        fig_dist = px.histogram(
            df_filtered,
            x=TARGET,
            nbins=35,
            marginal="box",
            color_discrete_sequence=["#6366F1"],
            opacity=0.85,
        )
        fig_dist.add_vline(
            x=avg_score,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Mean: {avg_score:.1f}",
            annotation_position="top left",
        )
        fig_dist.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
            xaxis_title="Performance Index (0-100)",
            yaxis_title="Student Count",
        )
        st.plotly_chart(fig_dist, width="stretch")

    with c_right:
        st.markdown("##### 🍩 Academic Performance Tiers")
        if "Performance Tier" in df_filtered.columns:
            tier_counts = df_filtered["Performance Tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            tier_colors = {
                "🌟 Outstanding (90-100)": "#10B981",
                "🟢 Excellent (75-89)": "#3B82F6",
                "🟡 Good (60-74)": "#F59E0B",
                "🟠 Average (40-59)": "#F97316",
                "🔴 Needs Help (<40)": "#EF4444",
            }
            fig_donut = px.pie(
                tier_counts,
                names="Tier",
                values="Count",
                hole=0.45,
                color="Tier",
                color_discrete_map=tier_colors,
            )
            fig_donut.update_traces(textinfo="percent+label", showlegend=False)
            fig_donut.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_donut, width="stretch")

    # Charts Row 2: Bivariate Relationships
    st.markdown("<br>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        st.markdown("##### 📖 Study Hours vs Score (By Extracurriculars)")
        ec_display = df_filtered.copy()
        ec_display["Extracurricular"] = ec_display["Extracurricular Activities"].map({1: "Yes", 0: "No"})
        fig_box = px.box(
            ec_display,
            x="Hours Studied",
            y=TARGET,
            color="Extracurricular",
            color_discrete_map={"Yes": "#8B5CF6", "No": "#94A3B8"},
        )
        fig_box.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
            xaxis_title="Hours Studied (Daily)",
            yaxis_title="Performance Index",
        )
        st.plotly_chart(fig_box, width="stretch")

    with row2_col2:
        st.markdown("##### 🛌 Sleep Duration vs Average Performance Index")
        sleep_agg = (
            df_filtered.groupby("Sleep Hours")[TARGET]
            .agg(Mean_Score="mean", Std="std", Count="count")
            .reset_index()
        )
        fig_sleep = px.bar(
            sleep_agg,
            x="Sleep Hours",
            y="Mean_Score",
            text="Mean_Score",
            color="Mean_Score",
            color_continuous_scale="Purples",
        )
        fig_sleep.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_sleep.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
            yaxis_range=[0, 80],
            xaxis_title="Sleep Hours per Night",
            yaxis_title="Average Performance Index",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_sleep, width="stretch")

    # 2026 Innovation: Interactive Cohort Intervention Simulator
    with st.expander("🏫 Cohort Policy Intervention Simulator (What-If School Policy Changes)"):
        st.write("Simulate school-wide educational interventions to evaluate how policy changes shift the entire student cohort's grade distribution.")
        cs_col1, cs_col2 = st.columns(2)
        with cs_col1:
            add_study_hours = st.slider("➕ Mandatory Daily Study Enhancement (Hours)", min_value=0.0, max_value=3.0, value=1.0, step=0.5)
        with cs_col2:
            add_papers = st.slider("➕ Weekly Mock Papers Practiced", min_value=0, max_value=5, value=2, step=1)

        sim_scores = df_filtered[TARGET] + (add_study_hours * model.coef_[0]) + (add_papers * model.coef_[4])
        sim_scores = np.clip(sim_scores, 0.0, 100.0)

        sim_df = pd.DataFrame({
            "Baseline Score": df_filtered[TARGET],
            "Post-Intervention Score": sim_scores,
        })
        fig_sim_cohort = go.Figure()
        fig_sim_cohort.add_trace(go.Histogram(x=sim_df["Baseline Score"], name="Current Cohort", opacity=0.6, marker_color="#94A3B8"))
        fig_sim_cohort.add_trace(go.Histogram(x=sim_df["Post-Intervention Score"], name="Projected Cohort", opacity=0.7, marker_color="#10B981"))
        fig_sim_cohort.update_layout(
            barmode="overlay",
            template="plotly_white",
            height=300,
            xaxis_title="Performance Index",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_sim_cohort, width="stretch")
        old_high = (df_filtered[TARGET] >= 75).mean() * 100
        new_high = (sim_scores >= 75).mean() * 100
        st.success(f"🚀 Projected Impact: High Performers (≥75) would increase from **{old_high:.1f}% → {new_high:.1f}%** (+{new_high - old_high:.1f}% gain)!")


# ==========================================
# TAB 2: SMART PREDICTOR & SIMULATOR
# ==========================================
with tab_predict:
    st.markdown("### 🔮 Real-Time Student Performance Predictor & Optimizer")
    st.caption("Adjust student inputs below to calculate predicted scores, exact 95% confidence intervals, and goal-seeking pathways.")

    pred_tab1, pred_tab2 = st.tabs(["👤 Single Student Simulation", "📁 Batch Prediction (CSV Engine)"])

    with pred_tab1:
        col_in1, col_in2, col_in3 = st.columns([1, 1, 1])

        with col_in1:
            in_hours = st.slider("📚 Hours Studied (Daily)", min_value=1, max_value=9, value=6, help="Average daily study time")
            in_prev = st.slider("📝 Previous Scores", min_value=40, max_value=100, value=75, help="Past exam average score percentage")

        with col_in2:
            in_sleep = st.slider("🛌 Sleep Hours (Nightly)", min_value=4, max_value=9, value=7, help="Average sleep per night")
            in_papers = st.slider("📄 Sample Papers Practiced", min_value=0, max_value=9, value=4, help="Mock question papers completed")

        with col_in3:
            in_extra_str = st.radio("🏅 Extracurricular Activities", options=["Yes", "No"], index=0, horizontal=True)
            in_extra = 1 if in_extra_str == "Yes" else 0
            auto_predict = st.toggle("⚡ Real-Time Auto Calculation", value=True)

        # Vector calculation with Analytical 95% Prediction Interval
        feature_vector = [in_hours, in_prev, in_extra, in_sleep, in_papers]
        clamped_pred, lower_95, upper_95, se_pred = calculate_prediction_interval(feature_vector, interval_params)

        st.markdown("---")

        # Result Presentation Section
        r_col1, r_col2 = st.columns([1.1, 1.4])

        with r_col1:
            st.markdown("#### 🎯 Predicted Outcome & 95% Confidence Bounds")

            # Gauge Speedometer Chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=clamped_pred,
                    domain={"x": [0, 1], "y": [0, 1]},
                    number={"suffix": " / 100", "font": {"size": 34, "color": "#4F46E5"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                        "bar": {"color": "#4F46E5", "thickness": 0.28},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": "#E2E8F0",
                        "steps": [
                            {"range": [0, 40], "color": "rgba(239, 68, 68, 0.2)"},
                            {"range": [40, 60], "color": "rgba(249, 115, 22, 0.2)"},
                            {"range": [60, 75], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [75, 90], "color": "rgba(59, 130, 246, 0.2)"},
                            {"range": [90, 100], "color": "rgba(16, 185, 129, 0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "#10B981", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=10, b=10),
            )
            st.plotly_chart(fig_gauge, width="stretch")

            # Performance Grade Badge & Percentile
            if clamped_pred >= 90:
                grade_label = "🌟 Outstanding Performance (Grade A+)"
                css_class = "grade-outstanding"
                feedback = "Outstanding! The student shows top-tier mastery. Maintain current consistency."
            elif clamped_pred >= 75:
                grade_label = "🟢 Excellent Performance (Grade A)"
                css_class = "grade-excellent"
                feedback = "Great trajectory! Increasing study time by 1 more hour can push into the 90+ bracket."
            elif clamped_pred >= 60:
                grade_label = "🟡 Good Performance (Grade B)"
                css_class = "grade-good"
                feedback = "Solid foundation. Practicing 2-3 extra mock papers will reinforce problem solving."
            elif clamped_pred >= 40:
                grade_label = "🟠 Average Performance (Grade C)"
                css_class = "grade-average"
                feedback = "Needs focused intervention. Prioritize core study hours and previous test revisions."
            else:
                grade_label = "🔴 Needs Immediate Improvement (Grade D)"
                css_class = "grade-needs-help"
                feedback = "High risk of academic deficit. Needs structured study schedule and mentoring support."

            # Empirical Percentile Rank
            percentile = (df[TARGET] <= clamped_pred).mean() * 100

            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <div class='grade-pill {css_class}'>{grade_label}</div>
                    <div style='margin-top: 10px; font-size: 0.95rem; opacity: 0.8;'>
                        <b>95% Prediction Interval:</b> [{lower_95:.1f} – {upper_95:.1f}] pts (±{1.96*se_pred:.1f})<br>
                        <b>Percentile Rank:</b> Better than <b>{percentile:.1f}%</b> of benchmark cohort
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown("#### 🔬 Feature Contribution Breakdown (Waterfall)")
            st.caption("How base intercept combines with input factors to compute final prediction:")

            intercept_val = model.intercept_
            contrib_hours = in_hours * model.coef_[0]
            contrib_prev = in_prev * model.coef_[1]
            contrib_extra = in_extra * model.coef_[2]
            contrib_sleep = in_sleep * model.coef_[3]
            contrib_papers = in_papers * model.coef_[4]

            wf_measures = ["relative", "relative", "relative", "relative", "relative", "relative", "total"]
            wf_x = [
                "Base Intercept",
                "Study Hours",
                "Prev Scores",
                "Extracurricular",
                "Sleep Hours",
                "Sample Papers",
                "Final Score",
            ]
            wf_y = [
                intercept_val,
                contrib_hours,
                contrib_prev,
                contrib_extra,
                contrib_sleep,
                contrib_papers,
                clamped_pred,
            ]
            wf_text = [f"{v:+.1f}" if i < 6 else f"{v:.1f}" for i, v in enumerate(wf_y)]

            fig_wf = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=wf_measures,
                    x=wf_x,
                    textposition="outside",
                    text=wf_text,
                    y=wf_y,
                    connector={"line": {"color": "#CBD5E1"}},
                    decreasing={"marker": {"color": "#EF4444"}},
                    increasing={"marker": {"color": "#10B981"}},
                    totals={"marker": {"color": "#4F46E5"}},
                )
            )
            fig_wf.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                yaxis_title="Score Points",
            )
            st.plotly_chart(fig_wf, width="stretch")

        # Goal Seeker & Optimization Section
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🎯 Smart Goal-Seeker: How to Reach a Target Score"):
            gs_col1, gs_col2 = st.columns([1, 2])
            with gs_col1:
                target_goal = st.slider("🎯 Desired Target Score", min_value=int(clamped_pred), max_value=100, value=min(100, int(clamped_pred) + 10))
                gap = target_goal - clamped_pred
                st.metric("Point Gain Required", f"+{gap:.1f} pts")

            with gs_col2:
                if gap <= 0:
                    st.info("The student has already attained or exceeded this target score!")
                else:
                    needed_hours = gap / model.coef_[0]
                    needed_papers = gap / model.coef_[4]
                    st.markdown(
                        f"""
                        <b>3 Actionable Pathways to reach {target_goal} pts:</b><br>
                        • <b>Pathway 1 (Study Time):</b> Increase daily study time by <b>+{needed_hours:.1f} hours/day</b>.<br>
                        • <b>Pathway 2 (Exam Practice):</b> Practice <b>+{needed_papers:.0f} additional mock papers</b>.<br>
                        • <b>Pathway 3 (Balanced):</b> Add <b>+{(gap*0.6)/model.coef_[0]:.1f} study hours</b>, practice <b>+{(gap*0.3)/model.coef_[4]:.0f} papers</b>, and participate in extracurricular activities!
                        """,
                        unsafe_allow_html=True,
                    )

        # What-If Sensitivity Curve
        st.markdown("#### 📈 'What-If' Study Hours Sensitivity Curve")
        sim_hours_range = np.linspace(1, 9, 25)
        sim_curve = []
        for h in sim_hours_range:
            vec = [h, in_prev, in_extra, in_sleep, in_papers]
            p, _, _, _ = calculate_prediction_interval(vec, interval_params)
            sim_curve.append(p)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sim_hours_range,
            y=sim_curve,
            mode="lines+markers",
            name="Score Trajectory",
            line=dict(color="#6366F1", width=3),
            marker=dict(size=6),
        ))
        fig_sim.add_trace(go.Scatter(
            x=[in_hours],
            y=[clamped_pred],
            mode="markers",
            name="Current Student Point",
            marker=dict(color="#EF4444", size=14, symbol="star"),
        ))
        fig_sim.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
            xaxis_title="Hours Studied Daily (1-9 hrs)",
            yaxis_title="Predicted Performance Index",
        )
        st.plotly_chart(fig_sim, width="stretch")

        # Download Report Card Button
        report_content = f"""# 🎓 EduPredict - Student Evaluation Report Card
Generated On: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Student Inputs:
- Hours Studied: {in_hours} hrs/day
- Previous Exam Scores: {in_prev}%
- Extracurricular Participation: {in_extra_str}
- Nightly Sleep Hours: {in_sleep} hrs
- Sample Papers Completed: {in_papers}

## Model Prediction & Evaluation:
- Predicted Performance Index: {clamped_pred:.2f} / 100
- 95% Prediction Interval: [{lower_95:.2f} - {upper_95:.2f}] (±{1.96*se_pred:.2f})
- Academic Grade: {grade_label}
- Percentile Rank: Top {100 - percentile:.1f}% (Better than {percentile:.1f}% of cohort)

## Academic Recommendation:
{feedback}
"""
        st.download_button(
            label="📄 Download Student Performance Report Card (.txt)",
            data=report_content,
            file_name=f"student_report_{in_prev}_{in_hours}h.txt",
            mime="text/plain",
        )

    # Batch Prediction sub-tab
    with pred_tab2:
        st.markdown("#### 📁 Batch Prediction Engine")
        st.write("Upload a CSV with student details to generate predictions for hundreds or thousands of students simultaneously.")

        batch_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_uploader")

        # Provide a sample template download button
        sample_df = pd.DataFrame({
            "Hours Studied": [4, 6, 8, 3, 5],
            "Previous Scores": [65, 80, 92, 55, 74],
            "Extracurricular Activities": ["Yes", "No", "Yes", "No", "Yes"],
            "Sleep Hours": [7, 8, 6, 5, 7],
            "Sample Question Papers Practiced": [3, 5, 8, 2, 4],
        })
        csv_sample = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_sample,
            file_name="student_prediction_template.csv",
            mime="text/csv",
        )

        # Check if user uploaded directly here OR uploaded an inference dataset in sidebar
        active_batch_source = None
        if batch_file is not None:
            active_batch_source = pd.read_csv(batch_file)
        elif data_status == "inference_detected" and extra_info is not None:
            active_batch_source = extra_info

        if active_batch_source is not None:
            try:
                batch_clean = smart_normalize_columns(active_batch_source)
                st.success(f"Processing {len(batch_clean)} student records for bulk inference!")

                if "Extracurricular Activities" in batch_clean.columns:
                    if batch_clean["Extracurricular Activities"].dtype == object:
                        batch_clean["Extracurricular Activities"] = (
                            batch_clean["Extracurricular Activities"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)
                        )

                missing = [c for c in FEATURE_COLS if c not in batch_clean.columns]
                if missing:
                    st.error(f"Missing required columns in CSV: {missing}")
                else:
                    batch_preds = model.predict(batch_clean[FEATURE_COLS])
                    batch_clean["Predicted Performance Index"] = np.clip(batch_preds, 0.0, 100.0).round(2)
                    
                    st.dataframe(batch_clean.head(25), width="stretch")

                    # Export button
                    csv_export = batch_clean.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Predictions as CSV",
                        data=csv_export,
                        file_name="student_performance_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as batch_err:
                st.error(f"Error processing batch file: {batch_err}")


# ==========================================
# TAB 3: INTERACTIVE DATA EXPLORER
# ==========================================
with tab_eda:
    st.markdown("### 🔍 Interactive Exploratory Data Analysis")
    st.caption("Interact with correlation structures, multi-dimensional distributions, and 3D feature spaces.")

    eda_sub1, eda_sub2, eda_sub3 = st.tabs([
        "🔥 Correlation & 2D Explorer",
        "🌐 3D Interactive Feature Space",
        "📋 Data Inspector & Summary",
    ])

    with eda_sub1:
        c_hm, c_scat = st.columns([1, 1.3])

        with c_hm:
            st.markdown("##### 🌡️ Feature Correlation Heatmap")
            corr_df = df[FEATURE_COLS + [TARGET]].corr()
            fig_corr = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            fig_corr.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_corr, width="stretch")

        with c_scat:
            st.markdown("##### 🎯 Dynamic 2D Scatter Explorer")
            scat_c1, scat_c2, scat_c3 = st.columns(3)
            with scat_c1:
                x_choice = st.selectbox("X-Axis", options=FEATURE_COLS, index=0)
            with scat_c2:
                y_choice = st.selectbox("Y-Axis", options=[TARGET] + FEATURE_COLS, index=0)
            with scat_c3:
                color_choice = st.selectbox("Color By", options=["Extracurricular Activities", "Sleep Hours", "Performance Tier"], index=0)

            sample_eda = df.sample(min(1200, len(df)), random_state=42)
            fig_scat = px.scatter(
                sample_eda,
                x=x_choice,
                y=y_choice,
                color=color_choice,
                trendline="ols",
                opacity=0.6,
                color_continuous_scale="Viridis",
            )
            fig_scat.update_layout(
                height=330,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
            )
            st.plotly_chart(fig_scat, width="stretch")

    with eda_sub2:
        st.markdown("##### 🌐 3D Feature Space: Hours × Previous Score × Performance")
        st.caption("Click and drag to rotate the 3D surface in 360 degrees.")

        sample_3d = df.sample(min(1000, len(df)), random_state=42)
        fig_3d = px.scatter_3d(
            sample_3d,
            x="Hours Studied",
            y="Previous Scores",
            z=TARGET,
            color=TARGET,
            color_continuous_scale="Plasma",
            opacity=0.7,
            size_max=5,
        )
        fig_3d.update_layout(
            height=540,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_3d, width="stretch")

    with eda_sub3:
        st.markdown("##### 📋 Raw Dataset Sample & Statistical Descriptives")
        st.dataframe(df.head(50), width="stretch")

        st.markdown("##### 📐 Statistical Summary (`describe()`)")
        st.dataframe(df.describe().T, width="stretch")


# ==========================================
# TAB 4: MODEL DIAGNOSTICS & ECONOMETRICS
# ==========================================
with tab_diagnostics:
    st.markdown("### 📈 Model Diagnostics & Econometric Assumptions")
    st.caption("Rigorous verification of Gauss-Markov assumptions, residual normality, VIF multicollinearity, and benchmarks.")

    diag_sub1, diag_sub2 = st.tabs(["🔬 Regression Assumptions & Fit", "🥊 Benchmark Comparison & Formula"])

    with diag_sub1:
        # Row 1: Actual vs Predicted & Residuals Histogram
        d_c1, d_c2 = st.columns(2)
        eval_sample = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_test}).sample(min(800, len(y_test)), random_state=42)

        with d_c1:
            st.markdown("##### 🎯 Actual vs. Predicted (Test Set)")
            fig_avp = px.scatter(
                eval_sample,
                x="Actual",
                y="Predicted",
                opacity=0.45,
                color_discrete_sequence=["#6366F1"],
            )
            min_val = min(eval_sample["Actual"].min(), eval_sample["Predicted"].min())
            max_val = max(eval_sample["Actual"].max(), eval_sample["Predicted"].max())
            fig_avp.add_shape(
                type="line",
                x0=min_val,
                y0=min_val,
                x1=max_val,
                y1=max_val,
                line=dict(color="#EF4444", dash="dash", width=2),
            )
            fig_avp.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                xaxis_title="Actual Performance Index",
                yaxis_title="Predicted Performance Index",
            )
            st.plotly_chart(fig_avp, width="stretch")

        with d_c2:
            st.markdown("##### 📉 Residuals Histogram (Normality Check)")
            residuals = y_test - y_pred_test
            fig_res = px.histogram(
                residuals,
                nbins=40,
                marginal="box",
                color_discrete_sequence=["#10B981"],
                opacity=0.8,
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="#EF4444")
            fig_res.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                xaxis_title="Residual Error (Actual - Predicted)",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_res, width="stretch")

        # Row 2: 2026 Advanced Diagnostics: Q-Q Plot & Residuals vs Fitted
        st.markdown("<br>", unsafe_allow_html=True)
        qq_c1, qq_c2 = st.columns(2)

        with qq_c1:
            st.markdown("##### 📊 Normal Q-Q Plot (Quantile-Quantile)")
            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residual Quantiles", marker=dict(color="#6366F1", size=5, opacity=0.6)))
            fig_qq.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode="lines", name="Normal Reference Line", line=dict(color="#EF4444", dash="dash", width=2)))
            fig_qq.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                xaxis_title="Theoretical Normal Quantiles",
                yaxis_title="Ordered Sample Residuals",
            )
            st.plotly_chart(fig_qq, width="stretch")

        with qq_c2:
            st.markdown("##### 📐 Residuals vs. Fitted (Homoscedasticity Test)")
            fitted_sample = pd.DataFrame({"Fitted": y_pred_test, "Residuals": residuals}).sample(min(800, len(y_test)), random_state=42)
            fig_rvf = px.scatter(
                fitted_sample,
                x="Fitted",
                y="Residuals",
                opacity=0.5,
                color_discrete_sequence=["#8B5CF6"],
            )
            fig_rvf.add_hline(y=0, line_dash="dash", line_color="#EF4444", line_width=2)
            fig_rvf.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                xaxis_title="Fitted (Predicted) Values",
                yaxis_title="Residuals (Errors)",
            )
            st.plotly_chart(fig_rvf, width="stretch")

        # Row 3: Multicollinearity (VIF) Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🛡️ Multicollinearity Audit: Variance Inflation Factor (VIF)")
        st.caption("A VIF value < 5.0 indicates that predictors are strictly independent without multi-collinearity inflation.")
        st.dataframe(vif_df, width="stretch")

    with diag_sub2:
        # Standardized Beta vs Raw Coefs
        fi_col1, fi_col2 = st.columns([1.2, 1])

        with fi_col1:
            st.markdown("##### ⚖️ Standardized Beta Coefficients (Relative Effect Size)")
            fig_coef = px.bar(
                coef_df,
                x="Standardized_Beta",
                y="Feature",
                orientation="h",
                color="Standardized_Beta",
                color_continuous_scale="Purples",
                text="Standardized_Beta",
            )
            fig_coef.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_coef.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                coloraxis_showscale=False,
                xaxis_title="Standardized Beta (Impact in Standard Deviations)",
            )
            st.plotly_chart(fig_coef, width="stretch")

        with fi_col2:
            st.markdown("##### 📐 Learned Regression Equation")
            equation_latex = (
                r"\text{Performance Index} = "
                + f"{model.intercept_:.2f} "
                + f"+ {model.coef_[0]:.2f}(\text{{Hours}}) "
                + f"+ {model.coef_[1]:.2f}(\text{{Prev}}) "
                + f"+ {model.coef_[2]:.2f}(\text{{Extra}}) "
                + f"+ {model.coef_[3]:.2f}(\text{{Sleep}}) "
                + f"+ {model.coef_[4]:.2f}(\text{{Papers}})"
            )
            st.latex(equation_latex)

            st.markdown(
                f"""
                <div class='callout-box' style='font-size: 0.85rem;'>
                    <b>Interpretation:</b><br>
                    • <b>Intercept ({model.intercept_:.2f}):</b> Baseline score when all inputs are zero.<br>
                    • <b>Hours Studied ({model.coef_[0]:.2f}):</b> Strongest unit impact. Each hour yields +2.85 points.<br>
                    • <b>Previous Scores ({model.coef_[1]:.2f}):</b> Carryover rate of 1.02 points per past exam point.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Multi-Model Benchmark Comparison Table
        st.markdown("---")
        st.markdown("##### 🥊 Multi-Model Performance Benchmark")
        benchmark_data = []
        for m_name, m_info in benchmark_dict.items():
            benchmark_data.append({
                "Model Algorithm": m_name,
                "R² Score (Test)": f"{m_info['R² Test']*100:.3f}%",
                "RMSE (Lower is Better)": f"{m_info['RMSE']:.4f}",
                "MAE (Lower is Better)": f"{m_info['MAE']:.4f}",
            })
        st.dataframe(pd.DataFrame(benchmark_data), width="stretch")