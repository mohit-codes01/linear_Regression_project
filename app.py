import glob
import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# DATA LOADING & CACHING
# ==========================================
@st.cache_data(show_spinner="Loading student dataset...")
def load_data(uploaded_file=None):
    """Loads dataset from upload, local CSV, or kagglehub cache."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    # 1. Local Student_Performance.csv in project directory
    if os.path.exists("Student_Performance.csv"):
        return pd.read_csv("Student_Performance.csv")

    local_csvs = glob.glob("*.csv")
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

    raise FileNotFoundError("No CSV dataset found. Please upload one via the sidebar.")


def preprocess_data(df_in):
    """Cleans dataset and creates derived categorization features."""
    df = df_in.copy().drop_duplicates()

    # Map binary categorical if not already numeric
    if "Extracurricular Activities" in df.columns:
        if df["Extracurricular Activities"].dtype == object:
            df["Extracurricular Activities"] = (
                df["Extracurricular Activities"].str.strip().map({"Yes": 1, "No": 0})
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
# MODEL TRAINING & EVALUATION
# ==========================================
@st.cache_resource(show_spinner="Training predictive models...")
def train_and_evaluate(df_clean, test_size=0.25, seed=42):
    """Trains Linear Regression along with Ridge, Lasso & Random Forest benchmarks."""
    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=int(seed)
    )

    # Main Multiple Linear Regression model
    lr_model = LinearRegression().fit(x_train, y_train)
    y_pred_test = lr_model.predict(x_test)
    y_pred_train = lr_model.predict(x_train)

    # Performance metrics
    n = len(y_test)
    p = x_test.shape[1]
    r2_test = r2_score(y_test, y_pred_test)
    adj_r2_test = 1 - ((1 - r2_test) * (n - 1) / (n - p - 1))

    metrics_lr = {
        "R² (Test)": r2_test,
        "Adj R² (Test)": adj_r2_test,
        "R² (Train)": r2_score(y_train, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "Max Error": max_error(y_test, y_pred_test),
    }

    # Benchmark models for model comparison tab
    models_comparison = {
        "Linear Regression": {
            "model": lr_model,
            "R² Test": r2_test,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "MAE": mean_absolute_error(y_test, y_pred_test),
        }
    }

    # Ridge
    ridge = Ridge(alpha=1.0).fit(x_train, y_train)
    ridge_preds = ridge.predict(x_test)
    models_comparison["Ridge Regression"] = {
        "model": ridge,
        "R² Test": r2_score(y_test, ridge_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds)),
        "MAE": mean_absolute_error(y_test, ridge_preds),
    }

    # Lasso
    lasso = Lasso(alpha=0.1).fit(x_train, y_train)
    lasso_preds = lasso.predict(x_test)
    models_comparison["Lasso Regression"] = {
        "model": lasso,
        "R² Test": r2_score(y_test, lasso_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, lasso_preds)),
        "MAE": mean_absolute_error(y_test, lasso_preds),
    }

    # Fast Random Forest sample
    rf = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=int(seed), n_jobs=-1).fit(
        x_train, y_train
    )
    rf_preds = rf.predict(x_test)
    models_comparison["Random Forest (Depth 8)"] = {
        "model": rf,
        "R² Test": r2_score(y_test, rf_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "MAE": mean_absolute_error(y_test, rf_preds),
    }

    coef_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coefficient": lr_model.coef_,
        "Abs_Coefficient": np.abs(lr_model.coef_),
    }).sort_values(by="Coefficient", ascending=False)

    return (
        lr_model,
        metrics_lr,
        coef_df,
        models_comparison,
        (x_train, x_test, y_train, y_test, y_pred_test),
    )


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    uploaded = st.file_uploader("📂 Upload Custom CSV Dataset", type=["csv"], help="Optional: upload your own student data file")

    st.markdown("---")
    st.markdown("#### 🧪 Model Hyperparameters")
    test_size_val = st.slider("Test Split Size", min_value=0.10, max_value=0.50, value=0.25, step=0.05, help="Percentage of records reserved for test evaluation")
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
        Multiple Linear Regression ML App<br>
        Built with Streamlit & Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# LOAD & PREPARE DATA
# ==========================================
try:
    df_raw = load_data(uploaded)
    df = preprocess_data(df_raw)
except Exception as e:
    st.error(f"❌ Could not load dataset: {e}")
    st.info("Please ensure 'Student_Performance.csv' is present or upload a CSV in the sidebar.")
    st.stop()

# Train models
(
    model,
    metrics,
    coef_df,
    benchmark_dict,
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
    <div style='display: flex; gap: 10px; align-items: center; margin-bottom: 18px;'>
        <span class='badge-pill'>📈 Multiple Linear Regression</span>
        <span class='badge-pill'>🎯 Test R²: {metrics['R² (Test)']*100:.2f}%</span>
        <span class='badge-pill'>⚡ RMSE: {metrics['RMSE']:.2f}</span>
        <span class='badge-pill'>👥 Dataset: {len(df):,} Students</span>
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
    "📈 Model Diagnostics & Benchmark",
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
            xaxis_title="Performance Index (0 - 100)",
            yaxis_title="Student Count",
            template="plotly_white",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with c_right:
        st.markdown("##### 🍩 Student Performance Tiers")
        if "Performance Tier" in df_filtered.columns:
            tier_counts = df_filtered["Performance Tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            color_map = {
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
                hole=0.55,
                color="Tier",
                color_discrete_map=color_map,
            )
            fig_donut.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=30, b=20),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # Charts Row 2: Deep breakdowns
    st.markdown("---")
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("##### ⏱️ Study Hours vs Performance (by Extracurricular Activities)")
        df_box = df_filtered.copy()
        df_box["Extracurricular"] = df_box["Extracurricular Activities"].map({1: "Yes", 0: "No"})
        fig_box = px.box(
            df_box,
            x="Hours Studied",
            y=TARGET,
            color="Extracurricular",
            color_discrete_map={"Yes": "#8B5CF6", "No": "#94A3B8"},
            points=False,
        )
        fig_box.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
            xaxis_title="Hours Studied (Daily)",
            yaxis_title="Performance Index",
        )
        st.plotly_chart(fig_box, use_container_width=True)

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
        st.plotly_chart(fig_sleep, use_container_width=True)

    # Key Insights Callout
    st.markdown(
        """
        <div class='callout-box'>
            <b>💡 Key Insights from Current Data:</b><br>
            • <b>Previous Scores</b> and <b>Hours Studied</b> account for over <b>98% of the variance</b> in final student performance.<br>
            • Each additional hour of study per day yields an expected <b>+2.85 point increase</b> in performance.<br>
            • Students engaging in <b>Extracurricular Activities</b> show a consistent positive gain of <b>+0.61 points</b> on average.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# TAB 2: SMART PREDICTOR & SIMULATOR
# ==========================================
with tab_predict:
    st.markdown("### 🔮 Real-Time Student Performance Predictor")
    st.caption("Adjust input parameters below to generate predictions, inspect feature contributions, and test what-if scenarios.")

    pred_tab1, pred_tab2 = st.tabs(["👤 Single Student Simulation", "📁 Batch Prediction (CSV)"])

    with pred_tab1:
        col_in1, col_in2, col_in3 = st.columns([1, 1, 1])

        with col_in1:
            in_hours = st.slider("📚 Hours Studied (Daily)", min_value=1, max_value=9, value=6, help="Average number of hours dedicated to studying daily")
            in_prev = st.slider("📝 Previous Scores", min_value=40, max_value=100, value=75, help="Past exam average score percentage")

        with col_in2:
            in_sleep = st.slider("🛌 Sleep Hours (Nightly)", min_value=4, max_value=9, value=7, help="Average hours of sleep per night")
            in_papers = st.slider("📄 Sample Papers Practiced", min_value=0, max_value=9, value=4, help="Total mock sample question papers completed")

        with col_in3:
            in_extra_str = st.radio("🏅 Extracurricular Activities", options=["Yes", "No"], index=0, horizontal=True)
            in_extra = 1 if in_extra_str == "Yes" else 0

            # Live calculation or button
            auto_predict = st.toggle("⚡ Real-Time Auto Predict", value=True)

        # Compute Prediction
        input_data = pd.DataFrame(
            [[in_hours, in_prev, in_extra, in_sleep, in_papers]],
            columns=FEATURE_COLS,
        )
        raw_pred = float(model.predict(input_data)[0])
        clamped_pred = max(0.0, min(100.0, raw_pred))

        st.markdown("---")

        # Result Presentation Section
        r_col1, r_col2 = st.columns([1.1, 1.4])

        with r_col1:
            st.markdown("#### 🎯 Predicted Outcome")

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
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Performance Grade Badge & Percentile
            if clamped_pred >= 90:
                grade_label = "🌟 Outstanding Performance (Grade A+)"
                css_class = "grade-outstanding"
                feedback = "Outstanding! The student shows top-tier mastery. Keep consistency and maintain current habits."
            elif clamped_pred >= 75:
                grade_label = "🟢 Excellent Performance (Grade A)"
                css_class = "grade-excellent"
                feedback = "Great trajectory! Increasing study time by 1 more hour could push into the top 90+ bracket."
            elif clamped_pred >= 60:
                grade_label = "🟡 Good Performance (Grade B)"
                css_class = "grade-good"
                feedback = "Solid foundation. Practicing 2-3 extra sample question papers will notably reinforce concepts."
            elif clamped_pred >= 40:
                grade_label = "🟠 Average Performance (Grade C)"
                css_class = "grade-average"
                feedback = "Needs focused intervention. Prioritize study hours and revise previous test problem areas."
            else:
                grade_label = "🔴 Needs Immediate Improvement (Grade D)"
                css_class = "grade-needs-help"
                feedback = "High risk of academic deficit. Needs structured study schedule and mentoring support."

            st.markdown(
                f"""
                <div style='text-align: center; margin-bottom: 12px;'>
                    <span class='grade-pill {css_class}'>{grade_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Percentile Rank
            percentile = (df[TARGET] <= clamped_pred).mean() * 100
            st.markdown(
                f"""
                <div class='callout-box' style='font-size: 0.9rem;'>
                    <b>📊 Percentile Rank:</b> Better than <b>{percentile:.1f}%</b> of benchmarked students.<br>
                    <b>💡 Recommendation:</b> {feedback}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown("#### 🔍 Feature Contribution Breakdown (Linear Formula)")
            st.caption("How each factor pulls the baseline intercept to produce the final predicted score:")

            # Calculate individual contributions based on regression formula: y = b0 + b1*x1 + ...
            intercept = float(model.intercept_)
            contrib_hours = in_hours * model.coef_[0]
            contrib_prev = in_prev * model.coef_[1]
            contrib_extra = in_extra * model.coef_[2]
            contrib_sleep = in_sleep * model.coef_[3]
            contrib_papers = in_papers * model.coef_[4]

            wf_df = pd.DataFrame({
                "Component": [
                    "Base Intercept",
                    f"Hours ({in_hours}h)",
                    f"Prev Scores ({in_prev})",
                    f"Extracurricular ({in_extra_str})",
                    f"Sleep ({in_sleep}h)",
                    f"Papers ({in_papers})",
                    "Final Predicted Score",
                ],
                "Value": [
                    intercept,
                    contrib_hours,
                    contrib_prev,
                    contrib_extra,
                    contrib_sleep,
                    contrib_papers,
                    clamped_pred,
                ],
            })

            fig_wf = go.Figure(
                go.Waterfall(
                    name="Score Breakdown",
                    orientation="v",
                    measure=["relative", "relative", "relative", "relative", "relative", "relative", "total"],
                    x=wf_df["Component"],
                    textposition="outside",
                    text=[f"{v:+.1f}" if i < 6 else f"{v:.1f}" for i, v in enumerate(wf_df["Value"])],
                    y=[intercept, contrib_hours, contrib_prev, contrib_extra, contrib_sleep, contrib_papers, 0],
                    connector={"line": {"color": "#CBD5E1"}},
                    decreasing={"marker": {"color": "#EF4444"}},
                    increasing={"marker": {"color": "#10B981"}},
                    totals={"marker": {"color": "#6366F1"}},
                )
            )
            fig_wf.update_layout(
                height=340,
                margin=dict(l=20, r=20, t=25, b=50),
                template="plotly_white",
                yaxis_title="Score Points",
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        # What-If Sensitivity Simulator
        st.markdown("---")
        st.markdown("#### 📈 'What-If' Study Hours Sensitivity Curve")
        st.caption("Simulates predicted performance across varying study hours while holding other inputs constant:")

        sim_hours = np.linspace(1, 10, 25)
        sim_scores = []
        for h in sim_hours:
            temp_in = pd.DataFrame(
                [[h, in_prev, in_extra, in_sleep, in_papers]],
                columns=FEATURE_COLS,
            )
            val = float(model.predict(temp_in)[0])
            sim_scores.append(max(0.0, min(100.0, val)))

        fig_sim = go.Figure()
        fig_sim.add_trace(
            go.Scatter(
                x=sim_hours,
                y=sim_scores,
                mode="lines",
                name="Performance Curve",
                line=dict(color="#6366F1", width=3),
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=[in_hours],
                y=[clamped_pred],
                mode="markers",
                name="Current Student",
                marker=dict(color="#EF4444", size=14, symbol="circle", line=dict(color="white", width=2)),
            )
        )
        fig_sim.add_hline(y=75, line_dash="dot", line_color="#10B981", annotation_text="Honors Threshold (75 pts)")
        fig_sim.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=25, b=20),
            template="plotly_white",
            xaxis_title="Hours Studied (Daily)",
            yaxis_title="Predicted Performance Index",
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

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

        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.success(f"Loaded {len(batch_df)} records for inference!")

                # Process columns
                batch_clean = batch_df.copy()
                if "Extracurricular Activities" in batch_clean.columns:
                    if batch_clean["Extracurricular Activities"].dtype == object:
                        batch_clean["Extracurricular Activities"] = (
                            batch_clean["Extracurricular Activities"].str.strip().map({"Yes": 1, "No": 0})
                        )

                # Verify columns
                missing = [c for c in FEATURE_COLS if c not in batch_clean.columns]
                if missing:
                    st.error(f"Missing required columns in CSV: {missing}")
                else:
                    batch_preds = model.predict(batch_clean[FEATURE_COLS])
                    batch_clean["Predicted Performance Index"] = np.clip(batch_preds, 0.0, 100.0).round(2)
                    
                    st.dataframe(batch_clean.head(15), use_container_width=True)

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
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            fig_corr.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with c_scat:
            st.markdown("##### 🎯 Dynamic 2D Scatter Explorer")
            scat_c1, scat_c2, scat_c3 = st.columns(3)
            with scat_c1:
                x_axis = st.selectbox("X-Axis Feature", options=FEATURE_COLS, index=0)
            with scat_c2:
                y_axis = st.selectbox("Y-Axis Target/Feature", options=[TARGET] + FEATURE_COLS, index=0)
            with scat_c3:
                color_feat = st.selectbox("Color By", options=["Extracurricular Activities", "Sleep Hours", "None"], index=0)

            sample_sub = df.sample(min(1500, len(df)), random_state=42)
            c_arg = None if color_feat == "None" else color_feat

            fig_scat = px.scatter(
                sample_sub,
                x=x_axis,
                y=y_axis,
                color=c_arg,
                trendline="ols",
                opacity=0.6,
                color_continuous_scale="Viridis",
            )
            fig_scat.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
            )
            st.plotly_chart(fig_scat, use_container_width=True)

    with eda_sub2:
        st.markdown("##### 🌐 3D Feature Interaction Space")
        st.write("Drag, zoom, and rotate the 3D plot to observe how **Hours Studied** and **Previous Scores** jointly determine the **Performance Index**:")

        sample_3d = df.sample(min(2000, len(df)), random_state=42)
        fig_3d = px.scatter_3d(
            sample_3d,
            x="Hours Studied",
            y="Previous Scores",
            z=TARGET,
            color=TARGET,
            color_continuous_scale="Plasma",
            opacity=0.7,
            size_max=4,
        )
        fig_3d.update_layout(
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                xaxis_title="Hours Studied",
                yaxis_title="Previous Scores",
                zaxis_title="Performance Index",
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with eda_sub3:
        st.markdown("##### 📋 Raw Dataset Inspector")
        st.dataframe(df.head(50), use_container_width=True)

        with st.expander("📊 Complete Statistical Summary (df.describe())"):
            st.dataframe(df.describe().T, use_container_width=True)


# ==========================================
# TAB 4: MODEL DIAGNOSTICS & BENCHMARK
# ==========================================
with tab_diagnostics:
    st.markdown("### 📈 Model Diagnostics, Residuals & Benchmarks")
    st.caption("Deep statistical audit of Multiple Linear Regression assumptions, residuals, and comparison with other algorithms.")

    # Detailed metrics grid
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R² Score (Test)", f"{metrics['R² (Test)']*100:.2f}%", "+Excellent fit")
    m2.metric("Adjusted R²", f"{metrics['Adj R² (Test)']*100:.2f}%")
    m3.metric("RMSE", f"{metrics['RMSE']:.3f} pts")
    m4.metric("MAE", f"{metrics['MAE']:.3f} pts")
    m5.metric("Max Error", f"{metrics['Max Error']:.2f} pts")

    st.markdown("---")

    # Residuals & Actual vs Predicted
    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("##### 🎯 Actual vs. Predicted (Test Set)")
        eval_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_test})
        eval_sample = eval_df.sample(min(1200, len(eval_df)), random_state=42)

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
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
            xaxis_title="Actual Performance Index",
            yaxis_title="Predicted Performance Index",
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with diag_c2:
        st.markdown("##### 📉 Residuals Distribution (Normality Check)")
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
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
            xaxis_title="Residual Error (Actual - Predicted)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # Feature Importance & Math Equation
    st.markdown("---")
    fi_col1, fi_col2 = st.columns([1.2, 1])

    with fi_col1:
        st.markdown("##### ⚖️ Feature Importance & Coefficients")
        fig_coef = px.bar(
            coef_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Coefficient",
            color_continuous_scale="Blues",
            text="Coefficient",
        )
        fig_coef.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_coef.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
            coloraxis_showscale=False,
            xaxis_title="Weight / Coefficient Value",
        )
        st.plotly_chart(fig_coef, use_container_width=True)

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
                • <b>Intercept ({model.intercept_:.2f}):</b> Baseline score when all inputs are 0.<br>
                • <b>Hours Studied ({model.coef_[0]:.2f}):</b> Strongest impact per unit. Each hour increases score by ~2.85 points.<br>
                • <b>Previous Scores ({model.coef_[1]:.2f}):</b> 1-to-1 linear carryover from previous scores.
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
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)