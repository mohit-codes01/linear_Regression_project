# 🎓 EduPredict — Student Performance Analytics & Prediction Dashboard

An advanced, production-grade interactive machine learning & econometric dashboard powered by **Multiple Linear Regression**, **Streamlit**, **Plotly**, and **SciPy** to predict, analyze, and optimize student academic performance.

---

## 🌟 Modern Architecture & Features

### 1. 📊 Executive Overview & KPI Dashboard
- **High-Level KPI Cards**: Active student count, average performance index, high achievers percentage ($\ge 75$), and model accuracy ($R^2 \approx 98.9\%$).
- **Performance Distribution**: Interactive histogram with KDE, mean reference lines, and box marginals.
- **Academic Performance Tiers**: Donut chart segmenting students into *Outstanding (90-100)*, *Excellent (75-89)*, *Good (60-74)*, *Average (40-59)*, and *Needs Help (<40)*.
- **Bivariate Insights**: Study hours vs score broken down by extracurricular activities, and sleep duration vs performance index.
- **🏫 Cohort Policy Intervention Simulator**: Simulate school-wide interventions (e.g. adding mandatory study hours or weekly mock papers) to preview projected shifts in grade distribution and high-achiever rates.

### 2. 🔮 Smart Predictor & Goal-Seeker Optimizer
- **Single Student Simulation**:
  - Interactive sliders for *Hours Studied*, *Previous Scores*, *Sleep Hours*, *Sample Papers Practiced*, and *Extracurricular Activities*.
  - **Speedometer Gauge**: Color-zoned radial meter reflecting real-time scores.
  - **🎯 95% Analytical Prediction Intervals**: Calculates mathematical uncertainty bounds ($\hat{y}_0 \pm 1.96 \cdot SE_{\text{pred}}$) for true statistical confidence.
  - **Academic Grade & Percentile Rank**: Contextual grading badge and percentile ranking relative to the benchmark dataset.
  - **Feature Contribution Waterfall Chart**: Visual breakdown showing how the model's intercept baseline combines with each input parameter ($Y = \beta_0 + \sum \beta_i X_i$).
  - **🎯 Smart Goal-Seeker**: Enter a target score (e.g. 85 pts) to compute 3 actionable improvement pathways (study hours focus, mock tests focus, or balanced lifestyle).
  - **'What-If' Sensitivity Curve**: Dynamic simulation curve illustrating score trajectory if daily study hours vary from 1 to 9 hours.
  - **📄 Downloadable Student Report Card**: Single-click export of an evaluation report card (.txt) with predictions and personalized advice.
- **📁 Batch CSV Inference Engine**:
  - Upload custom CSVs for bulk predictions on hundreds or thousands of students simultaneously.
  - Single-click export of predictions CSV.
  - Downloadable sample template CSV for reference.

### 3. 🔍 Interactive Exploratory Data Analysis (EDA)
- **Correlation Heatmap**: Annotated Plotly heatmap showing pairwise Pearson correlation coefficients.
- **Dynamic 2D Scatter Explorer**: Customizable X/Y axes with OLS regression trendline and color grouping.
- **3D Interactive Feature Space**: Rotate, zoom, and pan across a 3D scatter plot of *(Hours Studied $\times$ Previous Scores $\times$ Performance Index)*.
- **Raw Data Inspector**: Searchable data table with statistical summary (`df.describe()`).

### 4. 📈 Model Diagnostics & Gauss-Markov Econometric Audit
- **Model Quality Metrics**: $R^2$ Score, Adjusted $R^2$, RMSE, MAE, and Maximum Error.
- **Actual vs. Predicted**: Scatter plot with $y = x$ reference diagonal line.
- **Residuals Normality Audit**: Error distribution histogram to inspect linear regression assumptions.
- **📊 Normal Q-Q Plot (Quantile-Quantile)**: Evaluates residual normality against theoretical standard normal quantiles.
- **📐 Residuals vs. Fitted (Homoscedasticity Test)**: Validates constant error variance across predicted score levels.
- **🛡️ Multicollinearity Audit (VIF)**: Computes Variance Inflation Factors for all features ($VIF \approx 1.0$, proving predictor independence).
- **⚖️ Standardized Beta Coefficients**: Measures relative effect size in standard deviation units ($\beta^*_j$).
- **Learned Formula & Mathematical Equation**: Formatted LaTeX linear equation.
- **Multi-Model Benchmark Comparison**: Side-by-side performance comparison against *Ridge Regression (L2)*, *Lasso Regression (L1)*, and *Random Forest*.

### 5. 🛡️ Intelligent Schema Normalizer & Safe Uploads
- **Auto-Detects Column Variations**: Automatically matches snake_case, spaces, lowercase, and educational synonyms (`study_hours`, `prev_score`, `final_score`, etc.).
- **Automatic Inference Detection**: If a user uploads an input template without a target column, the app does not crash — it trains on benchmark data and routes the file to the batch predictor.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure Python 3.9+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard

**Option A (Windows 1-Click):**
Double-click `run_app.bat` in the project folder.

**Option B (Terminal):**
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
linear regression project/
├── app.py                   # Streamlit & Plotly analytics dashboard
├── Student_Performance.csv  # Benchmark student performance dataset (10,000 records)
├── requirements.txt         # Production dependencies
├── requirement.txt          # Secondary requirements reference
├── run_app.bat              # 1-Click launcher for Windows
├── Procfile                 # Cloud deployment procfile
├── .streamlit/
│   └── config.toml          # Production server configuration
├── .gitignore               # Ignored cache & environment files
└── README.md                # Comprehensive documentation
```

---

## 📐 Mathematical Model

The multiple linear regression equation fitted on this dataset:

$$\text{Performance Index} \approx -34.08 + 2.85(\text{Hours}) + 1.02(\text{Prev}) + 0.61(\text{Extra}) + 0.48(\text{Sleep}) + 0.19(\text{Papers})$$
