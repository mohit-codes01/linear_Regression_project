# 🎓 EduPredict — Student Performance Analytics & Prediction Dashboard

An advanced, interactive machine learning dashboard powered by **Multiple Linear Regression**, **Streamlit**, and **Plotly** to predict and analyze student academic performance.

---

## 🌟 Features Overview

### 1. 📊 Executive Overview & KPI Dashboard
- **High-Level KPI Cards**: Active student count, average performance index, percentage of high achievers ($\ge 75$), and model accuracy ($R^2 \approx 98.9\%$).
- **Performance Distribution**: Interactive histogram with KDE, mean reference lines, and box marginals.
- **Performance Tiers**: Donut chart segmenting students into *Outstanding (90-100)*, *Excellent (75-89)*, *Good (60-74)*, *Average (40-59)*, and *Needs Help (<40)*.
- **Bivariate Insights**: Study hours vs score broken down by extracurricular activities, and sleep hours vs average performance.

### 2. 🔮 Smart Predictor & Simulator
- **Single Student Prediction**:
  - Interactive sliders for *Hours Studied*, *Previous Scores*, *Sleep Hours*, *Sample Papers Practiced*, and *Extracurricular Activities*.
  - **Plotly Speedometer Gauge**: Color-zoned radial meter ($0-100$) reflecting real-time scores.
  - **Performance Grade & Percentile Rank**: Contextual grading badge and percentile ranking relative to the benchmark dataset.
  - **Feature Contribution Waterfall Chart**: Visual breakdown showing how the model's intercept baseline combines with each input parameter to arrive at the final score ($Y = \beta_0 + \sum \beta_i X_i$).
  - **'What-If' Study Hours Sensitivity Curve**: Dynamic simulation curve illustrating score trajectory if study hours vary from 1 to 10 hours daily.
- **Batch CSV Inference Engine**:
  - Upload a CSV containing multiple student records for bulk predictions.
  - Downloadable prediction output with single-click export.
  - Downloadable sample template CSV for reference.

### 3. 🔍 Interactive Exploratory Data Analysis (EDA)
- **Correlation Heatmap**: Annotated Plotly heatmap showing pairwise correlation coefficients.
- **Dynamic 2D Scatter Explorer**: Customizable X/Y axes with OLS regression trendline and color grouping.
- **3D Interactive Feature Space**: Rotate, zoom, and pan across a 3D scatter plot of *(Hours Studied $\times$ Previous Scores $\times$ Performance Index)*.
- **Raw Data Inspector**: Searchable data table with statistical summary (`df.describe()`).

### 4. 📈 Model Diagnostics & Benchmarks
- **Model Quality Metrics**: $R^2$ Score, Adjusted $R^2$, RMSE, MAE, and Maximum Error.
- **Actual vs. Predicted Evaluation**: Scatter plot with $y = x$ reference diagonal line.
- **Residuals Normality Audit**: Error distribution histogram to inspect linear regression assumptions.
- **Feature Importance & Mathematical Formula**:
  - Formatted LaTeX linear equation.
  - Horizontal bar chart of regression weights.
  - Clear plain-English coefficient interpretations.
- **Multi-Model Benchmark Comparison**: Side-by-side performance comparison against *Ridge Regression*, *Lasso Regression*, and *Random Forest*.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure Python 3.9+ is installed. Install all required packages:

```bash
pip install -r requirement.txt
```

### 2. Launch the Dashboard
Run the Streamlit application:

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
linear regression project/
├── app.py                   # Main Streamlit & Plotly dashboard application
├── Student_Performance.csv  # Local student performance dataset (10,000 records)
├── requirement.txt          # Python dependencies
└── README.md                # Project documentation
```

---

## 📐 Mathematical Model

The multiple linear regression equation fitted on this dataset:

$$\text{Performance Index} \approx -34.08 + 2.85(\text{Hours}) + 1.02(\text{Prev}) + 0.61(\text{Extra}) + 0.48(\text{Sleep}) + 0.19(\text{Papers})$$
