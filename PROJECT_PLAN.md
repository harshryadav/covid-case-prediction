# COVID-19 Case Prediction Project - Comprehensive Plan
## Using GluonTS for Probabilistic Time Series Forecasting

---

## 📊 Project Overview

**Objective:** Build a probabilistic time series forecasting system to predict COVID-19 cases in the United States using GluonTS, with uncertainty quantification and scenario analysis capabilities.

**Difficulty Level:** 3 (Hard)

**Focus Region:** United States (with potential state-level analysis)

---

## 🎯 Project Goals

1. **Forecast COVID-19 cases** using historical data with multiple advanced models
2. **Quantify uncertainty** in predictions with confidence intervals
3. **Evaluate model performance** using CRPS and other metrics
4. **Conduct scenario analysis** for different intervention strategies
5. **Visualize predictions** with uncertainty bands
6. **BONUS:** Incorporate mobility data to enhance predictions

---

## 📁 Required Datasets

### Core Datasets (Already Available ✅)

From the [JHU CSSE COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19):

1. **Time Series Confirmed Cases - US**
   - **File:** `time_series_covid19_confirmed_US.csv`
   - **Location:** `/data/time_series_covid19_confirmed_US.csv`
   - **Description:** Daily confirmed COVID-19 cases by county/state
   - **Columns:** Geographic metadata + daily case counts (columns from 2020-01-22 onwards)
   - **Status:** ✅ Available

2. **Time Series Deaths - US**
   - **File:** `time_series_covid19_deaths_US.csv`
   - **Location:** `/data/time_series_covid19_deaths_US.csv`
   - **Description:** Daily COVID-19 deaths by county/state
   - **Columns:** Geographic metadata + population + daily death counts
   - **Status:** ✅ Available

3. **Time Series Vaccines - US**
   - **File:** `time_series_covid19_vaccine_us.csv`
   - **Location:** `/data/time_series_covid19_vaccine_us.csv`
   - **Description:** Vaccination data by state
   - **Status:** ✅ Available

### Additional Datasets to Download

#### 4. **Google COVID-19 Community Mobility Reports** (BONUS - Recommended)
   - **Source:** [ActiveConclusion COVID-19 Mobility Archive](https://github.com/ActiveConclusion/COVID19_mobility)
   - **Direct File:** [mobility_report_US.csv](https://github.com/ActiveConclusion/COVID19_mobility/blob/master/google_reports/mobility_report_US.csv)
   - **File Format:** CSV
   - **Filename:** `mobility_report_US.csv`
   - **Date Coverage:** February 15, 2020 - February 1, 2022 (24 months)
   - **Columns:**
     - `state` (state name or "Total" for national)
     - `county` (county name or "Total" for state/national)
     - `date` (YYYY-MM-DD)
     - `retail and recreation` (% change from baseline)
     - `grocery and pharmacy` (% change from baseline)
     - `parks` (% change from baseline)
     - `transit stations` (% change from baseline)
     - `workplaces` (% change from baseline)
     - `residential` (% change from baseline)
   - **Usage:** Covariates for enhanced prediction models
   - **Status:** ✅ Already available in your data folder!

#### 5. **Policy/Intervention Data** (Optional Enhancement)
   - **Source:** [Oxford COVID-19 Government Response Tracker (OxCGRT)](https://github.com/OxCGRT/covid-policy-tracker)
   - **File Format:** CSV
   - **Suggested Filename:** `oxford_government_response_us.csv`
   - **Columns:** Stringency Index, containment measures, economic support
   - **Usage:** Scenario analysis for policy interventions
   - **Status:** 🔄 Optional

#### 6. **US Population Data** (Reference - Already in deaths dataset)
   - **Source:** Already included in `time_series_covid19_deaths_US.csv`
   - **Usage:** Calculate per capita rates, normalize predictions
   - **Status:** ✅ Available

---

## 🗂️ Data Acquisition Guide

### Step 1: Core JHU CSSE Data (Already Complete ✅)
Your current data files are from the JHU CSSE repository's time series data located at:
```
https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
```

**Files you have:**
- `time_series_covid19_confirmed_US.csv`
- `time_series_covid19_deaths_US.csv`
- `time_series_covid19_vaccine_us.csv`

### Step 2: Google Mobility Data (Bonus Feature)

**✅ GOOD NEWS: You already have this file!**

The file `mobility_report_US.csv` is already in your `data/` folder from the ActiveConclusion archive.

**If you need to re-download:**
```bash
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction/data

# Direct download from ActiveConclusion GitHub
curl -O https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv

# Or using wget
wget https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv
```

**Using Python:**
```python
import pandas as pd

# Download from ActiveConclusion
url = "https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv"
mobility_df = pd.read_csv(url)

# Save to your data directory
mobility_df.to_csv('data/mobility_report_US.csv', index=False)

# Verify
print(f"Downloaded {len(mobility_df):,} rows")
national = mobility_df[mobility_df['state'] == 'Total']
print(f"Date range: {national['date'].min()} to {national['date'].max()}")
```

**Why ActiveConclusion is better:**
- ✅ Covers Feb 2020 - Feb 2022 (24 months)
- ✅ Pre-filtered for US only
- ✅ Clean format, ready to use
- ✅ Includes all major pandemic waves

### Step 3: Oxford Government Response Data (Optional)
```bash
# Clone the OxCGRT repository or download specific file
wget https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv

# Filter for United States
# Save to data/oxford_government_response_us.csv
```

---

## 🏗️ Project Structure (Proposed)

```
covid-case-prediction/
│
├── data/
│   ├── raw/                                    # Raw data files
│   │   ├── time_series_covid19_confirmed_US.csv
│   │   ├── time_series_covid19_deaths_US.csv
│   │   ├── time_series_covid19_vaccine_us.csv
│   │   ├── mobility_report_US.csv             # ✅ Already available
│   │   └── oxford_government_response_us.csv  # Optional
│   │
│   └── processed/                              # Processed data for GluonTS
│       ├── train_data.json                     # Training dataset in GluonTS format
│       ├── test_data.json                      # Test dataset in GluonTS format
│       ├── us_daily_cases.csv                  # Aggregated daily cases
│       ├── us_daily_cases_with_covariates.csv  # With mobility features
│       └── state_level_data/                   # State-specific datasets
│
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── load_jhu_data.py                   # Load JHU CSSE data
│   │   ├── load_mobility_data.py              # Load Google mobility data
│   │   ├── prepare_gluonts_format.py          # Convert to GluonTS format
│   │   └── feature_engineering.py             # Create covariates & features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py                 # Simple baselines (Naive, ARIMA)
│   │   ├── deepar_model.py                    # DeepAR implementation
│   │   ├── transformer_model.py               # Transformer implementation
│   │   ├── gp_model.py                        # Gaussian Process implementation
│   │   └── model_factory.py                   # Factory for model selection
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                           # Training pipeline
│   │   ├── hyperparameter_tuning.py           # HPO with optuna/ray
│   │   └── config.py                          # Training configurations
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                         # CRPS, MAE, RMSE, etc.
│   │   ├── evaluate_models.py                 # Model evaluation pipeline
│   │   └── uncertainty_analysis.py            # Uncertainty quantification
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plot_predictions.py                # Prediction visualizations
│   │   ├── plot_uncertainty.py                # Uncertainty visualization
│   │   └── plot_scenario_analysis.py          # Scenario comparison
│   │
│   └── scenarios/
│       ├── __init__.py
│       └── scenario_generator.py              # Scenario analysis for interventions
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb     # EDA
│   ├── 02_data_preparation.ipynb              # Data prep for GluonTS
│   ├── 03_baseline_models.ipynb               # Simple baselines
│   ├── 04_deepar_experiments.ipynb            # DeepAR experiments
│   ├── 05_transformer_experiments.ipynb       # Transformer experiments
│   ├── 06_model_comparison.ipynb              # Compare all models
│   ├── 07_uncertainty_analysis.ipynb          # Uncertainty quantification
│   └── 08_scenario_analysis.ipynb             # Scenario analysis
│
├── results/
│   ├── models/                                 # Saved model checkpoints
│   ├── predictions/                            # Prediction outputs
│   ├── figures/                                # Generated plots
│   └── metrics/                                # Evaluation metrics
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── PROJECT_PLAN.md                            # This file
└── config.yaml                                 # Configuration file
```

---

## 🔧 Technology Stack

### Core Libraries
```
gluonts >= 0.14.0              # Time series forecasting framework
mxnet >= 1.9.1                 # Deep learning backend for GluonTS
pandas >= 1.5.0                # Data manipulation
numpy >= 1.23.0                # Numerical computing
matplotlib >= 3.6.0            # Visualization
seaborn >= 0.12.0              # Statistical visualization
```

### Additional Libraries
```
scikit-learn >= 1.2.0          # Preprocessing & metrics
statsmodels >= 0.14.0          # Statistical models (ARIMA)
pytorch >= 2.0.0               # Alternative backend (optional)
optuna >= 3.0.0                # Hyperparameter optimization
plotly >= 5.11.0               # Interactive visualizations
jupyter >= 1.0.0               # Notebook environment
tqdm >= 4.64.0                 # Progress bars
pyyaml >= 6.0                  # Configuration management
```

---

## 📋 Implementation Roadmap

### Phase 1: Data Preparation & EDA (Week 1)

#### Task 1.1: Download Additional Datasets
- [ ] Download Google Mobility data
- [ ] Filter for United States
- [ ] Store in `data/raw/` directory
- [ ] Verify data integrity

#### Task 1.2: Exploratory Data Analysis
- [ ] Analyze temporal patterns in cases/deaths
- [ ] Identify waves/peaks in the pandemic
- [ ] Check for missing values and outliers
- [ ] Visualize correlations between cases, deaths, vaccinations
- [ ] Analyze mobility patterns over time
- [ ] Identify seasonal patterns

**Deliverable:** Comprehensive EDA notebook with visualizations

#### Task 1.3: Data Preprocessing
- [ ] Aggregate county-level data to state/national level
- [ ] Handle missing values
- [ ] Create daily new cases from cumulative
- [ ] Smooth data using rolling averages
- [ ] Split data into train/validation/test sets
  - Training: 2020-01-22 to 2021-12-31
  - Validation: 2022-01-01 to 2022-06-30
  - Test: 2022-07-01 to 2023-03-10

**Deliverable:** Clean, preprocessed datasets

### Phase 2: GluonTS Data Transformation (Week 1-2)

#### Task 2.1: Understand GluonTS Data Format
GluonTS expects data in a specific format:
```python
{
    "start": "2020-01-22",                    # Start date
    "target": [1, 2, 5, 8, ...],             # Time series values
    "feat_static_cat": [0],                   # Static categorical features
    "feat_static_real": [331000000],          # Static real features (e.g., population)
    "feat_dynamic_real": [[...], [...]]       # Dynamic covariates (e.g., mobility)
}
```

#### Task 2.2: Create Training/Test Datasets
- [ ] Convert pandas DataFrames to GluonTS ListDataset format
- [ ] Create univariate datasets (cases only)
- [ ] Create multivariate datasets (with covariates)
- [ ] Define prediction horizon (e.g., 7, 14, 30 days)
- [ ] Save datasets as JSON/pickle files

**Key Script:** `prepare_gluonts_format.py`

```python
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas

# Example structure
def create_gluonts_dataset(df, start_date, prediction_length):
    """
    Convert pandas DataFrame to GluonTS format
    
    Parameters:
    - df: DataFrame with columns [Date, Daily_Cases, Mobility_Features...]
    - start_date: Start date of the time series
    - prediction_length: Forecast horizon
    """
    train_data = ListDataset(
        [
            {
                "start": start_date,
                "target": df['Daily_Cases'].values,
                "feat_dynamic_real": df[mobility_features].values.T
            }
        ],
        freq="D"  # Daily frequency
    )
    return train_data
```

#### Task 2.3: Feature Engineering
Create dynamic covariates:
- [ ] Lagged features (cases from 7, 14, 30 days ago)
- [ ] Rolling statistics (7-day, 14-day averages)
- [ ] Day of week indicators
- [ ] Vaccination rates (if incorporating vaccines)
- [ ] Mobility indices (if using mobility data)
- [ ] Policy stringency indices (if using Oxford data)

**Deliverable:** GluonTS-compatible datasets with features

### Phase 3: Baseline Models (Week 2)

#### Task 3.1: Implement Simple Baselines
- [ ] Naive Forecast (repeat last value)
- [ ] Seasonal Naive (repeat same day last week)
- [ ] Moving Average
- [ ] ARIMA/SARIMA

**Purpose:** Establish performance benchmarks

#### Task 3.2: Evaluate Baselines
- [ ] Calculate CRPS, MAE, RMSE, MAPE
- [ ] Visualize predictions
- [ ] Document baseline performance

**Deliverable:** Baseline model results

### Phase 4: Advanced GluonTS Models (Week 3-4)

#### Task 4.1: DeepAR Model
**Description:** Autoregressive RNN-based model with probabilistic outputs

```python
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

estimator = DeepAREstimator(
    freq="D",
    prediction_length=14,
    context_length=56,  # Use 56 days of history
    num_layers=3,
    num_cells=40,
    dropout_rate=0.1,
    trainer=Trainer(
        epochs=100,
        learning_rate=1e-3,
        batch_size=32
    )
)

predictor = estimator.train(training_data=train_ds)
```

**Experiments:**
- [ ] Tune hyperparameters (layers, cells, dropout)
- [ ] Vary context length (30, 60, 90 days)
- [ ] Test with/without covariates
- [ ] Evaluate on validation set

#### Task 4.2: Transformer Model
**Description:** Attention-based model for long-range dependencies

```python
from gluonts.model.transformer import TransformerEstimator

estimator = TransformerEstimator(
    freq="D",
    prediction_length=14,
    context_length=56,
    model_dim=32,
    num_heads=4,
    num_layers=3,
    dropout_rate=0.1,
    trainer=Trainer(epochs=100, learning_rate=1e-3)
)

predictor = estimator.train(training_data=train_ds)
```

**Experiments:**
- [ ] Tune attention heads and layers
- [ ] Test different context windows
- [ ] Compare with DeepAR

#### Task 4.3: Gaussian Process Model (Optional)
**Description:** Non-parametric model with uncertainty quantification

```python
from gluonts.model.gp_forecaster import GaussianProcessEstimator

estimator = GaussianProcessEstimator(
    freq="D",
    prediction_length=14,
    context_length=56,
    trainer=Trainer(epochs=50)
)

predictor = estimator.train(training_data=train_ds)
```

**Experiments:**
- [ ] Test different kernel functions
- [ ] Compare computational efficiency
- [ ] Analyze uncertainty estimates

#### Task 4.4: Other Models to Consider
- [ ] **SimpleFeedForward:** Simple MLP baseline
- [ ] **Temporal Fusion Transformer:** State-of-the-art for interpretability
- [ ] **N-BEATS:** Neural basis expansion for interpretable forecasting
- [ ] **WaveNet:** CNN-based probabilistic model

**Deliverable:** Trained models with evaluation metrics

### Phase 5: Model Evaluation (Week 4)

#### Task 5.1: Comprehensive Evaluation
Metrics to calculate:
- **CRPS (Continuous Ranked Probability Score):** Primary metric for probabilistic forecasts
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **QuantileLoss:** Evaluate specific quantiles (10%, 50%, 90%)
- **Winkler Score:** Interval-based metric

```python
from gluonts.evaluation import make_evaluation_predictions, Evaluator

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)
tss = list(ts_it)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(tss, forecasts)

print(agg_metrics)
```

#### Task 5.2: Model Comparison
- [ ] Create comparison table of all models
- [ ] Rank by CRPS
- [ ] Analyze trade-offs (accuracy vs. uncertainty)
- [ ] Statistical significance tests

**Deliverable:** Comprehensive evaluation report

### Phase 6: Uncertainty Quantification (Week 5)

#### Task 6.1: Analyze Prediction Intervals
- [ ] Plot 50%, 80%, 95% confidence intervals
- [ ] Check calibration (do 95% intervals contain 95% of actuals?)
- [ ] Visualize prediction fans
- [ ] Analyze where uncertainty is highest/lowest

#### Task 6.2: Probabilistic Forecast Visualization
```python
import matplotlib.pyplot as plt

def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50, 90)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Plot actual data
    ts_entry[-plot_length:].plot(ax=ax)
    
    # Plot forecast
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    
    plt.grid(which="both")
    plt.legend(["observations", "median prediction", "50% confidence interval", "90% confidence interval"])
    plt.show()
```

#### Task 6.3: Uncertainty Analysis
- [ ] Identify periods of high uncertainty
- [ ] Correlate uncertainty with external events (new variants, policy changes)
- [ ] Compare uncertainty across models

**Deliverable:** Uncertainty analysis report with visualizations

### Phase 7: Scenario Analysis (Week 5-6)

#### Task 7.1: Define Scenarios
Create what-if scenarios:
1. **Baseline:** Current trends continue
2. **Optimistic:** Increased mobility restrictions (simulate lockdown)
3. **Pessimistic:** Relaxed restrictions, new variant emergence
4. **Vaccination Impact:** Accelerated vaccination rates

#### Task 7.2: Implement Scenario Generator
```python
class ScenarioGenerator:
    def __init__(self, base_mobility_data):
        self.base_data = base_mobility_data
    
    def generate_lockdown_scenario(self, reduction_percent=30):
        """Simulate mobility reduction"""
        scenario_data = self.base_data.copy()
        scenario_data['mobility'] *= (1 - reduction_percent/100)
        return scenario_data
    
    def generate_variant_scenario(self, transmission_increase=20):
        """Simulate more transmissible variant"""
        # Adjust R0 or case multiplier
        pass
```

#### Task 7.3: Run Scenarios
- [ ] Generate forecasts for each scenario
- [ ] Compare outcomes
- [ ] Visualize scenario fan charts
- [ ] Quantify impact of interventions

#### Task 7.4: Policy Recommendations
- [ ] Analyze which interventions are most effective
- [ ] Quantify uncertainty in scenario outcomes
- [ ] Create actionable insights

**Deliverable:** Scenario analysis report

### Phase 8: Bonus - Mobility Integration (Week 6)

#### Task 8.1: Prepare Mobility Features
- [ ] Merge Google Mobility data with case data
- [ ] Handle missing values
- [ ] Normalize mobility indices
- [ ] Create lagged mobility features

#### Task 8.2: Train Models with Mobility Covariates
- [ ] Add mobility as `feat_dynamic_real`
- [ ] Retrain all models
- [ ] Compare performance with baseline (no mobility)

#### Task 8.3: Analyze Mobility Impact
- [ ] Feature importance analysis
- [ ] Correlation between mobility and case trends
- [ ] Quantify prediction improvement

**Deliverable:** Enhanced models with mobility features

### Phase 9: Documentation & Presentation (Week 7)

#### Task 9.1: Code Documentation
- [ ] Add docstrings to all functions
- [ ] Create API documentation
- [ ] Write usage examples

#### Task 9.2: Results Documentation
- [ ] Write final report
- [ ] Create presentation slides
- [ ] Prepare visualizations

#### Task 9.3: Repository Cleanup
- [ ] Organize code structure
- [ ] Update README
- [ ] Add installation instructions
- [ ] Create reproducibility guide

**Deliverable:** Complete, documented project

---

## 📊 Key Metrics & Evaluation

### Primary Metric: CRPS
**Continuous Ranked Probability Score** - Measures accuracy of probabilistic forecasts
- Lower is better
- Accounts for both accuracy and calibration
- Standard metric for probabilistic time series

### Secondary Metrics
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **Quantile Loss:** At 10%, 50%, 90%
- **Coverage:** Percentage of actuals within prediction intervals

---

## 🎨 Visualization Requirements

1. **Time Series Plots**
   - Historical cases with train/validation/test splits
   - Predictions overlaid on actuals
   - Multi-step ahead forecasts

2. **Uncertainty Visualization**
   - Prediction intervals (fan charts)
   - Quantile forecasts
   - Spaghetti plots (sample trajectories)

3. **Model Comparison**
   - Metrics comparison bar charts
   - Error distribution plots
   - Calibration plots

4. **Scenario Analysis**
   - Scenario comparison fan charts
   - Impact quantification plots
   - Policy intervention analysis

5. **Interactive Dashboards** (Optional)
   - Plotly/Dash dashboard for exploration
   - Streamlit app for predictions

---

## ⚠️ Challenges & Considerations

### Data Challenges
1. **Non-stationarity:** COVID trends changed dramatically over time
2. **Structural breaks:** Vaccines, variants, policy changes
3. **Data quality:** Reporting delays, weekend effects, revisions
4. **Multiple waves:** Different dynamics across pandemic phases

### Modeling Challenges
1. **Cold start:** Limited historical data at pandemic start
2. **Distribution shift:** Training data may not reflect current dynamics
3. **External factors:** Policy, behavior changes not captured in data
4. **Computational resources:** Deep learning models require GPU

### Solutions
- Use appropriate train/test splits that reflect realistic forecasting scenarios
- Implement robust preprocessing (smoothing, outlier handling)
- Ensemble multiple models for robustness
- Regular model retraining as new data arrives

---

## 🎯 Success Criteria

### Minimum Requirements
- ✅ Implement at least 2 GluonTS models (e.g., DeepAR + Transformer)
- ✅ Evaluate using CRPS metric
- ✅ Generate probabilistic forecasts with uncertainty intervals
- ✅ Create visualizations of predictions and uncertainty
- ✅ Conduct basic scenario analysis

### Stretch Goals
- 🎖️ Incorporate mobility data as covariates
- 🎖️ Implement state-level forecasting (50 state models)
- 🎖️ Create ensemble model combining multiple approaches
- 🎖️ Build interactive dashboard for exploration
- 🎖️ Publish analysis as blog post or paper

---

## 📚 Resources & References

### GluonTS Documentation
- [GluonTS Official Docs](https://ts.gluon.ai/)
- [GluonTS Tutorials](https://ts.gluon.ai/stable/tutorials/index.html)
- [GluonTS GitHub](https://github.com/awslabs/gluonts)

### COVID-19 Data
- [JHU CSSE COVID-19 Dashboard](https://coronavirus.jhu.edu/map.html)
- [JHU CSSE GitHub Repository](https://github.com/CSSEGISandData/COVID-19)
- [Google Mobility Reports](https://www.google.com/covid19/mobility/)

### Research Papers
1. Dong, E., Du, H., & Gardner, L. (2020). "An interactive web-based dashboard to track COVID-19 in real time." *The Lancet Infectious Diseases*, 20(5), 533-534.
2. Salinas, D., et al. (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." *International Journal of Forecasting*, 36(3), 1181-1191.
3. Lim, B., et al. (2021). "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting*, 37(4), 1748-1764.

### Tutorials
- [Time Series Forecasting with GluonTS](https://github.com/awslabs/gluon-ts/tree/master/examples)
- [Probabilistic Forecasting Guide](https://otexts.com/fpp3/)

---

## 📅 Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Data Prep & EDA | Clean datasets, EDA notebook |
| 2 | GluonTS Transformation & Baselines | GluonTS format data, baseline results |
| 3 | Advanced Models (DeepAR) | Trained DeepAR model |
| 4 | Advanced Models (Transformer) & Evaluation | All models trained, evaluation complete |
| 5 | Uncertainty & Scenarios | Uncertainty analysis, scenario results |
| 6 | Bonus (Mobility) | Enhanced models with mobility |
| 7 | Documentation | Final report, presentation |

---

## 🚀 Getting Started

### Installation Steps

1. **Clone the repository (if not already):**
```bash
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download additional data:**
```bash
# Run data download script (to be created)
python src/data_processing/download_mobility_data.py
```

5. **Run EDA notebook:**
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 💡 Next Immediate Steps

1. **Create `requirements.txt`** with all dependencies
2. **Download Google Mobility data**
3. **Create project structure** (directories and empty files)
4. **Start with EDA notebook** to understand data patterns
5. **Implement data preprocessing pipeline** for GluonTS

---

## 📞 Support & Questions

- **GluonTS GitHub Issues:** https://github.com/awslabs/gluonts/issues
- **Stack Overflow:** Tag questions with `gluonts` and `time-series`
- **Project Documentation:** See README.md

---

## 📝 License & Attribution

**Data Attribution:**
- COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University
- Citation: Dong E, Du H, Gardner L. An interactive web-based dashboard to track COVID-19 in real time. Lancet Inf Dis. 20(5):533-534. doi: 10.1016/S1473-3099(20)30120-1

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

**End of Project Plan**

Good luck with your COVID-19 case prediction project! 🎉

