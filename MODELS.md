# COVID-19 Forecasting Models Guide

This document provides detailed information about all forecasting models implemented in this project.

## Model Overview

| Model | Type | Multivariate | Strengths | Use Case |
|-------|------|--------------|-----------|----------|
| **Naive** | Baseline | No | Simple, fast | Baseline comparison |
| **Seasonal Naive** | Baseline | No | Captures weekly patterns | Baseline with seasonality |
| **DeepAR** | Deep Learning (RNN) | No | Probabilistic, handles uncertainty | General-purpose forecasting |
| **TFT** | Deep Learning (Transformer) | No | Attention mechanism, interpretable | Multi-horizon forecasting |
| **Prophet** | Statistical | No | Robust to outliers, handles holidays | Trend + seasonality |
| **DeepVAR** | Deep Learning (VAR) | Yes | Models dependencies | Multivariate forecasting |

---

## 1. Baseline Models

### Naive Forecast
- **Description**: Uses the last observed value as the forecast for all future time steps
- **Pros**: Simple, fast, good baseline
- **Cons**: No trend or seasonality
- **Best for**: Stable time series without patterns

### Seasonal Naive Forecast  
- **Description**: Uses the value from the same day of the previous week as the forecast
- **Pros**: Captures weekly seasonality
- **Cons**: No trend adjustment
- **Best for**: Time series with strong weekly patterns (like COVID-19 cases)

**Training Script**: `src/models/train_baseline.py`

---

## 2. DeepAR (Deep Autoregressive Model)

### Overview
DeepAR is a probabilistic forecasting method based on autoregressive recurrent networks. It produces probabilistic forecasts with quantile estimates.

### Architecture
- **Model Type**: LSTM-based RNN
- **Input**: Historical time series (univariate)
- **Output**: Probability distribution over future values
- **Parameters**:
  - `context_length`: 56 days (8 weeks of history)
  - `prediction_length`: 14 days
  - `hidden_size`: 40
  - `num_layers`: 2
  - `dropout_rate`: 0.1

### Advantages
- Produces full probability distributions (not just point forecasts)
- Provides confidence intervals (50%, 90%)
- Handles missing data well
- Can learn complex patterns

### Limitations
- Univariate (doesn't use mobility data)
- Requires sufficient training data
- Computationally intensive

**Training Script**: `src/models/train_deepar.py`  
**Output**: `results/deepar_forecast.png`, `results/deepar_metrics.csv`

---

## 3. Temporal Fusion Transformer (TFT)

### Overview
TFT uses attention mechanisms to identify important time steps and features, making it interpretable and powerful for multi-horizon forecasting.

### Architecture
- **Model Type**: Transformer with temporal fusion
- **Input**: Historical time series (univariate)
- **Output**: Multi-step ahead forecasts with attention weights
- **Parameters**:
  - `context_length`: 56 days
  - `prediction_length`: 14 days
  - `hidden_dim`: 32
  - `variable_dim`: 32
  - `num_heads`: 4 (multi-head attention)

### Advantages
- Attention mechanism shows which time steps are important
- Better at capturing long-term dependencies
- Interpretable feature importance
- State-of-the-art performance

### Limitations
- More complex than DeepAR
- Requires more training time
- More hyperparameters to tune

**Training Script**: `src/models/train_tft.py`  
**Output**: `results/tft_forecast.png`, `results/tft_metrics.csv`

---

## 4. Prophet

### Overview
Prophet is a decomposable time series model developed by Facebook. It breaks down the time series into trend, seasonality, and holidays components.

### Architecture
- **Model Type**: Additive regression model
- **Components**:
  - Piecewise linear trend
  - Weekly seasonality (enabled)
  - Yearly seasonality (disabled - not enough data)
- **Parameters**:
  - `growth`: 'linear'
  - `changepoint_prior_scale`: 0.05 (trend flexibility)
  - `seasonality_prior_scale`: 10.0 (seasonality strength)

### Advantages
- Robust to missing data and outliers
- Interpretable components (trend + seasonality)
- Fast training
- Handles changepoints (sudden trend changes)
- No need for data preprocessing

### Limitations
- Less flexible than deep learning models
- Assumes specific decomposition structure
- May not capture complex non-linear patterns

**Training Script**: `src/models/train_prophet.py`  
**Output**: `results/prophet_forecast.png`, `results/prophet_metrics.csv`

---

## 5. DeepVAR (Deep Vector Autoregression)

### Overview
DeepVAR extends DeepAR to multivariate time series, modeling dependencies between multiple related series.

### Architecture
- **Model Type**: Multi-variate LSTM-based RNN
- **Input**: 7 time series:
  - Daily COVID-19 cases (7-day MA)
  - Retail & recreation mobility
  - Grocery & pharmacy mobility
  - Parks mobility
  - Transit stations mobility
  - Workplaces mobility
  - Residential mobility
- **Output**: Joint probability distribution for all series
- **Parameters**:
  - `target_dim`: 7 (number of time series)
  - `context_length`: 56 days
  - `prediction_length`: 14 days
  - `hidden_dim`: 40
  - `num_layers`: 2

### Advantages
- **Models dependencies** between COVID cases and mobility data
- Can forecast all 7 series simultaneously
- Captures how mobility changes affect case trends
- Provides multivariate confidence intervals

### Limitations
- Requires mobility data for future forecasts
- More complex than univariate models
- Slower training due to multivariate nature
- Needs all series to have aligned timestamps

### How Mobility Data is Used
DeepVAR learns patterns like:
- When workplace mobility ↓ → cases ↓ (after ~2 weeks)
- When residential time ↑ → cases ↓ (lockdown effect)
- When retail/recreation ↑ → cases ↑ (increased transmission)

**Training Script**: `src/models/train_deepvar.py`  
**Output**: `results/deepvar_forecast.png`, `results/deepvar_metrics.csv`

---

## Evaluation Metrics

All models are evaluated using the following metrics:

| Metric | Description | Lower is Better |
|--------|-------------|-----------------|
| **RMSE** | Root Mean Squared Error | ✓ |
| **MAE** | Mean Absolute Error | ✓ |
| **MAPE** | Mean Absolute Percentage Error | ✓ |
| **sMAPE** | Symmetric MAPE | ✓ |
| **CRPS** | Continuous Ranked Probability Score | ✓ |
| **Coverage** | % of actuals within prediction intervals | Higher is better |

---

## Training All Models

### Individual Training
```bash
# Baseline models
python src/models/train_baseline.py

# Deep learning models
python src/models/train_deepar.py
python src/models/train_tft.py
python src/models/train_deepvar.py

# Statistical model
python src/models/train_prophet.py
```

### Full Pipeline
```bash
# Train all models
python run_pipeline.py

# Quick mode (baseline + DeepAR only)
python run_pipeline.py --quick

# Specific models only
python run_pipeline.py --steps baseline deepar tft compare
```

### Model Comparison
After training all models, compare their performance:
```bash
python src/models/compare_models.py
```

This generates:
- `results/model_comparison.csv` - Performance metrics table
- `results/model_comparison.png` - Visual comparison charts
- `results/model_rankings.csv` - Ranked by each metric
- `results/all_forecasts_comparison.png` - Side-by-side forecast plots

---

## Model Selection Guidelines

**Use DeepAR if**:
- You want probabilistic forecasts with confidence intervals
- You have sufficient historical data (6+ months)
- You need general-purpose forecasting

**Use TFT if**:
- You want to understand which time periods are important
- You need state-of-the-art performance
- Interpretability is valuable

**Use Prophet if**:
- You want fast, robust forecasts
- You need to handle missing data/outliers
- You want interpretable trend + seasonality decomposition
- You have limited computational resources

**Use WaveNet if**:
- You want CNN-based temporal modeling
- You need a large receptive field efficiently
- You're interested in dilated convolutions for time series
- You want an alternative to RNN/attention architectures

**Use DeepVAR if**:
- ⚠️ **NOT AVAILABLE** - DeepVAREstimator not in PyTorch backend
- Only available in MXNet backend (incompatible with Python 3.10+)
- Consider alternative multivariate models (MQ-CNN, TCN)
- ~~You have mobility data available~~
- ~~You want to model relationships between cases and mobility~~

---

## Hyperparameter Tuning

Key parameters to tune for each model:

### DeepAR / DeepVAR
- `context_length`: How much history to use (try 28, 56, 84 days)
- `hidden_size`: LSTM hidden units (try 20, 40, 60)
- `num_layers`: LSTM layers (try 1, 2, 3)
- `dropout_rate`: Regularization (try 0.1, 0.2, 0.3)
- `learning_rate`: Training speed (try 1e-4, 1e-3, 1e-2)

### TFT
- `hidden_dim`: Feature dimension (try 16, 32, 64)
- `num_heads`: Attention heads (try 2, 4, 8)
- `dropout_rate`: Regularization (try 0.1, 0.2)

### Prophet
- `changepoint_prior_scale`: Trend flexibility (try 0.01, 0.05, 0.5)
- `seasonality_prior_scale`: Seasonality strength (try 1.0, 10.0, 20.0)

---

## Technical Notes

### Backend
- **GluonTS**: PyTorch backend (MXNet not compatible with Python 3.10+)
- **Device**: CPU (with MPS fallback enabled for macOS)
- **Framework**: PyTorch Lightning for training

### Data Requirements
- **Minimum**: 3-6 months of daily COVID case data
- **Optimal**: 1+ years for seasonal patterns
- **Frequency**: Daily observations
- **Format**: Time series with date index

### Computational Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Training Time** (10 epochs):
  - Baseline: < 1 minute
  - Prophet: 1-2 minutes
  - DeepAR: 5-10 minutes
  - TFT: 5-10 minutes
  - DeepVAR: 10-15 minutes

---

## References

- **DeepAR**: [Salinas et al., 2020](https://arxiv.org/abs/1704.04110)
- **TFT**: [Lim et al., 2021](https://arxiv.org/abs/1912.09363)
- **Prophet**: [Taylor & Letham, 2018](https://peerj.com/preprints/3190/)
- **DeepVAR**: [Salinas et al., 2019](https://arxiv.org/abs/1910.03002)
- **GluonTS**: [Alexandrov et al., 2020](https://arxiv.org/abs/1906.05264)

---

## Next Steps

1. **Run all models**: `python run_pipeline.py`
2. **Compare results**: `python src/models/compare_models.py`
3. **Tune hyperparameters**: Modify parameters in training scripts
4. **Add covariates**: Extend DeepAR/TFT to use mobility data (future work)
5. **Ensemble models**: Combine predictions from multiple models

