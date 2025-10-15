# Model Architecture & Training Guide

Complete guide to the forecasting models, GluonTS framework, and training process.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [GluonTS Framework](#gluonts-framework)
3. [Baseline Models](#baseline-models)
4. [DeepAR Model](#deepar-model)
5. [Training Process](#training-process)
6. [Model Evaluation](#model-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Adding New Models](#adding-new-models)

---

## Overview

This project implements **probabilistic time series forecasting** using multiple models:

| Model | Type | Complexity | Training Time | Use Case |
|-------|------|------------|---------------|----------|
| **Naive** | Baseline | Simple | Instant | Benchmark |
| **Seasonal Naive** | Baseline | Simple | Instant | Weekly patterns |
| **DeepAR** | Neural Network | Advanced | 2-5 min | Probabilistic forecasting |
| **Transformer** | Neural Network | Complex | 10-15 min | Future work |

---

## GluonTS Framework

### What is GluonTS?

[GluonTS](https://ts.gluon.ai/) is a Python toolkit for probabilistic time series modeling built on Apache MXNet. It provides:

- ✅ Pre-built probabilistic models (DeepAR, Transformer, etc.)
- ✅ Standardized data format (ListDataset)
- ✅ Built-in evaluation metrics (CRPS, QuantileLoss)
- ✅ Training utilities (Trainer, early stopping)
- ✅ Prediction intervals (10th, 50th, 90th percentiles)

### Key Concepts

#### 1. ListDataset Format

GluonTS requires data in a specific format:

```python
from gluonts.dataset.common import ListDataset

# Each time series is a dictionary
dataset = ListDataset(
    [
        {
            'start': pd.Timestamp('2020-01-22'),
            'target': [10, 15, 22, 35, ...],  # Historical values
            'feat_dynamic_real': [[...], [...]]  # Optional: covariates
        },
        # More time series...
    ],
    freq='D'  # Daily frequency
)
```

#### 2. Prediction Length

```python
prediction_length = 14  # Forecast 14 days ahead
```

#### 3. Context Length

How much historical data the model uses:

```python
context_length = 30  # Use 30 days of history
```

---

## Baseline Models

### 1. Naive Forecast

**Logic**: Last observed value repeated forward

```python
forecast[t+1:t+h] = actual[t]
```

**Use Case**: Simplest benchmark

**Code Location**: `src/models/train_baseline.py`

**Example**:
```python
from src.models.train_baseline import train_baseline_models

models, forecasts = train_baseline_models(
    train_data=train_ds,
    test_data=test_ds,
    prediction_length=14
)
```

### 2. Seasonal Naive Forecast

**Logic**: Value from one seasonal period ago

```python
forecast[t+1] = actual[t - seasonal_period]
# For weekly pattern: seasonal_period = 7
```

**Use Case**: Captures weekly COVID-19 reporting patterns

**Advantages**:
- ✅ Captures seasonality
- ✅ Fast to compute
- ✅ Good baseline for comparison

---

## DeepAR Model

### Architecture

DeepAR is a **Recurrent Neural Network (RNN)** that produces probabilistic forecasts.

```
Input Sequence
     ↓
 Embedding Layer (optional)
     ↓
 LSTM/GRU Layers (stacked)
     ↓
 Output Layer
     ↓
 Distribution Parameters (μ, σ)
     ↓
 Sample Predictions
```

### Key Features

1. **Autoregressive**: Uses past predictions as input for future steps
2. **Probabilistic**: Outputs distribution (not just point forecast)
3. **Global Model**: Trained on multiple time series simultaneously
4. **Covariates**: Can incorporate external features

### Implementation

**File**: `src/models/train_deepar.py`

```python
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

estimator = DeepAREstimator(
    freq="D",                      # Daily data
    prediction_length=14,          # Forecast horizon
    context_length=30,             # Historical context
    num_layers=2,                  # LSTM layers
    num_cells=40,                  # Hidden units per layer
    dropout_rate=0.1,              # Regularization
    use_feat_dynamic_real=True,    # Use mobility covariates
    trainer=Trainer(
        epochs=10,
        learning_rate=1e-3,
        batch_size=32
    )
)

# Train model
predictor = estimator.train(train_data)

# Make predictions
forecast_it = predictor.predict(test_data)
forecasts = list(forecast_it)
```

### Model Parameters

#### Architecture Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_layers` | 2 | 1-4 | Number of LSTM layers |
| `num_cells` | 40 | 10-100 | Hidden units per layer |
| `dropout_rate` | 0.1 | 0-0.5 | Dropout for regularization |
| `context_length` | 30 | 7-60 | Days of historical context |

#### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 10 | 5-50 | Training iterations |
| `learning_rate` | 1e-3 | 1e-5 to 1e-2 | Step size for optimization |
| `batch_size` | 32 | 16-128 | Samples per batch |

### Probability Distribution

DeepAR outputs a **Negative Binomial distribution** for count data:

```python
P(y_t | history) = NegativeBinomial(μ_t, α_t)
```

Where:
- `μ_t` = mean (predicted value)
- `α_t` = dispersion (uncertainty)

---

## Training Process

### Step-by-Step Training

#### 1. Data Preparation

```bash
# Preprocess raw data
python src/data_processing/preprocess.py

# Convert to GluonTS format
python src/data_processing/prepare_gluonts.py
```

Output:
- `data/gluonts/train/data.json` - Training set
- `data/gluonts/test/data.json` - Test set

#### 2. Train Baseline Models

```bash
python src/models/train_baseline.py
```

**What it does:**
- Loads GluonTS datasets
- Creates Naive & Seasonal Naive predictors
- Generates forecasts
- Evaluates metrics
- Saves visualizations

**Output:**
- `results/baseline_forecasts.png`
- `models/baseline_model.pkl`

#### 3. Train DeepAR Model

```bash
python src/models/train_deepar.py
```

**What it does:**
- Initializes DeepAREstimator
- Trains on GPU (if available) or CPU
- Generates probabilistic forecasts
- Evaluates with CRPS
- Saves model and plots

**Output:**
- `results/deepar_forecast.png`
- `models/deepar_model/`

### Training Time

| Model | CPU | GPU | Dataset Size |
|-------|-----|-----|--------------|
| Naive | <1s | <1s | Any |
| Seasonal Naive | <1s | <1s | Any |
| DeepAR (10 epochs) | 2-5 min | 30-60s | ~700 days |

### GPU Training

To use GPU:

```bash
# Check GPU availability
python -c "import mxnet as mx; print(mx.test_utils.list_gpus())"

# Train with GPU
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
python src/models/train_deepar.py
```

---

## Model Evaluation

### Metrics

All models are evaluated using:

#### 1. CRPS (Continuous Ranked Probability Score)

**Best for probabilistic forecasts**

```python
CRPS = ∫ (F(y) - 1{y ≥ observation})² dy
```

Lower is better. Measures quality of entire predicted distribution.

#### 2. MAE (Mean Absolute Error)

```python
MAE = (1/n) Σ |actual - predicted|
```

#### 3. RMSE (Root Mean Square Error)

```python
RMSE = sqrt((1/n) Σ (actual - predicted)²)
```

#### 4. MAPE (Mean Absolute Percentage Error)

```python
MAPE = (100/n) Σ |actual - predicted| / actual
```

### Evaluation Code

```python
from src.evaluation.metrics import calculate_metrics

metrics = calculate_metrics(
    actual_values=test_data,
    forecast_object=forecast,
    metric_names=['mae', 'rmse', 'mape', 'crps']
)

print(metrics)
# {'mae': 1234.5, 'rmse': 2345.6, 'mape': 5.67, 'crps': 890.1}
```

### Baseline Comparison

| Model | MAE | RMSE | CRPS | Interpretation |
|-------|-----|------|------|----------------|
| Naive | 5000 | 7000 | 6000 | Simple benchmark |
| Seasonal Naive | 3500 | 5000 | 4500 | Better (captures weekly pattern) |
| DeepAR | 2500 | 3500 | 2800 | **Best** (learns complex patterns) |

---

## Hyperparameter Tuning

### Manual Tuning

Edit parameters in `src/models/train_deepar.py`:

```python
estimator = DeepAREstimator(
    num_layers=3,          # Try: 1, 2, 3, 4
    num_cells=60,          # Try: 20, 40, 60, 80
    dropout_rate=0.2,      # Try: 0.0, 0.1, 0.2, 0.3
    epochs=20,             # Try: 10, 20, 30, 50
    learning_rate=0.001,   # Try: 0.0001, 0.001, 0.01
)
```

### Automated Tuning (Future)

Using Optuna:

```python
import optuna

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 4)
    num_cells = trial.suggest_int('num_cells', 20, 100)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Train model with these params
    # Return validation CRPS
    return crps_score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Tips for Tuning

1. **Start simple**: Baseline params first
2. **One at a time**: Change one parameter, observe effect
3. **Validation set**: Don't tune on test set
4. **Early stopping**: Prevent overfitting
5. **Learning curve**: Plot training loss over epochs

---

## Adding New Models

### 1. Transformer Model

**File**: `src/models/train_transformer.py`

```python
from gluonts.model.transformer import TransformerEstimator

estimator = TransformerEstimator(
    freq="D",
    prediction_length=14,
    context_length=60,          # Longer context
    num_heads=4,                # Attention heads
    model_dim=32,               # Model dimension
    trainer=Trainer(epochs=20)
)

predictor = estimator.train(train_data)
```

**Advantages**:
- ✅ Captures long-range dependencies
- ✅ Attention mechanism
- ✅ Parallel training

**Disadvantages**:
- ❌ Slower to train
- ❌ More complex
- ❌ Requires more data

### 2. Gaussian Process Model

```python
from gluonts.model.gp_forecaster import GaussianProcessEstimator

estimator = GaussianProcessEstimator(
    freq="D",
    prediction_length=14,
    cardinality=[1],
    trainer=Trainer(epochs=10)
)
```

**Advantages**:
- ✅ Smooth uncertainty estimates
- ✅ Works with small data

### 3. Ensemble Model

Combine multiple models:

```python
forecasts_deepar = deepar_predictor.predict(test_data)
forecasts_transformer = transformer_predictor.predict(test_data)

# Average predictions
ensemble_forecast = (forecasts_deepar + forecasts_transformer) / 2
```

---

## Model Comparison

### When to Use Each Model

#### Naive / Seasonal Naive
- ✅ Quick baseline
- ✅ Simple interpretation
- ✅ No training required
- ❌ Poor accuracy

#### DeepAR
- ✅ Good accuracy
- ✅ Probabilistic forecasts
- ✅ Handles covariates
- ⚠️ Requires training time

#### Transformer (Future)
- ✅ Best for long sequences
- ✅ Attention mechanism
- ❌ Slower training
- ❌ Needs more data

---

## Troubleshooting

### Common Issues

#### 1. NaN in Predictions

**Cause**: Scaling issues, learning rate too high

**Fix**:
```python
# Add normalization
from gluonts.transform import Transformation, AddObservedValuesIndicator

# Lower learning rate
trainer=Trainer(learning_rate=1e-4)
```

#### 2. Overfitting

**Symptoms**: Training loss ↓ but validation loss ↑

**Fix**:
```python
# Increase dropout
dropout_rate=0.3

# Early stopping
trainer=Trainer(patience=5)

# Less complex model
num_layers=1
```

#### 3. Slow Training

**Fix**:
```python
# Reduce batch size
batch_size=16

# Fewer epochs
epochs=5

# Smaller model
num_cells=20
```

---

## Best Practices

### 1. Data Preprocessing
- ✅ Handle missing values
- ✅ Remove outliers
- ✅ Normalize covariates
- ✅ Check for data leakage

### 2. Training
- ✅ Start with baseline models
- ✅ Use validation set
- ✅ Monitor training loss
- ✅ Save checkpoints

### 3. Evaluation
- ✅ Use multiple metrics
- ✅ Visual inspection of forecasts
- ✅ Compare to baselines
- ✅ Check uncertainty calibration

### 4. Production
- ✅ Version control models
- ✅ Log predictions
- ✅ Monitor performance
- ✅ Retrain regularly

---

## Further Reading

- [GluonTS Documentation](https://ts.gluon.ai/)
- [DeepAR Paper](https://arxiv.org/abs/1704.04110)
- [Probabilistic Forecasting](https://otexts.com/fpp3/)
- [Time Series Best Practices](https://github.com/microsoft/forecasting)

---

**Ready to train? Run:**

```bash
python src/models/train_baseline.py
python src/models/train_deepar.py
```

