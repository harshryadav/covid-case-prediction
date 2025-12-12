# GluonTS API Documentation

## Overview

GluonTS is a Python toolkit for probabilistic time series forecasting built on PyTorch and MXNet. This document describes:
1. **GluonTS Native API** - The core PyTorch estimators and predictors
2. **Our Wrapper Layer** - Simplified functions built on top of GluonTS

---

## 🎯 GluonTS Native API

### Architecture

GluonTS follows this pattern:

```
Data → Estimator → Predictor → Forecast
```

1. **Dataset**: `ListDataset` - Time series data structure
2. **Estimator**: Defines and trains the model
3. **Predictor**: Makes predictions from trained model
4. **Forecast**: Contains predictions with quantiles

### Core Components

#### 1. ListDataset (Data Structure)

```python
from gluonts.dataset.common import ListDataset

dataset = ListDataset(
    [
        {
            "start": "2020-01-01",  # Start date
            "target": [1.0, 2.0, 3.0, ...],  # Time series values
            "feat_static_cat": [0],  # Optional: static features
        }
    ],
    freq="D"  # Frequency: 'D' (daily), 'H' (hourly), etc.
)
```

#### 2. Estimators (Model Training)

**DeepAR Estimator**:
```python
from gluonts.torch.model.deepar import DeepAREstimator

estimator = DeepAREstimator(
    freq="D",
    prediction_length=14,
    context_length=60,
    num_layers=2,
    hidden_size=40,
    dropout_rate=0.1,
    lr=0.001,
    batch_size=32,
    trainer_kwargs={"max_epochs": 20}
)

predictor = estimator.train(train_dataset)
```

**SimpleFeedForward Estimator**:
```python
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

estimator = SimpleFeedForwardEstimator(
    freq="D",
    prediction_length=14,
    context_length=30,
    hidden_dims=[40],
    lr=0.001,
    batch_size=32,
    trainer_kwargs={"max_epochs": 10}
)

predictor = estimator.train(train_dataset)
```

**DeepNPTS Estimator**:
```python
from gluonts.torch.model.deep_npts import DeepNPTSEstimator

estimator = DeepNPTSEstimator(
    freq="D",
    prediction_length=14,
    context_length=30,
    hidden_dim=16,
    num_layers=2,
    lr=0.001,
    batch_size=32,
    trainer_kwargs={"max_epochs": 10}
)

predictor = estimator.train(train_dataset)
```

#### 3. Predictor (Inference)

```python
# Generate forecasts
forecast_it = predictor.predict(test_dataset)
forecasts = list(forecast_it)

# Access forecast properties
forecast = forecasts[0]
forecast.mean          # Point forecast
forecast.quantile(0.1) # 10th percentile
forecast.quantile(0.9) # 90th percentile
forecast.start_date    # Forecast start
```

---

## 🔧 Our Wrapper Layer

We provide simplified wrapper functions in `utils/gluonts_utils.py` that handle common tasks.

### Data Preparation

#### create_gluonts_dataset()

Convert pandas DataFrame to GluonTS format.

```python
from utils.gluonts_utils import create_gluonts_dataset

train_ds = create_gluonts_dataset(
    df=train_df,
    target_column='Daily_Cases_MA7',
    freq='D',
    prediction_length=14
)
```

**Parameters**:
- `df`: pandas DataFrame with Date column
- `target_column`: Column name to forecast
- `freq`: Time frequency ('D', 'H', 'W', etc.)
- `prediction_length`: Forecast horizon

**Returns**: `ListDataset` ready for training

---

### Model Training

#### train_deepar()

Train DeepAR model with optimized defaults.

```python
from utils.gluonts_utils import train_deepar

predictor = train_deepar(
    train_data=train_ds,
    prediction_length=14,
    context_length=30,
    num_layers=1,
    num_cells=20,
    epochs=10
)
```

**Key Parameters**:
- `train_data`: GluonTS ListDataset
- `prediction_length`: How many steps to forecast
- `context_length`: How many historical steps to use
- `num_layers`: Number of RNN layers
- `num_cells`: Hidden layer size
- `epochs`: Training iterations

**Returns**: Trained `Predictor` object

---

#### train_feedforward()

Train SimpleFeedForward baseline model.

```python
from utils.gluonts_utils import train_feedforward

predictor = train_feedforward(
    train_data=train_ds,
    prediction_length=14,
    context_length=30,
    hidden_dims=[20],
    epochs=8
)
```

**Key Parameters**:
- `hidden_dims`: List of hidden layer sizes
- Other params same as `train_deepar()`

**Returns**: Trained `Predictor` object

---

#### train_deepnpts()

Train DeepNPTS lightweight model.

```python
from utils.gluonts_utils import train_deepnpts

predictor = train_deepnpts(
    train_data=train_ds,
    prediction_length=14,
    context_length=30,
    hidden_dim=16,
    num_layers=2,
    epochs=10
)
```

**Key Parameters**:
- `hidden_dim`: Hidden layer dimensionality
- `num_layers`: Number of layers

**Returns**: Trained `Predictor` object

---

### Forecasting

#### generate_forecast()

Generate predictions from trained model.

```python
from utils.gluonts_utils import generate_forecast

forecasts, truths = generate_forecast(
    predictor=predictor,
    test_data=test_ds,
    num_samples=100
)
```

**Parameters**:
- `predictor`: Trained model
- `test_data`: Test dataset
- `num_samples`: Monte Carlo samples for uncertainty

**Returns**:
- `forecasts`: List of Forecast objects
- `truths`: List of ground truth time series

---

### Evaluation

#### evaluate_forecast()

Compute forecast accuracy metrics.

```python
from utils.gluonts_utils import evaluate_forecast

metrics = evaluate_forecast(
    forecast=forecasts[0],
    ground_truth=truths[0]
)

print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

**Returns**: Dictionary with keys:
- `mae`: Mean Absolute Error
- `rmse`: Root Mean Squared Error
- `mape`: Mean Absolute Percentage Error
- `crps`: Continuous Ranked Probability Score

---

### Visualization

#### plot_forecast()

Plot forecast with confidence intervals.

```python
from utils.gluonts_utils import plot_forecast

plot_forecast(
    forecast=forecasts[0],
    ground_truth=truths[0],
    title="COVID-19 Cases Forecast",
    save_path="forecast.png"
)
```

**Parameters**:
- `forecast`: Forecast object
- `ground_truth`: Actual values
- `title`: Plot title
- `save_path`: Where to save figure

**Creates**: Time series plot with 10-90% confidence bands

---

#### compare_models()

Compare multiple models side-by-side.

```python
from utils.gluonts_utils import compare_models

compare_models(
    model_metrics={
        'DeepAR': {'mae': 1050, 'rmse': 1345, 'mape': 18.4},
        'SimpleFeedForward': {'mae': 1234, 'rmse': 1523, 'mape': 21.2},
        'DeepNPTS': {'mae': 1123, 'rmse': 1434, 'mape': 19.7}
    },
    save_path="comparison.png"
)
```

**Creates**: Bar chart comparing model performance

---

### Scenario Analysis

#### scenario_analysis()

Simulate public health interventions.

```python
from utils.gluonts_utils import scenario_analysis

scenarios = {
    'No Intervention': 1.0,
    'Mild Measures': 0.85,
    'Moderate Lockdown': 0.65,
    'Strict Lockdown': 0.40
}

results = scenario_analysis(
    predictor=predictor,
    base_forecast=forecasts[0],
    scenarios=scenarios,
    prediction_length=14
)
```

**Parameters**:
- `predictor`: Trained model
- `base_forecast`: Baseline forecast
- `scenarios`: Dict of intervention multipliers
- `prediction_length`: Forecast horizon

**Returns**: Dict of adjusted forecasts per scenario

---

## 📊 Complete Workflow Example

```python
# 1. Load data
from utils.load_data_utils import load_all_data
data = load_all_data("data")

# 2. Preprocess
from utils.preprocess_data_utils import preprocess_pipeline
merged, train_df, test_df = preprocess_pipeline(
    data['cases'], data['deaths'], data['mobility']
)

# 3. Create GluonTS datasets
from utils.gluonts_utils import create_gluonts_dataset
train_ds = create_gluonts_dataset(train_df, 'Daily_Cases_MA7', 'D', 14)
test_ds = create_gluonts_dataset(test_df, 'Daily_Cases_MA7', 'D', 14)

# 4. Train model
from utils.gluonts_utils import train_deepar
predictor = train_deepar(train_ds, prediction_length=14, epochs=10)

# 5. Generate forecasts
from utils.gluonts_utils import generate_forecast
forecasts, truths = generate_forecast(predictor, test_ds)

# 6. Evaluate
from utils.gluonts_utils import evaluate_forecast
metrics = evaluate_forecast(forecasts[0], truths[0])

# 7. Visualize
from utils.gluonts_utils import plot_forecast
plot_forecast(forecasts[0], truths[0], save_path="forecast.png")
```

---

## 🔑 Key Concepts

### Probabilistic Forecasting
GluonTS generates **distributions** not just point forecasts:
- Mean prediction
- Confidence intervals (10-90%, 25-75%)
- Full quantile function

### Context vs Prediction Length
- **Context Length**: Historical data used as input
- **Prediction Length**: How far ahead to forecast
- Rule of thumb: context_length ≥ 2× prediction_length

### Frequency Strings
- `'D'`: Daily
- `'H'`: Hourly
- `'W'`: Weekly
- `'M'`: Monthly

---

## 📚 References

- [GluonTS Documentation](https://ts.gluon.ai/)
- [GluonTS GitHub](https://github.com/awslabs/gluonts)
- [GluonTS Tutorial](https://ts.gluon.ai/stable/tutorials/index.html)

---

**For complete examples**, see:
- `GluonTS_DeepAR.example.ipynb`
- `GluonTS_SimpleFeedForward.example.ipynb`
- `GluonTS_DeepNPTS.example.ipynb`

**For minimal API demos**, see:
- `GluonTS_DeepAR.API.ipynb`
- `GluonTS_SimpleFeedForward.API.ipynb`
- `GluonTS_DeepNPTS.API.ipynb`

