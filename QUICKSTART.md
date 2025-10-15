# 🚀 30-Minute Tutorial - COVID-19 Forecasting

**Goal:** Understand the project by running it step-by-step with explanations.

> **Just want to run it?** → See [GETTING_STARTED.md](GETTING_STARTED.md) for quick setup  
> **Need a command reference?** → See [QUICKREF.txt](QUICKREF.txt) for one-page guide

---

## 📖 What You'll Learn

This tutorial walks you through each step of the COVID-19 forecasting pipeline:

1. **Data Preprocessing** - Clean and merge COVID-19 datasets
2. **Time Series Preparation** - Format data for GluonTS
3. **Exploratory Analysis** - Visualize trends and patterns
4. **Baseline Modeling** - Train simple forecasting models
5. **Deep Learning** - Train DeepAR neural network model

---

## Step 1: Data Preprocessing (3 minutes)

### What It Does
- Downloads JHU CSSE COVID-19 data (cases, deaths, vaccines)
- Aggregates county-level data to national level
- Merges datasets by date
- Calculates daily new cases and 7-day moving average

### Run It
```bash
source venv/bin/activate
python src/data_processing/preprocess.py
```

### Expected Output
```
✓ Saved to data/processed/national_data.csv
```

### What To Check
```bash
head -5 data/processed/national_data.csv
# Should show: Date, Confirmed_Cases, Daily_Cases, Daily_MA7, etc.
```

**Key Insight:** We use 7-day moving average to smooth out weekly reporting patterns.

---

## Step 2: GluonTS Data Preparation (1 minute)

### What It Does
- Converts CSV data into GluonTS `ListDataset` format
- Splits into training (80%) and test (20%) sets
- Saves metadata (frequency, prediction length)

### Run It
```bash
python src/data_processing/prepare_gluonts.py
```

### Expected Output
```
✓ GluonTS datasets ready!
  Output directory: data/gluonts
  Prediction length: 14 days
  Frequency: D
```

### What To Check
```bash
cat data/gluonts/metadata.json
# Shows prediction_length: 14, freq: "D"
```

**Key Insight:** GluonTS requires specific format with start date and target time series.

---

## Step 3: Exploratory Data Analysis (2 minutes)

### What It Does
- Creates 4-panel visualization:
  1. Cumulative confirmed cases
  2. Daily new cases (7-day MA)
  3. Cumulative deaths
  4. Daily new deaths (7-day MA)

### Run It
```bash
python src/visualization/create_eda_plots.py
```

### Expected Output
```
✓ EDA visualization saved: results/eda_visualization.png
```

### View Results
```bash
open results/eda_visualization.png
```

**Key Insight:** Notice the multiple waves - Delta (mid-2021) and Omicron (winter 2021-22).

---

## Step 4: Baseline Models (1 minute)

### What It Does
- **Naive Forecast:** Uses last known value as prediction
- **Seasonal Naive:** Uses value from 7 days ago (weekly seasonality)
- Calculates metrics: MAE, RMSE, MAPE

### Run It
```bash
python src/models/train_baseline.py
```

### Expected Output
```
[1/2] Training Naive Forecaster...
  MAE:  8284.50
  RMSE: 8694.53
  MAPE: 23.29%

[2/2] Training Seasonal Naive Forecaster (7-day)...
  MAE:  8579.56
  RMSE: 9108.59
  MAPE: 24.01%
```

### View Results
```bash
open results/baseline_forecasts.png
cat results/baseline_metrics.csv
```

**Key Insight:** Simple baselines provide benchmark - our deep learning model should beat these!

---

## Step 5: DeepAR Neural Network (10-15 minutes)

### What It Does
- Trains autoregressive recurrent neural network
- Uses PyTorch backend (not MXNet - for Python 3.10 compatibility)
- Generates probabilistic forecasts with uncertainty intervals
- 10 epochs on CPU (macOS MPS has compatibility issues)

### Run It
```bash
python src/models/train_deepar.py
```

### Expected Output (abbreviated)
```
[2/4] Initializing DeepAR model (PyTorch backend on CPU)...
✓ Model initialized

[3/4] Training model...
(This may take 5-10 minutes...)
Epoch 0: ... train_loss=12.3
Epoch 1: ... train_loss=11.8
...
Epoch 9: ... train_loss=9.4
✓ Training complete!

[4/4] Evaluating model...

============================================================
EVALUATION RESULTS
============================================================
RMSE                : 5126.83
MAPE                : 12.19%
sMAPE               : 11.09%

✓ Plot saved to results/deepar_forecast.png
✓ Metrics saved to results/deepar_metrics.csv
```

### View Results
```bash
open results/deepar_forecast.png
cat results/deepar_metrics.csv
```

**Key Insight:** 
- DeepAR RMSE: 5,127 vs Baseline RMSE: 8,695 (41% improvement!)
- DeepAR MAPE: 12.19% vs Baseline MAPE: 23.29% (48% better!)

---

## 🎯 Understanding the Results

### Metrics Explained

**RMSE (Root Mean Square Error):**
- Average prediction error in actual case counts
- Lower is better
- DeepAR: ~5,127 daily cases error

**MAPE (Mean Absolute Percentage Error):**
- Average error as percentage of actual values
- Lower is better
- DeepAR: ~12% error on average

**Coverage Metrics:**
- `Coverage[0.9]: 1.00` = 100% of actual values fall within 90% confidence interval
- Indicates model properly quantifies uncertainty

### Why DeepAR Is Better

1. **Learns patterns** - Captures trends, seasonality, autocorrelation
2. **Probabilistic** - Provides uncertainty estimates, not just point forecasts
3. **Autoregressive** - Uses its own predictions to forecast further ahead
4. **Deep learning** - Can model complex non-linear relationships

---

## 📊 Full Pipeline (All at Once)

```bash
source venv/bin/activate
python run_pipeline.py
```

Runs all 5 steps automatically. Takes ~15-20 minutes.

---

## 🔍 What's Actually Implemented

### ✅ Completed Features

- [x] Data ingestion from JHU CSSE
- [x] Preprocessing and aggregation
- [x] GluonTS data formatting
- [x] Exploratory data analysis
- [x] Baseline forecasting (Naive, Seasonal Naive)
- [x] DeepAR probabilistic forecasting
- [x] Evaluation metrics (RMSE, MAE, MAPE)
- [x] Uncertainty quantification
- [x] Visualization with confidence intervals
- [x] Docker support
- [x] PyTorch backend (Python 3.10+ compatible)

### 🔄 Not Yet Implemented (Future Work)

- [ ] Transformer models
- [ ] Gaussian Process models
- [ ] N-BEATS models
- [ ] State-level forecasting (currently national only)
- [ ] Mobility data as covariates (data available but not used)
- [ ] Scenario analysis with interventions
- [ ] Oxford COVID Policy Tracker integration
- [ ] Multi-horizon forecasting (7, 14, 30 days)
- [ ] Cross-validation
- [ ] Hyperparameter tuning

---

## 💡 Next Steps

### Beginner
1. ✓ Run the tutorial (you just did!)
2. Compare baseline vs DeepAR plots side-by-side
3. Read [MODEL_GUIDE.md](MODEL_GUIDE.md) to understand architecture

### Intermediate
1. Increase training epochs in `src/models/train_deepar.py`:
   ```python
   EPOCHS = 50  # Change from 10
   ```
2. Experiment with model parameters:
   - `CONTEXT_LENGTH` (historical window)
   - `hidden_size` (model capacity)
   - `num_layers` (network depth)

### Advanced
1. Add state-level forecasting (50 series instead of 1)
2. Incorporate mobility data as dynamic covariates
3. Implement Transformer or N-BEATS models
4. Set up cross-validation framework

---

## 🐛 Common Issues

**Issue:** "ModuleNotFoundError: No module named 'gluonts'"
```bash
source venv/bin/activate  # Always activate first!
pip install -r requirements.txt
```

**Issue:** Training is slow
- Expected: 10-15 minutes on CPU
- Using CPU because macOS MPS doesn't support all PyTorch operations
- To speed up: Reduce `EPOCHS` or use GPU-enabled machine

**Issue:** Out of memory
- Edit `src/models/train_deepar.py`:
  ```python
  batch_size=16  # Reduce from 32
  EPOCHS=5       # Reduce from 10
  ```

---

## ✅ Summary

You now understand:
- ✓ How COVID-19 data is processed
- ✓ What GluonTS format looks like
- ✓ How baseline models work
- ✓ How DeepAR neural network forecasts
- ✓ Why DeepAR outperforms baselines (41% better RMSE)

**Key Takeaway:** Probabilistic forecasting with deep learning provides both accurate predictions AND uncertainty quantification - crucial for pandemic planning!

---

## 📚 Documentation Index

| File | Purpose |
|------|---------|
| **QUICKSTART.md** | 👈 Tutorial (you are here) |
| **GETTING_STARTED.md** | Quick setup guide |
| **QUICKREF.txt** | One-page command reference |
| **README.md** | Project overview |
| **MODEL_GUIDE.md** | Model architecture details |
| **DATASET_GUIDE.md** | Data sources & structure |
| **PROJECT_PLAN.md** | 7-week implementation plan |

---

**Congratulations! You've completed the tutorial.** 🎉
