# 🚀 Quick Start Guide - COVID-19 Case Prediction

This guide will get you up and running with the COVID-19 case prediction project in under 30 minutes.

---

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies

```bash
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install all requirements
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected time:** 5-10 minutes

---

### Step 2: Download Additional Data (Bonus)

You already have the core COVID-19 data. Let's get the mobility data from ActiveConclusion:

```bash
# Quick download of Google Mobility data (ActiveConclusion archive)
python3 << 'EOF'
import pandas as pd

print("Downloading Google Mobility data from ActiveConclusion...")
url = "https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv"
df = pd.read_csv(url)

# Save to data directory
df.to_csv('data/mobility_report_US.csv', index=False)

# Get stats for national level
national = df[df['state'] == 'Total']
print(f"✓ Downloaded {len(df):,} rows of mobility data")
print(f"  Date range: {national['date'].min()} to {national['date'].max()}")
print(f"  Coverage: Feb 2020 - Feb 2022 (24 months)")
EOF
```

**Expected time:** 2-3 minutes

**Note:** You may already have this file! Check if `data/mobility_report_US.csv` exists.

---

### Step 3: Verify Your Data

```bash
ls -lh data/
```

**You should see:**
```
✓ time_series_covid19_confirmed_US.csv  (~17 MB)
✓ time_series_covid19_deaths_US.csv     (~11 MB)
✓ time_series_covid19_vaccine_us.csv    (~3 MB)
✓ mobility_report_US.csv                (~97 MB) [if downloaded]
✓ 2020_US_Region_Mobility_Report.csv    (~78 MB) [can be removed]
```

---

## 📊 Your Current Progress

**What you've done so far:** ✅
- Basic data visualization (`visualize_covid_data_analysis.py`)
- Data aggregation to national level
- Created processed CSV files with daily cases/deaths

**What's next:** 🔄
- Transform data to GluonTS format
- Build forecasting models
- Evaluate predictions with uncertainty

---

## 🎯 Next Immediate Actions

### Action 1: Run Exploratory Data Analysis (15 min)

Create and run this notebook:

**File:** `notebooks/01_eda.ipynb`

```python
# Quick EDA Script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your processed data
cases_df = pd.read_csv('src/model/training/processed_us_cases.csv')
cases_df['Date'] = pd.to_datetime(cases_df['Date'])

# Basic statistics
print("Dataset Overview:")
print(f"Date range: {cases_df['Date'].min()} to {cases_df['Date'].max()}")
print(f"Total days: {len(cases_df)}")
print(f"Total cumulative cases: {cases_df['Cumulative'].iloc[-1]:,.0f}")

# Plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(cases_df['Date'], cases_df['Cumulative']/1e6)
plt.title('Cumulative COVID-19 Cases (US)')
plt.ylabel('Cases (Millions)')
plt.xlabel('Date')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(cases_df['Date'], cases_df['Daily_MA7']/1e3)
plt.title('Daily New Cases (7-day average)')
plt.ylabel('Cases (Thousands)')
plt.xlabel('Date')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Identify pandemic waves
print("\nKey Statistics:")
print(f"Peak daily cases: {cases_df['Daily'].max():,.0f} on {cases_df.loc[cases_df['Daily'].idxmax(), 'Date']}")
print(f"Recent 7-day avg: {cases_df['Daily_MA7'].iloc[-1]:,.0f}")
```

---

### Action 2: Prepare Data for GluonTS (30 min)

Create this script:

**File:** `src/data_processing/prepare_gluonts_data.py`

```python
"""
Convert processed COVID data to GluonTS format
"""
import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from pathlib import Path
import json

# Configuration
PREDICTION_LENGTH = 14  # Forecast 14 days ahead
TRAIN_END_DATE = '2022-12-31'  # Use data up to end of 2022 for training
FREQ = 'D'  # Daily frequency

# Load processed data
print("Loading processed data...")
cases_df = pd.read_csv('src/model/training/processed_us_cases.csv')
cases_df['Date'] = pd.to_datetime(cases_df['Date'])
cases_df = cases_df.sort_values('Date')

# Use daily cases (not cumulative)
# Apply smoothing to reduce noise
cases_df['Daily_Smooth'] = cases_df['Daily_MA7'].fillna(cases_df['Daily'])

# Split train/test
train_df = cases_df[cases_df['Date'] <= TRAIN_END_DATE]
test_df = cases_df[cases_df['Date'] > TRAIN_END_DATE]

print(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} days)")
print(f"Test period: {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} days)")

# Create GluonTS dataset
def create_gluonts_dataset(df, start_date):
    """Convert DataFrame to GluonTS ListDataset"""
    target = df['Daily_Smooth'].values
    
    data = [{
        "start": pd.Timestamp(start_date),
        "target": target.tolist()
    }]
    
    return ListDataset(data, freq=FREQ)

# Create train and test datasets
train_ds = create_gluonts_dataset(train_df, train_df['Date'].iloc[0])
test_ds = create_gluonts_dataset(cases_df, cases_df['Date'].iloc[0])  # Test uses all data

# Save datasets
output_dir = Path('data/processed')
output_dir.mkdir(exist_ok=True, parents=True)

# Save as JSON (GluonTS format)
with open(output_dir / 'train_data.json', 'w') as f:
    for entry in train_ds:
        json.dump({
            "start": str(entry["start"]),
            "target": entry["target"]
        }, f)
        f.write('\n')

with open(output_dir / 'test_data.json', 'w') as f:
    for entry in test_ds:
        json.dump({
            "start": str(entry["start"]),
            "target": entry["target"]
        }, f)
        f.write('\n')

print(f"\n✓ Datasets saved to {output_dir}")
print(f"  Training samples: {len(list(train_ds))}")
print(f"  Test samples: {len(list(test_ds))}")
print(f"  Prediction length: {PREDICTION_LENGTH} days")

# Save metadata
metadata = {
    "prediction_length": PREDICTION_LENGTH,
    "freq": FREQ,
    "train_start": str(train_df['Date'].iloc[0]),
    "train_end": str(train_df['Date'].iloc[-1]),
    "test_start": str(test_df['Date'].iloc[0]),
    "test_end": str(test_df['Date'].iloc[-1]),
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✓ Ready for model training!")
```

**Run it:**
```bash
python src/data_processing/prepare_gluonts_data.py
```

---

### Action 3: Train Your First Model (20 min)

Create this script:

**File:** `src/models/train_deepar.py`

```python
"""
Train DeepAR model for COVID-19 forecasting
"""
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import json
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
PREDICTION_LENGTH = 14
EPOCHS = 50  # Start with fewer epochs for quick testing
LEARNING_RATE = 1e-3
NUM_LAYERS = 2
NUM_CELLS = 40

print("=" * 60)
print("COVID-19 FORECASTING WITH DEEPAR")
print("=" * 60)

# Load data
print("\n[1/4] Loading data...")
with open('data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)

def load_gluonts_dataset(filepath, freq='D'):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entry['start'] = pd.Timestamp(entry['start'])
            data.append(entry)
    return ListDataset(data, freq=freq)

train_ds = load_gluonts_dataset('data/processed/train_data.json')
test_ds = load_gluonts_dataset('data/processed/test_data.json')

print("✓ Data loaded successfully")

# Initialize model
print("\n[2/4] Initializing DeepAR model...")
estimator = DeepAREstimator(
    freq="D",
    prediction_length=PREDICTION_LENGTH,
    context_length=PREDICTION_LENGTH * 4,  # Use 8 weeks of history
    num_layers=NUM_LAYERS,
    num_cells=NUM_CELLS,
    dropout_rate=0.1,
    trainer=Trainer(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=32,
        num_batches_per_epoch=50
    )
)

print(f"✓ Model configured:")
print(f"  Prediction length: {PREDICTION_LENGTH} days")
print(f"  Context length: {PREDICTION_LENGTH * 4} days")
print(f"  Layers: {NUM_LAYERS}, Cells: {NUM_CELLS}")

# Train model
print("\n[3/4] Training model...")
print("(This may take 5-10 minutes...)")
predictor = estimator.train(training_data=train_ds)

print("✓ Training complete!")

# Evaluate
print("\n[4/4] Evaluating model...")
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)
tss = list(ts_it)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(tss, forecasts)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"RMSE: {agg_metrics['RMSE']:.2f}")
print(f"MAE: {agg_metrics['MAE']:.2f}")
print(f"MAPE: {agg_metrics['MAPE']:.2%}")
print(f"sMAPE: {agg_metrics['sMAPE']:.2%}")

# Plot
print("\n[5/5] Creating visualization...")
plt.figure(figsize=(12, 6))

# Plot historical data
ts = tss[0]
forecast = forecasts[0]

plt.plot(ts[-60:].to_timestamp(), label='Actual', linewidth=2)
forecast.plot(prediction_intervals=[50, 90], color='g')

plt.title('COVID-19 Case Forecast (DeepAR Model)')
plt.ylabel('Daily Cases (7-day avg)')
plt.xlabel('Date')
plt.legend(['Actual', 'Forecast (median)', '50% CI', '90% CI'])
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('results/deepar_forecast.png', dpi=150)
print("✓ Plot saved to results/deepar_forecast.png")

print("\n" + "=" * 60)
print("DONE! Your first COVID-19 forecast is complete! 🎉")
print("=" * 60)
```

**Create results directory and run:**
```bash
mkdir -p results
python src/models/train_deepar.py
```

---

## 📈 Expected Results

After completing these steps, you should have:

1. ✅ **Environment set up** with all dependencies
2. ✅ **Data downloaded** and validated
3. ✅ **GluonTS datasets** created
4. ✅ **First model trained** (DeepAR)
5. ✅ **Evaluation metrics** calculated
6. ✅ **Forecast visualization** generated

**Typical Performance (14-day forecast):**
- RMSE: ~5,000 - 15,000 (depends on data period)
- MAE: ~3,000 - 10,000
- MAPE: 15% - 40%

---

## 🐛 Troubleshooting

### Issue: MXNet installation fails
**Solution:**
```bash
# Try installing with specific version
pip install mxnet==1.9.1

# Or use PyTorch backend instead
pip install torch
pip install gluonts[torch]
```

### Issue: "ModuleNotFoundError: No module named 'gluonts'"
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install gluonts
```

### Issue: Training is very slow
**Solution:**
- Reduce `epochs` to 20-30 for initial experiments
- Reduce `num_batches_per_epoch` to 30
- Use smaller model: `num_layers=2, num_cells=30`

### Issue: Model predictions are poor
**Solutions:**
- Use more training data (longer context_length)
- Smooth the target variable more aggressively
- Try different prediction_length (7 days instead of 14)
- Add covariates (mobility data)

---

## 🎓 Learning Resources

### GluonTS Tutorials
1. [Official Tutorial](https://ts.gluon.ai/stable/tutorials/forecasting/quick_start_tutorial.html)
2. [Extended Example](https://github.com/awslabs/gluonts/tree/dev/examples)

### Understanding Metrics
- **CRPS:** Lower is better, evaluates probabilistic forecast quality
- **MAE:** Mean absolute error in original units (cases)
- **MAPE:** Mean absolute percentage error (good for comparing across scales)

### Next Steps in the Full Plan
After this quick start:
1. Experiment with Transformer model
2. Add mobility data as covariates
3. Tune hyperparameters
4. Run scenario analysis
5. Create comprehensive report

---

## 📋 Checklist

Quick reference for your immediate tasks:

- [ ] Activate virtual environment
- [ ] Install requirements.txt
- [ ] Download Google mobility data (optional)
- [ ] Run EDA script/notebook
- [ ] Create data/processed/ directory
- [ ] Run prepare_gluonts_data.py
- [ ] Create results/ directory
- [ ] Run train_deepar.py
- [ ] Review results/deepar_forecast.png
- [ ] Document your findings

---

## 🚀 What's Next?

Once you've completed this quick start:

1. **Read** `PROJECT_PLAN.md` for comprehensive roadmap
2. **Explore** `DATASET_GUIDE.md` for detailed data information
3. **Experiment** with different models (Transformer, SimpleFeedForward)
4. **Enhance** by adding mobility data as covariates
5. **Analyze** uncertainty in predictions
6. **Compare** multiple models' performance

---

## 💬 Need Help?

- **GluonTS Issues:** https://github.com/awslabs/gluonts/issues
- **Documentation:** https://ts.gluon.ai/
- **Community:** Stack Overflow (tag: `gluonts`)

---

**Good luck with your project! 🎉**

Remember: Start simple, iterate quickly, and build up complexity gradually.

