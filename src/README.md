# Source Code Structure

This directory contains all the source code for the COVID-19 case prediction project.

## 📁 Directory Structure

```
src/
├── __init__.py
├── README.md (this file)
│
├── data_processing/          # Data loading and preprocessing
│   ├── __init__.py
│   ├── load_data.py          # Load CSV files
│   ├── preprocess.py         # Aggregate and clean data
│   └── prepare_gluonts.py    # Convert to GluonTS format
│
├── models/                   # Model training scripts
│   ├── __init__.py
│   ├── train_baseline.py     # Baseline models (Naive, Seasonal Naive)
│   └── train_deepar.py       # DeepAR model
│
├── evaluation/               # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py            # MAE, RMSE, MAPE, etc.
│
├── visualization/            # Plotting utilities
│   ├── __init__.py
│   └── plot_utils.py         # Reusable plotting functions
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── config.py             # Configuration settings
│
└── model/                    # Legacy code (Phase 1)
    ├── prepare_time_series.py
    ├── visualize_covid_data_analysis.py
    └── training/
        ├── processed_us_cases.csv
        ├── processed_us_deaths.csv
        └── us_covid_analysis.png
```

## 🚀 Quick Start

### Step 1: Load and Preprocess Data

```bash
# Load raw data and aggregate to national level
python src/data_processing/preprocess.py
```

**Output:** `data/processed/national_data.csv`

### Step 2: Prepare for GluonTS

```bash
# Convert to GluonTS format
python src/data_processing/prepare_gluonts.py
```

**Output:** `data/gluonts/` directory with train/test datasets

### Step 3: Train Baseline Models

```bash
# Train Naive and Seasonal Naive models
python src/models/train_baseline.py
```

**Output:** `results/baseline_forecasts.png` and `results/baseline_metrics.csv`

### Step 4: Train DeepAR Model

```bash
# Train DeepAR (requires GluonTS installed)
python src/models/train_deepar.py
```

**Output:** `results/deepar_forecast.png` and `results/deepar_metrics.csv`

## 📊 Module Descriptions

### data_processing/

**Purpose:** Load, clean, and prepare data for modeling

**Files:**
- `load_data.py` - DataLoader class for loading all CSV files
- `preprocess.py` - Aggregate to national level, calculate daily cases, merge with mobility
- `prepare_gluonts.py` - Convert pandas DataFrames to GluonTS ListDataset format

**Key Functions:**
```python
from src.data_processing.load_data import DataLoader
from src.data_processing.preprocess import aggregate_to_national

loader = DataLoader()
cases_df = loader.load_cases()
national_cases = aggregate_to_national(cases_df)
```

### models/

**Purpose:** Train forecasting models

**Files:**
- `train_baseline.py` - Simple baseline models (Naive, Seasonal Naive)
- `train_deepar.py` - DeepAR probabilistic forecasting model

**To add more models:**
1. Create new file: `train_<model_name>.py`
2. Follow the structure of `train_deepar.py`
3. Use configuration from `utils/config.py`

### evaluation/

**Purpose:** Calculate forecast evaluation metrics

**Files:**
- `metrics.py` - Standard metrics (MAE, RMSE, MAPE, sMAPE)

**Usage:**
```python
from src.evaluation.metrics import calculate_all_metrics

metrics = calculate_all_metrics(actual, predicted)
print(metrics)  # {'MAE': 100.5, 'RMSE': 150.2, ...}
```

### visualization/

**Purpose:** Create plots and visualizations

**Files:**
- `plot_utils.py` - Reusable plotting functions

**Usage:**
```python
from src.visualization.plot_utils import plot_time_series, save_figure

fig, ax = plot_time_series(df, date_col='Date', value_col='Daily_MA7')
save_figure(fig, 'my_plot.png')
```

### utils/

**Purpose:** Configuration and helper utilities

**Files:**
- `config.py` - Central configuration (paths, model parameters)

**Usage:**
```python
from src.utils.config import MODEL_PARAMS, DATA_DIR

print(MODEL_PARAMS['prediction_length'])  # 14
print(DATA_DIR)  # Path to data directory
```

## 🔧 Dependencies

**Required:**
- pandas
- numpy
- matplotlib
- seaborn

**For GluonTS models:**
- gluonts
- mxnet

Install with:
```bash
pip install -r requirements.txt
```

## 📝 Adding New Components

### Add a New Preprocessing Step

1. Create file in `data_processing/`
2. Import and use in `preprocess.py`
3. Update `prepare_gluonts.py` if needed

### Add a New Model

1. Create `models/train_<model_name>.py`
2. Follow this structure:
   ```python
   # Load data
   # Initialize model
   # Train
   # Evaluate
   # Save results
   ```
3. Use configuration from `utils/config.py`

### Add New Metrics

1. Add function to `evaluation/metrics.py`
2. Use in model training scripts

### Add New Plots

1. Add function to `visualization/plot_utils.py`
2. Use in model training or notebook

## 🎯 Next Steps

1. **Explore the data:**
   ```bash
   python src/data_processing/load_data.py
   ```

2. **Train baseline models:**
   ```bash
   python src/models/train_baseline.py
   ```

3. **Set up GluonTS:**
   ```bash
   pip install gluonts mxnet
   python src/models/train_deepar.py
   ```

4. **Create notebooks** for exploration:
   - See `notebooks/` directory
   - Start with EDA (Exploratory Data Analysis)

## 📚 Related Documentation

- Main README: `../README.md`
- Quick Start Guide: `../QUICK_START.md`
- Complete Plan: `../PROJECT_PLAN.md`
- Dataset Guide: `../DATASET_GUIDE.md`

## 💡 Tips

1. **Start simple:** Run baseline models first to understand the data
2. **Incremental development:** Add features one at a time
3. **Version control:** Commit frequently
4. **Document:** Add docstrings to your functions
5. **Test:** Verify each step produces expected output

## 🐛 Troubleshooting

**Error: "File not found"**
- Check you're running from project root
- Verify data files exist in `data/` directory

**Error: "Module not found"**
- Install missing packages: `pip install <package>`
- Check virtual environment is activated

**Error: "GluonTS not installed"**
- Install: `pip install gluonts mxnet`
- For PyTorch backend: `pip install gluonts[torch]`

---

**Happy Coding! 🚀**

