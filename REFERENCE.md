# Quick Reference Card

Essential commands and information at a glance.

---

## 🚀 Quick Commands

### Setup

```bash
# Clone/navigate to project
cd /path/to/covid-case-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline

```bash
# Full pipeline (local)
python run_pipeline.py

# Full pipeline (Docker)
docker-compose up -d
docker-compose exec covid-forecasting python run_pipeline.py
docker-compose down
```

### Individual Scripts

```bash
# Data processing
python src/data_processing/preprocess.py
python src/data_processing/prepare_gluonts.py

# Visualization
python src/visualization/create_eda_plots.py

# Model training
python src/models/train_baseline.py
python src/models/train_deepar.py
```

---

## 📁 File Locations

### Data

```
data/
├── time_series_covid19_confirmed_US.csv    # JHU cases ✅
├── time_series_covid19_deaths_US.csv       # JHU deaths ✅
├── time_series_covid19_vaccine_us.csv      # JHU vaccines ✅
└── mobility_report_US.csv                  # Google Mobility ✅
```

### Generated Files

```
data/processed/national_data.csv            # Preprocessed data
data/gluonts/train/data.json                # Training set
data/gluonts/test/data.json                 # Test set
```

### Results

```
results/
├── eda_visualization.png                   # EDA plots
├── baseline_forecasts.png                  # Baseline model
├── deepar_forecast.png                     # DeepAR model
└── metrics.json                            # Evaluation metrics
```

### Models

```
models/
├── baseline_model.pkl                      # Saved baseline
└── deepar_model/                           # DeepAR checkpoint
```

---

## 🐳 Docker Commands

### Build & Run

```bash
docker-compose build                         # Build image
docker-compose up -d                         # Start container
docker-compose exec covid-forecasting bash   # Enter shell
docker-compose down                          # Stop container
```

### Execute Scripts

```bash
# Run pipeline
docker-compose exec covid-forecasting python run_pipeline.py

# Train specific model
docker-compose exec covid-forecasting python src/models/train_deepar.py

# Create visualizations
docker-compose exec covid-forecasting python src/visualization/create_eda_plots.py
```

### Management

```bash
docker ps                                    # List containers
docker-compose logs -f                       # View logs
docker-compose restart                       # Restart
docker system prune -a                       # Clean up
```

---

## 📊 Datasets at a Glance

| Dataset | Source | Period | Granularity | Status |
|---------|--------|--------|-------------|--------|
| **US Cases** | JHU CSSE | Jan 2020 - Mar 2023 | County | ✅ Available |
| **US Deaths** | JHU CSSE | Jan 2020 - Mar 2023 | County | ✅ Available |
| **US Vaccines** | JHU CSSE | Dec 2020 - Mar 2023 | State | ✅ Available |
| **Mobility** | Google/ActiveConclusion | Feb 2020 - Feb 2022 | State | ✅ Available |

**Download Mobility (if missing):**
```bash
curl -o data/mobility_report_US.csv \
  https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv
```

---

## 🤖 Models Overview

| Model | Type | Training Time | Use Case |
|-------|------|---------------|----------|
| **Naive** | Baseline | Instant | Simple benchmark |
| **Seasonal Naive** | Baseline | Instant | Weekly patterns |
| **DeepAR** | Neural Network | 2-5 min (CPU) | Probabilistic forecasts |

### Model Files

```bash
src/models/train_baseline.py                # Naive models
src/models/train_deepar.py                  # DeepAR model
```

---

## 📈 Evaluation Metrics

| Metric | Formula | Purpose | Lower is Better |
|--------|---------|---------|-----------------|
| **MAE** | Mean Absolute Error | Average error | ✅ |
| **RMSE** | Root Mean Square Error | Penalizes large errors | ✅ |
| **MAPE** | Mean Absolute % Error | Relative error | ✅ |
| **CRPS** | Continuous Ranked Probability Score | Probabilistic accuracy | ✅ |

**Code:**
```python
from src.evaluation.metrics import calculate_metrics

metrics = calculate_metrics(actual, forecast, metric_names=['mae', 'rmse', 'crps'])
```

---

## 🔧 Configuration

### Key Parameters

```python
# In src/utils/config.py or src/models/train_deepar.py

PREDICTION_LENGTH = 14        # Forecast horizon (days)
CONTEXT_LENGTH = 30           # Historical context (days)
TRAIN_TEST_SPLIT = 0.8        # 80% train, 20% test

# DeepAR hyperparameters
NUM_LAYERS = 2                # LSTM layers
NUM_CELLS = 40                # Hidden units
DROPOUT_RATE = 0.1            # Regularization
EPOCHS = 10                   # Training iterations
LEARNING_RATE = 1e-3          # Optimization step size
BATCH_SIZE = 32               # Samples per batch
```

### Modify for Experiments

```python
# For longer forecasts
PREDICTION_LENGTH = 30

# For more complex model
NUM_LAYERS = 3
NUM_CELLS = 60

# For longer training
EPOCHS = 20
```

---

## 🎯 Common Tasks

### Task: Create EDA Plots

```bash
python src/visualization/create_eda_plots.py
open results/eda_visualization.png
```

### Task: Train All Models

```bash
python src/models/train_baseline.py
python src/models/train_deepar.py
```

### Task: Compare Model Performance

```python
from src.evaluation.metrics import calculate_metrics

# After training both models
baseline_metrics = calculate_metrics(actual, baseline_forecast)
deepar_metrics = calculate_metrics(actual, deepar_forecast)

print(f"Baseline CRPS: {baseline_metrics['crps']:.2f}")
print(f"DeepAR CRPS: {deepar_metrics['crps']:.2f}")
```

### Task: Forecast Deaths Instead of Cases

Edit `src/data_processing/prepare_gluonts.py`:
```python
# Change target column
train_ds, test_ds = create_gluonts_dataset(
    df,
    target_col='daily_deaths',  # Changed from 'daily_cases'
    prediction_length=14
)
```

---

## 📚 Documentation Structure

| File | Purpose |
|------|---------|
| **README.md** | Project overview & quick start |
| **QUICKSTART.md** | 30-minute hands-on tutorial |
| **DATASET_GUIDE.md** | Complete data documentation |
| **MODEL_GUIDE.md** | Model architecture & training |
| **CODE_REFERENCE.md** | Code structure & API |
| **DOCKER.md** | Docker setup & usage |
| **REFERENCE.md** | This file (quick commands) |
| **PROJECT_PLAN.md** | Detailed 7-week roadmap |

---

## 🐛 Troubleshooting

### Issue: Module not found

```bash
# Make sure you're in project root
cd /path/to/covid-case-prediction

# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Data files missing

```bash
# Check data directory
ls -lh data/

# Should see:
#   time_series_covid19_confirmed_US.csv
#   time_series_covid19_deaths_US.csv
#   time_series_covid19_vaccine_us.csv
#   mobility_report_US.csv
```

### Issue: Docker build fails

```bash
# Clean up Docker
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

### Issue: Slow training

```bash
# Check if using GPU
python -c "import mxnet as mx; print(mx.test_utils.list_gpus())"

# Reduce model complexity
# Edit src/models/train_deepar.py:
#   num_layers=1
#   num_cells=20
#   epochs=5
```

---

## 🔗 Useful Links

| Resource | URL |
|----------|-----|
| **GluonTS Docs** | https://ts.gluon.ai/ |
| **GluonTS API** | https://ts.gluon.ai/stable/api/gluonts/ |
| **DeepAR Paper** | https://arxiv.org/abs/1704.04110 |
| **JHU CSSE Data** | https://github.com/CSSEGISandData/COVID-19 |
| **Google Mobility** | https://github.com/ActiveConclusion/COVID19_mobility |

---

## ⚡ One-Liners

```bash
# Complete pipeline (local)
python run_pipeline.py && open results/baseline_forecasts.png

# Complete pipeline (Docker)
docker-compose up -d && docker-compose exec covid-forecasting python run_pipeline.py && docker-compose down

# Quick EDA
python src/visualization/create_eda_plots.py && open results/eda_visualization.png

# Train and evaluate DeepAR
python src/models/train_deepar.py && open results/deepar_forecast.png
```

---

## 🎓 Learning Path

1. **Start**: Read [README.md](README.md)
2. **Tutorial**: Follow [QUICKSTART.md](QUICKSTART.md)
3. **Data**: Understand [DATASET_GUIDE.md](DATASET_GUIDE.md)
4. **Models**: Study [MODEL_GUIDE.md](MODEL_GUIDE.md)
5. **Code**: Explore [CODE_REFERENCE.md](CODE_REFERENCE.md)
6. **Docker**: Learn [DOCKER.md](DOCKER.md)
7. **Plan**: Review [PROJECT_PLAN.md](PROJECT_PLAN.md)

---

## ✅ Quick Checklist

### First Time Setup
- [ ] Python 3.8+ installed
- [ ] Clone/download project
- [ ] Create virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Verify data files exist in `data/`

### Run Pipeline
- [ ] Activate virtual environment
- [ ] Run `python run_pipeline.py`
- [ ] Check `results/` folder for outputs

### Docker Setup
- [ ] Docker Desktop installed
- [ ] Run `docker-compose build`
- [ ] Run `docker-compose up -d`
- [ ] Execute `docker-compose exec covid-forecasting python run_pipeline.py`

---

**Print this page for quick reference!**

