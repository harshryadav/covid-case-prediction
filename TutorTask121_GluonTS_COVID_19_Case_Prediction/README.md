# COVID-19 Case Prediction using GluonTS

**MSML610 Fall 2025 - Class Project**

> Probabilistic time series forecasting of COVID-19 cases using GluonTS with PyTorch backend.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![GluonTS](https://img.shields.io/badge/GluonTS-0.14.0-green.svg)](https://ts.gluon.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

This project implements **end-to-end probabilistic forecasting** for COVID-19 daily cases using three GluonTS models with comprehensive evaluation and scenario analysis.

**Goal**: Forecast 14 days ahead with uncertainty quantification for public health decision support

**Dataset**:
- Johns Hopkins CSSE COVID-19 Data (cases, deaths)
- Google COVID-19 Community Mobility Reports (6 metrics)
- CDC Vaccination Data (kept for future use, not currently used)

**Models Implemented**:
1. **DeepAR** - Autoregressive RNN with external features (most sophisticated)
2. **SimpleFeedForward** - Baseline MLP for quick benchmarking (fastest)
3. **DeepNPTS** - Non-parametric forecasting for regime changes (most flexible)

**Key Features**:
- ✅ Probabilistic forecasts with confidence intervals
- ✅ Multi-feature learning (cases, deaths, mobility, CFR)
- ✅ Scenario analysis for public health interventions
- ✅ CPU-optimized (<6 min training per model on M1/M2/M3 Macs)
- ✅ Production-ready modular code with comprehensive documentation
- ✅ Docker environment for reproducibility

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (recommended) or local Jupyter setup
- Data files in `data/` folder (included in repository)

### Installation

**Option 1: Docker** (recommended - everything pre-configured):

```bash
# Step 1: Build Docker image
./docker_build.sh

# Step 2: Start Jupyter
./docker_jupyter.sh

# Step 3: Open browser to http://localhost:8888
# Notebooks will be available in the file browser!
```

**Option 2: Local Jupyter**:

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Open GluonTS.API.ipynb or GluonTS.example.ipynb
```

See **[docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)** for complete Docker instructions.

---

## 📚 Where to Start?

### New to GluonTS? Start Here! 👇

1. **📖 Read**: `GluonTS.API.md` - Learn the basics
2. **🧪 Run**: `GluonTS.API.ipynb` - See models in action (5-10 min)
3. **📊 Explore**: `GluonTS.example.ipynb` - Complete application (15-20 min)

### Want the Complete Story?

Open **`GluonTS.example.ipynb`** and run "Restart & Run All"!

This shows you:
- Full data pipeline (loading, exploration, preprocessing)
- All 3 models trained and compared
- Comprehensive evaluation with beautiful visualizations
- Scenario analysis for public health decisions
- Actionable insights and recommendations

**Runtime**: ~10-15 minutes on CPU

---

## 📁 Project Structure

```
TutorTask121_GluonTS_COVID_19_Case_Prediction/
│
├── 📊 MAIN NOTEBOOKS (Start Here!)
│   ├── GluonTS.API.ipynb                   # Educational API demo (all 3 models)
│   ├── GluonTS.API.md                      # API documentation & guide
│   ├── GluonTS.example.ipynb               # Complete application (all 3 models)
│   └── GluonTS.example.md                  # Example documentation & guide
│
├── 🔧 UTILITY MODULES
│   ├── GluonTS_utils_data_io.py            # Data loading (JHU, Google, CDC)
│   ├── GluonTS_utils_preprocessing.py      # Preprocessing pipeline
│   ├── GluonTS_utils_gluonts.py            # GluonTS dataset creation
│   ├── GluonTS_utils_evaluation.py         # Metrics & plotting
│   ├── GluonTS_utils_notebook_loader.py    # One-line data loader
│   └── GluonTS_utils_models.py             # Model training wrappers
│
├── 📂 DATA
│   ├── data/
│   │   ├── cases.csv                       # JHU COVID-19 cases by state
│   │   ├── deaths.csv                      # JHU COVID-19 deaths by state
│   │   ├── mobility.csv                    # Google Mobility (6 metrics)
│   │   └── vaccine.csv                     # CDC vaccination data (future use)
│
├── 🐳 DOCKER SETUP
│   ├── Dockerfile                          # Docker configuration
│   ├── docker_build.sh                     # Build Docker image
│   ├── docker_jupyter.sh                   # Start Jupyter server
│   └── docker_bash.sh                      # Interactive shell access
│
├── 📚 DOCUMENTATION
│   ├── docs/
│   │   ├── START_HERE.md                   # Quick orientation (30 sec)
│   │   ├── DOCKER_GUIDE.md                 # Complete Docker instructions
│   │   ├── DATA_USAGE.md                   # What data is used and why
│   │   ├── DEATHS_DATA_INTEGRATION.md      # Deaths as a feature
│   │   └── FOLDER_STRUCTURE.md             # Detailed structure guide
│
├── 📦 ARCHIVE
│   ├── archive/                            # Old notebooks (superseded)
│   │   ├── GluonTS_DeepAR.API.ipynb
│   │   ├── GluonTS_DeepAR.example.ipynb
│   │   ├── GluonTS_SimpleFeedForward.API.ipynb
│   │   ├── GluonTS_SimpleFeedForward.example.ipynb
│   │   ├── GluonTS_DeepNPTS.API.ipynb
│   │   ├── GluonTS_DeepNPTS.example.ipynb
│   │   └── README.md                       # Why these are archived
│
├── 📝 PROJECT FILES
│   ├── requirements.txt                    # Python dependencies
│   ├── README.md                           # This file
│   └── .gitignore                          # Git ignore rules
```

---

## 📖 What Each Notebook Does

### API Notebook: `GluonTS.API.ipynb`

**Purpose**: Educational demonstration of model APIs

**Content**:
- Introduction to each model (DeepAR, SimpleFeedForward, DeepNPTS)
- Parameter explanations (what each parameter does)
- Basic configuration examples
- Quick training demonstrations with real COVID data
- Simple forecasting and evaluation

**Best for**: 
- Learning how to use GluonTS models
- Understanding parameter choices
- Quick API reference

**Runtime**: ~5-10 minutes

**Documentation**: See `GluonTS.API.md` for comprehensive guide

---

### Example Notebook: `GluonTS.example.ipynb`

**Purpose**: Complete, production-ready COVID-19 forecasting application

**Content**:
- Full data pipeline (load, explore, preprocess)
- Advanced feature engineering (deaths, mobility, CFR)
- Training all 3 models with detailed output
- Side-by-side model comparison
- Comprehensive evaluation (metrics + visualizations)
- Scenario analysis (intervention simulations)
- Conclusions and recommendations

**Best for**:
- Understanding complete forecasting applications
- Seeing all models compared
- Real-world problem solving
- Learning best practices

**Runtime**: ~10-15 minutes

**Documentation**: See `GluonTS.example.md` for comprehensive guide

---

## 🎯 Key Features Explained

### 1. Probabilistic Forecasting

Unlike simple point predictions, our models provide **uncertainty estimates**:

```
Not just: "We predict 50,000 cases"
But: "We predict 50,000 cases, with 80% confidence between 40,000-60,000"
```

This is critical for:
- **Risk assessment**: Know when uncertainty is high
- **Resource planning**: Plan for the range, not just the average
- **Decision confidence**: Understand when predictions are reliable

---

### 2. Multi-Model Comparison

We train **three different models** because:
- No single model is always best
- Different models excel in different situations
- Comparing models reveals insights about the data

**Model Characteristics**:

| Model | External Features | Training Time | Best For |
|-------|------------------|---------------|----------|
| **DeepAR** | ✅ Yes | ~3-4 min | Highest accuracy needs |
| **SimpleFeedForward** | ❌ No | ~30 sec | Fast baselines |
| **DeepNPTS** | ✅ Yes | ~3-4 min | Regime changes |

---

### 3. External Features (Covariates)

Our advanced models use **external features** to improve predictions:

- **Daily Deaths**: Leading indicator of case severity
- **Mobility Patterns**: 6 metrics showing behavioral changes
- **CFR**: Case Fatality Ratio (deaths/cases)
- **Moving Averages**: Smoothed trends

Why this matters:
- More information → better predictions
- Can model causality (lockdowns → reduced mobility → fewer cases)
- Enables scenario analysis

---

### 4. Scenario Analysis

Answer "what if?" questions:
- "What if we implement a lockdown?"
- "What if vaccination rates increase?"
- "What if a new variant emerges?"

**How it works**:
1. Train model on historical data
2. Create scenarios with different future conditions
3. Generate forecasts for each scenario
4. Compare to guide decisions

See Section 6 in `GluonTS.example.ipynb` for complete demonstration!

---

## 🔧 Usage Examples

### Quick Start: Load COVID Data

```python
from GluonTS_utils_notebook_loader import load_covid_data_for_gluonts

# One function to load and prepare everything!
data = load_covid_data_for_gluonts(
    data_dir="data",
    prediction_length=14,
    context_length=60,
    verbose=True
)

# Ready-to-use datasets
train_ds = data['train_ds']
test_ds = data['test_ds']
num_features = data['num_features']
```

---

### Train DeepAR Model

```python
from GluonTS_utils_models import train_deepar_covid

results = train_deepar_covid(
    train_ds=train_ds,
    test_ds=test_ds,
    prediction_length=14,
    num_feat_dynamic_real=num_features,
    epochs=10,
    learning_rate=0.001,
    context_length=60,
    num_layers=2,
    hidden_size=40,
    dropout=0.1,
    verbose=True
)

print(f"MAE: {results.metrics['mae']:.2f}")
print(f"Training time: {results.training_time:.1f}s")
```

---

### Compare All Models

```python
from GluonTS_utils_models import compare_models, print_model_comparison

# Train all models
deepar_results = train_deepar_covid(...)
ff_results = train_feedforward_covid(...)
npts_results = train_deepnpts_covid(...)

# Compare
comparison = compare_models([deepar_results, ff_results, npts_results])
print_model_comparison(comparison)
```

---

## 📊 Expected Results

### Model Performance (on test set)

| Model | MAE | RMSE | MAPE | Training Time |
|-------|-----|------|------|---------------|
| DeepAR | ~X,XXX | ~X,XXX | ~XX% | ~3-4 min |
| SimpleFeedForward | ~X,XXX | ~X,XXX | ~XX% | ~30-60 sec |
| DeepNPTS | ~X,XXX | ~X,XXX | ~XX% | ~3-4 min |

*Actual results vary based on train/test split and data quality*

### What You'll See

1. **Beautiful Visualizations**:
   - Time series plots (cases, deaths, mobility)
   - Forecast plots with confidence intervals
   - Model comparison charts
   - Scenario analysis comparisons

2. **Comprehensive Metrics**:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)
   - Training time comparisons

3. **Actionable Insights**:
   - Which model performs best
   - How external features improve forecasts
   - Impact of intervention scenarios
   - Recommendations for public health policy

---

## 🎓 Learning Path

### Beginner Path (New to Forecasting)

1. **Read**: `GluonTS.API.md` - Understand the basics
2. **Run**: `GluonTS.API.ipynb` - See models in action
3. **Focus on**: 
   - What each model does
   - How to configure parameters
   - What metrics mean

**Time**: 30-45 minutes

---

### Intermediate Path (Know Forecasting Basics)

1. **Run**: `GluonTS.example.ipynb` - Complete application
2. **Study**:
   - Feature engineering strategies
   - Model comparison techniques
   - Evaluation best practices
3. **Experiment**: Change parameters, try different features

**Time**: 1-2 hours

---

### Advanced Path (Practitioners)

1. **Study**: All utility code (`GluonTS_utils_*.py`)
2. **Adapt**: Replace COVID data with your own time series
3. **Extend**: 
   - Add new features
   - Try ensemble methods
   - Implement hierarchical forecasting
4. **Deploy**: Production considerations

**Time**: 2-4 hours

---

## 🔬 Technical Details

### Models

**DeepAR** (Salinas et al., 2020)
- Architecture: Autoregressive RNN with external features
- Distribution: Student's t-distribution for uncertainty
- Training: PyTorch Lightning backend
- Best for: Complex temporal patterns, rich feature sets

**SimpleFeedForward**
- Architecture: Feed-forward MLP
- Distribution: Gaussian or Student's t
- Training: Fast CPU training
- Best for: Baselines, quick experiments, stable trends
- Limitation: No external features

**DeepNPTS** (Rangapuram et al., 2021)
- Architecture: Non-parametric deep learning
- Distribution: Kernel density estimation
- Training: Lightweight, flexible
- Best for: Regime changes, distribution shifts

---

### Feature Engineering

Engineered features (all computed in preprocessing):

1. **Daily_Cases_MA7**: 7-day moving average of cases (target)
2. **Daily_Deaths_MA7**: 7-day moving average of deaths
3. **Cumulative_Deaths**: Total deaths up to each date
4. **CFR**: Case Fatality Ratio (deaths/cases)
5. **Mobility Metrics** (6 features):
   - Retail & Recreation
   - Grocery & Pharmacy
   - Parks
   - Transit Stations
   - Workplaces
   - Residential

All features are standardized and aligned temporally.

---

### Training Configuration

**Hardware**: CPU-optimized (M1/M2/M3 Mac compatible)

**Default Parameters**:
```python
# DeepAR
epochs = 10
context_length = 60  # 2 months history
prediction_length = 14  # 2 weeks forecast
hidden_size = 40
num_layers = 2
dropout = 0.1

# SimpleFeedForward
epochs = 20  # Fast training, can use more
context_length = 60
prediction_length = 14
hidden_dimensions = [40, 40]

# DeepNPTS
epochs = 15
context_length = 60
prediction_length = 14
num_hidden_nodes = [40]
dropout_rate = 0.1
```

---

## 📝 Submission Structure

This project follows **MSML610 Fall 2025 submission guidelines** with a streamlined approach:

### Consolidated Structure

Instead of 6 separate notebooks (3 API + 3 example), we provide:
- ✅ **2 main notebooks**: `GluonTS.API.ipynb` and `GluonTS.example.ipynb`
- ✅ **Both cover all 3 models** side-by-side for easy comparison
- ✅ **Comprehensive documentation**: Detailed `.md` files for each notebook

### Benefits

- ✅ Easier navigation (2 notebooks vs. 6)
- ✅ Direct model comparison
- ✅ Aligned with submission template structure
- ✅ More maintainable and consistent

### Archive

Old individual notebooks are in `archive/` folder for reference.

---

## 🐛 Troubleshooting

### Docker Issues

**Problem**: `docker_build.sh` fails
```bash
# Solution: Ensure Docker is running
docker info

# Rebuild from scratch
docker_build.sh --no-cache
```

**Problem**: Port 8888 already in use
```bash
# Solution: Change port in docker_jupyter.sh
# Edit: -p 8889:8888 (use 8889 instead)
```

---

### Training Issues

**Problem**: Models training very slowly
```bash
# Solution 1: Reduce epochs
epochs = 5  # instead of 10

# Solution 2: Reduce network size
hidden_size = 20  # instead of 40

# Solution 3: Use SimpleFeedForward for quick tests
```

**Problem**: Out of memory errors
```bash
# Solution: Reduce batch size
batch_size = 16  # instead of 32
```

---

### Data Issues

**Problem**: Data files not found
```bash
# Verify data/ folder exists
ls data/

# Should see: cases.csv, deaths.csv, mobility.csv, vaccine.csv
```

**Problem**: Feature count mismatch
```python
# Check num_feat_dynamic_real matches actual features
print(f"Features: {data['num_features']}")
print(f"Feature names: {data['features']}")
```

---

## 📚 Additional Resources

### GluonTS Documentation
- Official Docs: https://ts.gluon.ai/
- Tutorials: https://ts.gluon.ai/stable/tutorials/
- API Reference: https://ts.gluon.ai/stable/api/
- GitHub: https://github.com/awslabs/gluonts

### Time Series Forecasting
- Book: "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)
- Course: fast.ai Time Series
- Papers:
  - DeepAR: https://arxiv.org/abs/1704.04110
  - DeepNPTS: https://arxiv.org/abs/1906.05264

### COVID-19 Data
- JHU CSSE: https://github.com/CSSEGISandData/COVID-19
- Google Mobility: https://www.google.com/covid19/mobility/
- CDC Data: https://covid.cdc.gov/covid-data-tracker/

---

## 🤝 Contributing

This is a class project for **MSML610 Fall 2025**. 

For questions or issues:
1. Check the documentation (`GluonTS.API.md`, `GluonTS.example.md`)
2. Review troubleshooting section above
3. Consult utility code (well-commented!)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎓 Course Information

**Course**: MSML610 Fall 2025  
**Topic**: Probabilistic Time Series Forecasting  
**Framework**: GluonTS with PyTorch backend  
**Hardware**: CPU-optimized for M1/M2/M3 Macs

---

## 🎉 Acknowledgments

**Data Sources**:
- Johns Hopkins University CSSE COVID-19 Data
- Google COVID-19 Community Mobility Reports
- CDC COVID Data Tracker

**Frameworks**:
- GluonTS (Amazon Web Services)
- PyTorch and PyTorch Lightning
- Pandas, NumPy, Matplotlib, Seaborn

---

**Last Updated**: December 2025  
**Project Status**: ✅ Complete and ready for submission!

---

## 📞 Quick Reference

**Main Notebooks**:
- `GluonTS.API.ipynb` - API demonstrations
- `GluonTS.example.ipynb` - Complete application

**Documentation**:
- `GluonTS.API.md` - API guide
- `GluonTS.example.md` - Example guide
- `docs/START_HERE.md` - Quick start

**Docker**:
```bash
./docker_build.sh      # Build
./docker_jupyter.sh    # Run Jupyter
./docker_bash.sh       # Shell access
```

**Data**:
- `data/cases.csv` - COVID-19 cases
- `data/deaths.csv` - COVID-19 deaths
- `data/mobility.csv` - Google Mobility
- `data/vaccine.csv` - CDC vaccines (future use)

**Models**:
- DeepAR: Advanced RNN (~3-4 min)
- SimpleFeedForward: Fast baseline (~30 sec)
- DeepNPTS: Flexible non-parametric (~3-4 min)

---

Ready to forecast COVID-19 cases? Start with `GluonTS.API.ipynb`! 🚀📈
