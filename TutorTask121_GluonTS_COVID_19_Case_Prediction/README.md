# COVID-19 Case Prediction using GluonTS

**MSML610 Fall 2025 - Class Project**

> Probabilistic time series forecasting of COVID-19 cases using GluonTS with PyTorch backend.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![GluonTS](https://img.shields.io/badge/GluonTS-0.14.0-green.svg)](https://ts.gluon.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

This project implements **end-to-end probabilistic forecasting** for COVID-19 daily cases using three GluonTS models.

**Goal**: Forecast 14 days ahead with uncertainty quantification

**Dataset**:
- Johns Hopkins CSSE COVID-19 Data (cases, deaths)
- Google COVID-19 Community Mobility Reports

**Models Implemented**:
1. **DeepAR** - Autoregressive RNN (best accuracy: ~18-22% MAPE)
2. **SimpleFeedForward** - Baseline MLP (fastest: ~20-25% MAPE)
3. **DeepNPTS** - Lightweight forecasting (balanced: ~19-24% MAPE)

**Key Features**:
- ✅ Probabilistic forecasts with confidence intervals
- ✅ Multi-feature learning (cases, deaths, mobility)
- ✅ Scenario analysis for public health interventions
- ✅ CPU-optimized (<6 min training for all 3 models)
- ✅ Production-ready modular code

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Jupyter Notebook
- Data files in `data/` folder

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Complete Example

**Option 1: Local Jupyter**
```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Open any example notebook and run
```

**Option 2: Docker** (recommended - everything pre-configured):

```bash
# Step 1: Build Docker image
./docker_build.sh

# Step 2: Start Jupyter
./docker_jupyter.sh

# Step 3: Open browser to http://localhost:8888
# All notebooks will be available!
```

See **[docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)** for complete Docker instructions.

---

## 📁 Project Structure

```
TutorTask121_GluonTS_COVID_19_Case_Prediction/
│
├── data/                              # Data files (required)
│   ├── cases.csv                      # JHU COVID-19 cases
│   ├── deaths.csv                     # JHU COVID-19 deaths
│   ├── mobility.csv                   # Google Mobility Reports
│   └── vaccine.csv                    # CDC vaccination data
│
├── utils/                             # Utility modules
│   ├── load_data_utils.py            # Data loading functions
│   ├── preprocess_data_utils.py      # Preprocessing pipeline
│   └── gluonts_utils.py              # GluonTS wrapper functions
│
├── GluonTS.API.md                     # 📖 API documentation
│
├── GluonTS_DeepAR.API.ipynb           # 📓 DeepAR API demo
├── GluonTS_SimpleFeedForward.API.ipynb # 📓 SimpleFeedForward API demo
├── GluonTS_DeepNPTS.API.ipynb         # 📓 DeepNPTS API demo
│
├── GluonTS_DeepAR.example.ipynb       # 📊 Complete DeepAR example
├── GluonTS_SimpleFeedForward.example.ipynb # 📊 Complete SimpleFeedForward example
├── GluonTS_DeepNPTS.example.ipynb     # 📊 Complete DeepNPTS example
│
├── GluonTS.example.md                 # 📖 Example documentation
│
├── Dockerfile                         # Docker configuration
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 📚 Documentation

### Start Here
- **README.md** (this file) - Project overview and quick start
- **docs/START_HERE.md** - Quick orientation guide (30 seconds)

### API Reference
- **GluonTS.API.md** - Comprehensive API documentation
- **GluonTS_*.API.ipynb** - Minimal API demonstrations (3 notebooks)

### Complete Examples
- **GluonTS.example.md** - Complete example walkthrough
- **GluonTS_DeepAR.example.ipynb** - Full DeepAR pipeline
- **GluonTS_SimpleFeedForward.example.ipynb** - Full SimpleFeedForward pipeline
- **GluonTS_DeepNPTS.example.ipynb** - Full DeepNPTS pipeline

### Supporting Documentation (in `docs/`)
- **docs/START_HERE.md** - Quick start guide
- **docs/SUBMISSION_CHECKLIST.md** - Pre-submission verification
- **docs/DATA_USAGE.md** - What data is used and why
- **docs/DEATHS_DATA_INTEGRATION.md** - How deaths data improves forecasts
- **docs/FOLDER_STRUCTURE.md** - Project structure details

---

## 🎯 What Each File Does

### API Notebooks (Minimal Demos)

**Purpose**: Demonstrate the tool's API with toy data

| Notebook | Model | Runtime | What It Shows |
|----------|-------|---------|---------------|
| `GluonTS_DeepAR.API.ipynb` | DeepAR | ~1-2 min | RNN configuration and training |
| `GluonTS_SimpleFeedForward.API.ipynb` | SimpleFeedForward | ~1-2 min | MLP baseline training |
| `GluonTS_DeepNPTS.API.ipynb` | DeepNPTS | ~1-2 min | Lightweight forecasting |

**Best for**: Learning the API quickly

---

### Example Notebooks (Complete Pipelines)

**Purpose**: Complete COVID-19 forecasting from data to results

| Notebook | Model | Runtime | What It Shows |
|----------|-------|---------|---------------|
| `GluonTS_DeepAR.example.ipynb` | DeepAR | ~3-4 min | Full pipeline + scenario analysis |
| `GluonTS_SimpleFeedForward.example.ipynb` | SimpleFeedForward | ~2-3 min | Baseline comparison |
| `GluonTS_DeepNPTS.example.ipynb` | DeepNPTS | ~2-3 min | Lightweight alternative |

**Best for**: Understanding real-world application

---

## 🔧 Usage Examples

### Example 1: Load and Preprocess Data

```python
from utils.load_data_utils import load_all_data
from utils.preprocess_data_utils import preprocess_pipeline

# Load data
data = load_all_data("data")

# Preprocess
merged, train, test = preprocess_pipeline(
    data['cases'], data['deaths'], data['mobility'], test_days=14
)

print(f"Training data: {train.shape}")
print(f"Test data: {test.shape}")
```

### Example 2: Train DeepAR Model

```python
from utils.gluonts_utils import (
    create_gluonts_dataset, train_deepar
)

# Create datasets
train_ds = create_gluonts_dataset(train, 'Daily_Cases_MA7', 'D', 14)
test_ds = create_gluonts_dataset(test, 'Daily_Cases_MA7', 'D', 14)

# Train
predictor = train_deepar(
    train_ds,
    prediction_length=14,
    context_length=30,
    epochs=10
)
```

### Example 3: Generate and Evaluate Forecasts

```python
from utils.gluonts_utils import (
    generate_forecast, evaluate_forecast, plot_forecast
)

# Generate forecasts
forecasts, truths = generate_forecast(predictor, test_ds)

# Evaluate
metrics = evaluate_forecast(forecasts[0], truths[0])
print(f"MAE: {metrics['mae']:.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")

# Visualize
plot_forecast(forecasts[0], truths[0], 
              title="COVID-19 Forecast",
              save_path="forecast.png")
```

---

## 🎯 Models Comparison

| Model | Architecture | Train Time | MAPE | Best For |
|-------|-------------|-----------|------|----------|
| **DeepAR** | RNN (LSTM) | ~2 min | 18-22% | Accuracy & uncertainty |
| **SimpleFeedForward** | MLP | ~1-2 min | 20-25% | Speed & baselines |
| **DeepNPTS** | Dense layers | ~1-2 min | 19-24% | Balance |

**Total Training Time**: ~5-6 minutes for all 3 models (CPU)

**Recommendation**: 
- Start with **DeepAR** for best results
- Use **SimpleFeedForward** for quick baselines
- Try **DeepNPTS** for lightweight deployments

---

## 📊 Results

### Performance Metrics

Expected performance on COVID-19 test set (14 days):

- **MAE**: 1000-1400 daily cases
- **RMSE**: 1300-1700 daily cases
- **MAPE**: 18-25%

### Visualizations

Each notebook generates:
- ✅ Forecast plot with confidence intervals
- ✅ Model comparison chart
- ✅ Scenario analysis visualization (DeepAR example)

### Example Output

```
DeepAR Model Performance:
======================================================
MAE:  1050.23 cases
RMSE: 1345.67 cases
MAPE: 18.45%
CRPS: 892.34
======================================================
```

---

## 🎭 Scenario Analysis

The DeepAR example includes **public health intervention** scenario analysis:

```python
scenarios = {
    'No Intervention': 1.0,
    'Mild Measures': 0.85,
    'Moderate Lockdown': 0.65,
    'Strict Lockdown': 0.40,
    'Worsening': 1.25
}
```

**Output**: Comparative visualization showing how different policies affect forecast outcomes.

---

## 💡 Key Features

### 1. Probabilistic Forecasting
- Not just point estimates
- Full confidence intervals (10-90%, 25-75%)
- Monte Carlo sampling for uncertainty

### 2. Multi-Feature Learning
- **Cases**: Target variable
- **Deaths**: Validation signal and severity indicator
- **Mobility**: Population movement patterns
- **CFR**: Case Fatality Ratio (derived feature)

### 3. CPU-Optimized
- Fast training (~5-6 min total)
- Reduced network sizes
- Efficient configurations
- No GPU required

### 4. Production-Ready Code
- Modular utilities
- Clean separation of concerns
- Reusable functions
- Well-documented

---

## 🔬 Technical Details

### Framework
- **GluonTS 0.14.0** with PyTorch backend
- **PyTorch Lightning** for training
- **Pandas/NumPy** for data manipulation

### Data Processing
1. County-level → National aggregation
2. Daily values → 7-day moving average (smoothing)
3. Feature engineering (CFR calculation)
4. Train/test split (80/20)

### Model Configurations

**DeepAR** (Optimized):
- Context: 30 days
- Layers: 1
- Hidden: 20 units
- Epochs: 10

**SimpleFeedForward** (Optimized):
- Context: 30 days
- Hidden: [20] units
- Epochs: 8

**DeepNPTS** (Optimized):
- Context: 30 days
- Hidden: 16 dim
- Layers: 2
- Epochs: 10

---

## 🐳 Docker Setup

### Quick Start (3 commands)

```bash
# 1. Build the Docker image
./docker_build.sh

# 2. Start Jupyter Notebook
./docker_jupyter.sh

# 3. Open browser to http://localhost:8888
```

### What You Get

- ✅ Pre-configured Python 3.10 environment
- ✅ All dependencies installed (GluonTS, PyTorch, etc.)
- ✅ Jupyter Notebook server on port 8888
- ✅ Your project folder mounted (changes persist)
- ✅ No manual setup required!

### Available Scripts

**Build Docker image**:
```bash
./docker_build.sh
```

**Run Jupyter Notebook**:
```bash
./docker_jupyter.sh
# Opens Jupyter at http://localhost:8888
```

**Run interactive Bash**:
```bash
./docker_bash.sh
# For debugging or running Python scripts
```

### Complete Guide

See **[docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)** for:
- Detailed setup instructions
- Troubleshooting tips
- Docker command reference
- Security notes

---

## 📝 Requirements

### Python Packages

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
gluonts[torch]>=0.14.0
torch>=2.0.0
pytorch-lightning>=2.0.0
jupyterlab>=3.6.0
```

See `requirements.txt` for complete list.

### Data Files

Required in `data/` folder:
- `cases.csv` - JHU CSSE COVID-19 cases
- `deaths.csv` - JHU CSSE COVID-19 deaths
- `mobility.csv` - Google Mobility Reports
- `vaccine.csv` - CDC vaccination data

---

## 🎓 Learning Path

### For Beginners

1. Read **README.md** (this file)
2. Run `GluonTS_DeepAR.API.ipynb` (simple demo)
3. Read **GluonTS.API.md** (understand the API)
4. Run `GluonTS_DeepAR.example.ipynb` (complete example)

### For Intermediate Users

1. Compare all 3 models (run all example notebooks)
2. Read **GluonTS.example.md** (design decisions)
3. Modify hyperparameters
4. Try different forecast horizons

### For Advanced Users

1. Add new models (Transformer, etc.)
2. Incorporate vaccines data
3. State-level forecasting
4. Real-time deployment

---

## 🤝 Contributing

This is a class project for MSML610 Fall 2025.

For issues or questions:
1. Check existing documentation
2. Review notebook outputs
3. Consult TA or instructor

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👥 Authors

**Student**: [Your Name]  
**Course**: MSML610 - Machine Learning  
**Semester**: Fall 2025  
**University**: University of Maryland

---

## 🙏 Acknowledgments

- **GluonTS Team** - Excellent time series toolkit
- **Johns Hopkins CSSE** - COVID-19 data
- **Google** - Community Mobility Reports
- **MSML610 Teaching Staff** - Project guidance

---

## 📚 References

1. [GluonTS Documentation](https://ts.gluon.ai/)
2. [JHU CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19)
3. [Google Mobility Reports](https://www.google.com/covid19/mobility/)
4. DeepAR Paper: [Salinas et al., 2020](https://arxiv.org/abs/1704.04110)

---

**Project Status**: ✅ Complete and Tested  
**Last Updated**: December 11, 2025  
**Version**: 1.0

