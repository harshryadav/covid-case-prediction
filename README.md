# COVID-19 Case Prediction with GluonTS

> Probabilistic time series forecasting of COVID-19 cases using deep learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GluonTS](https://img.shields.io/badge/GluonTS-0.14.0+-orange.svg)](https://ts.gluon.ai/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 📋 Overview

This project implements **probabilistic time series forecasting** for COVID-19 cases in the United States using **GluonTS**, a Python toolkit for deep learning-based forecasting. Unlike traditional point predictions, this approach provides:

- 🎯 **Probabilistic Forecasts** with confidence intervals
- 📊 **Uncertainty Quantification** for informed decision-making
- 🔮 **Scenario Analysis** for policy intervention planning
- 🤖 **Advanced Models**: DeepAR, Transformer, Baseline comparisons
- 📈 **Rigorous Evaluation**: CRPS, MAE, RMSE metrics

### Key Features

✅ **Data Pipeline**: Automated loading, preprocessing, and GluonTS formatting  
✅ **Multiple Models**: Naive baselines + DeepAR probabilistic forecasting  
✅ **Visualization**: EDA plots, forecast visualizations with uncertainty bands  
✅ **Evaluation**: Comprehensive metrics (CRPS, MAE, RMSE, MAPE)  
✅ **Docker Support**: Reproducible containerized environment  
✅ **Mobility Data**: Optional Google Mobility data integration  

---

## 🚀 Quick Start

### Option 1: Local Setup (5 minutes)

```bash
# 1. Install dependencies
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run full pipeline
python run_pipeline.py

# 3. View results
open results/baseline_forecasts.png
```

### Option 2: Docker Setup (Recommended)

```bash
# 1. Build and start
docker-compose build
docker-compose up -d

# 2. Run pipeline
docker-compose exec covid-forecasting python run_pipeline.py

# 3. Stop when done
docker-compose down
```

**Results will be in the `results/` folder on your host machine!**

---

## 📁 Project Structure

```
covid-case-prediction/
│
├── 📊 data/                          # Datasets
│   ├── time_series_covid19_confirmed_US.csv
│   ├── time_series_covid19_deaths_US.csv
│   ├── time_series_covid19_vaccine_us.csv
│   └── mobility_report_US.csv
│
├── 💻 src/                           # Source code
│   ├── data_processing/              # Data pipeline
│   ├── models/                       # Model training
│   ├── evaluation/                   # Metrics
│   ├── visualization/                # Plotting
│   └── utils/                        # Helpers
│
├── 🎯 run_pipeline.py                # Main entry point
├── 📄 requirements.txt               # Dependencies
│
├── 🐳 Dockerfile                     # Container setup
├── 🐳 docker-compose.yml             # Orchestration
│
└── 📚 Documentation/
    ├── README.md                     # This file
    ├── QUICKSTART.md                 # 30-minute tutorial
    ├── DATASET_GUIDE.md              # Data documentation
    ├── MODEL_GUIDE.md                # Model architecture & training
    ├── CODE_REFERENCE.md             # Code organization
    ├── DOCKER.md                     # Docker usage
    ├── REFERENCE.md                  # Quick commands
    └── PROJECT_PLAN.md               # Detailed 7-week roadmap
```

---

## 📖 Documentation

### For New Users
- **[QUICKSTART.md](QUICKSTART.md)** - 30-minute hands-on tutorial
- **[README.md](README.md)** - This file (overview)

### For Development
- **[CODE_REFERENCE.md](CODE_REFERENCE.md)** - Code structure & modules
- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** - GluonTS models & training
- **[DATASET_GUIDE.md](DATASET_GUIDE.md)** - Dataset documentation

### For Deployment
- **[DOCKER.md](DOCKER.md)** - Docker setup & usage
- **[REFERENCE.md](REFERENCE.md)** - Quick command reference

### For Planning
- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Detailed 7-week implementation roadmap

---

## 🎯 What This Project Does

### 1. Data Processing
```python
# Loads and preprocesses COVID-19 data
from src.data_processing import DataLoader, preprocess_national_data

loader = DataLoader()
cases_df = loader.load_cases()
deaths_df = loader.load_deaths()

# Aggregates to national level, merges with mobility data
national_df = preprocess_national_data(cases_df, deaths_df)
```

### 2. Model Training
```python
# Trains baseline and DeepAR models
python src/models/train_baseline.py      # Naive, Seasonal Naive
python src/models/train_deepar.py        # Probabilistic DeepAR
```

### 3. Evaluation & Visualization
```python
# Evaluates forecasts and creates visualizations
from src.evaluation.metrics import calculate_metrics
from src.visualization.plot_utils import plot_forecast

metrics = calculate_metrics(actual, forecast)
plot_forecast(forecast, actual, save_path='results/forecast.png')
```

---

## 🤖 Models Implemented

| Model | Type | Use Case | Status |
|-------|------|----------|--------|
| **Naive** | Baseline | Simple benchmark | ✅ |
| **Seasonal Naive** | Baseline | Weekly pattern | ✅ |
| **DeepAR** | Probabilistic | Advanced forecasting | ✅ |
| **Transformer** | Deep Learning | Long sequences | 🔄 Future |
| **Gaussian Process** | Probabilistic | Uncertainty | 🔄 Future |

---

## 📊 Datasets

### Required (Included)
- **JHU CSSE US Confirmed Cases** - Daily cumulative cases by county
- **JHU CSSE US Deaths** - Daily cumulative deaths by county
- **JHU CSSE US Vaccines** - Vaccination data by state

### Optional (Bonus)
- **Google Mobility Data** - Movement trends for enhanced predictions

**All datasets are already in the `data/` folder!**

See **[DATASET_GUIDE.md](DATASET_GUIDE.md)** for detailed information.

---

## 📈 Example Output

The pipeline generates:

```
results/
├── eda_visualization.png          # 4-panel EDA plot
├── baseline_forecasts.png         # Naive model forecasts
├── deepar_forecast.png            # DeepAR predictions
└── metrics.json                   # Evaluation metrics
```

**Sample Forecast:**
- 14-day ahead predictions
- 10th, 50th, 90th percentile confidence intervals
- Uncertainty bands visualized
- CRPS metric for probabilistic evaluation

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Forecasting** | GluonTS 0.14+ |
| **Deep Learning** | Apache MXNet |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Containerization** | Docker, Docker Compose |

---

## 📝 Usage Examples

### Run Full Pipeline
```bash
python run_pipeline.py
```

### Individual Steps
```bash
# 1. Preprocess data
python src/data_processing/preprocess.py

# 2. Create EDA visualizations
python src/visualization/create_eda_plots.py

# 3. Train models
python src/models/train_baseline.py
python src/models/train_deepar.py
```

### With Docker
```bash
docker-compose up -d
docker-compose exec covid-forecasting bash
# Inside container:
python run_pipeline.py
```

---

## 🔬 Project Goals

**Primary Objectives:**
1. ✅ Build probabilistic forecasting models (DeepAR, Transformer)
2. ✅ Quantify prediction uncertainty with confidence intervals
3. ✅ Evaluate models using CRPS and other metrics
4. ✅ Visualize forecasts with uncertainty bands

**Bonus Objectives:**
- ✅ Incorporate Google Mobility data as covariates
- 🔄 Implement state-level forecasting (multi-region)
- 🔄 Create ensemble models
- 🔄 Build interactive dashboard

---

## 📚 Learning Outcomes

By working through this project, you'll learn:

- ✅ **Time Series Forecasting**: GluonTS framework, probabilistic models
- ✅ **Deep Learning**: RNN-based architectures (DeepAR)
- ✅ **Uncertainty Quantification**: Confidence intervals, CRPS metric
- ✅ **Data Engineering**: ETL pipelines, data preprocessing
- ✅ **Software Engineering**: Modular design, Docker containerization
- ✅ **Evaluation**: Proper backtesting, multiple metrics

---

## 🚧 Development Status

| Phase | Status | Completion |
|-------|--------|------------|
| **Data Pipeline** | ✅ Complete | 100% |
| **EDA & Visualization** | ✅ Complete | 100% |
| **Baseline Models** | ✅ Complete | 100% |
| **DeepAR Model** | ✅ Complete | 100% |
| **Docker Setup** | ✅ Complete | 100% |
| **Documentation** | ✅ Complete | 100% |
| **Advanced Models** | 🔄 Future | 0% |
| **Dashboard** | 🔄 Future | 0% |

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with different models
- Add new datasets
- Improve visualizations
- Optimize hyperparameters

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Data**: Johns Hopkins University CSSE COVID-19 Data Repository
- **Framework**: Amazon GluonTS Team
- **Mobility Data**: Google COVID-19 Community Mobility Reports (via ActiveConclusion)

---

## 📞 Quick Links

| Resource | Link |
|----------|------|
| **Quick Start** | [QUICKSTART.md](QUICKSTART.md) |
| **Models** | [MODEL_GUIDE.md](MODEL_GUIDE.md) |
| **Data** | [DATASET_GUIDE.md](DATASET_GUIDE.md) |
| **Code** | [CODE_REFERENCE.md](CODE_REFERENCE.md) |
| **Docker** | [DOCKER.md](DOCKER.md) |
| **Commands** | [REFERENCE.md](REFERENCE.md) |
| **GluonTS Docs** | [ts.gluon.ai](https://ts.gluon.ai/) |

---

**Ready to forecast? Start with [QUICKSTART.md](QUICKSTART.md)!** 🚀
