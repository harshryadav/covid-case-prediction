# Code Reference & Structure

Complete reference for the source code organization, modules, and usage.

---

## 📁 Project Structure

```
covid-case-prediction/
│
├── 📊 data/                          # Raw datasets (gitignored processed files)
│   ├── time_series_covid19_confirmed_US.csv
│   ├── time_series_covid19_deaths_US.csv
│   ├── time_series_covid19_vaccine_us.csv
│   └── mobility_report_US.csv
│
├── 💻 src/                           # Source code modules
│   ├── data_processing/              # ETL pipeline
│   │   ├── load_data.py             # DataLoader class
│   │   ├── preprocess.py            # Preprocessing functions
│   │   └── prepare_gluonts.py       # GluonTS formatting
│   │
│   ├── models/                       # Model training
│   │   ├── train_baseline.py        # Naive, Seasonal Naive
│   │   └── train_deepar.py          # DeepAR model
│   │
│   ├── evaluation/                   # Metrics & evaluation
│   │   └── metrics.py               # MAE, RMSE, MAPE, CRPS
│   │
│   ├── visualization/                # Plotting utilities
│   │   ├── plot_utils.py            # Reusable plot functions
│   │   └── create_eda_plots.py      # 4-panel EDA visualization
│   │
│   └── utils/                        # Helper functions
│       └── config.py                # Central configuration
│
├── 🎯 run_pipeline.py                # Main pipeline entry point
│
├── 🐳 Docker files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── 📚 Documentation
│   ├── README.md                     # Project overview
│   ├── QUICKSTART.md                 # Quick tutorial
│   ├── DATASET_GUIDE.md              # Data documentation
│   ├── MODEL_GUIDE.md                # Model architecture
│   ├── CODE_REFERENCE.md             # This file
│   ├── DOCKER.md                     # Docker usage
│   ├── REFERENCE.md                  # Quick commands
│   └── PROJECT_PLAN.md               # Detailed roadmap
│
├── 📄 Config files
│   ├── requirements.txt              # Python dependencies
│   └── .gitignore
│
└── 📂 Generated (not in git)
    ├── data/processed/               # Preprocessed data
    ├── data/gluonts/                 # GluonTS format data
    ├── results/                      # Output plots & metrics
    ├── models/                       # Saved model checkpoints
    └── venv/                         # Virtual environment
```

---

## 📦 Module Reference

### 1. `src/data_processing/` - Data Pipeline

#### `load_data.py` - DataLoader Class

**Purpose**: Load all COVID-19 datasets

**Classes**:
```python
class DataLoader:
    """Centralized data loading"""
    
    def load_cases(self) -> pd.DataFrame:
        """Load US confirmed cases from JHU CSSE"""
        
    def load_deaths(self) -> pd.DataFrame:
        """Load US deaths from JHU CSSE"""
        
    def load_vaccines(self) -> pd.DataFrame:
        """Load US vaccine data from JHU CSSE"""
        
    def load_mobility(self) -> pd.DataFrame:
        """Load Google Mobility data"""
```

**Usage**:
```python
from src.data_processing.load_data import DataLoader

loader = DataLoader()
cases_df = loader.load_cases()
deaths_df = loader.load_deaths()
mobility_df = loader.load_mobility()
```

**Output**: Raw pandas DataFrames

---

#### `preprocess.py` - Data Preprocessing

**Purpose**: Clean, aggregate, and merge datasets

**Key Functions**:

```python
def preprocess_jhu_data(
    df: pd.DataFrame,
    value_col: str,
    start_col_index: int = 11
) -> pd.DataFrame:
    """
    Convert JHU wide format to long format
    
    Args:
        df: JHU dataframe
        value_col: Name for value column (e.g., 'cases', 'deaths')
        start_col_index: Where date columns start
        
    Returns:
        Long format DataFrame with Date, state, value columns
    """
```

```python
def aggregate_to_national(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate county-level to national level
    
    Returns:
        DataFrame with columns: Date, daily_cases, daily_deaths
    """
```

```python
def preprocess_national_data(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    mobility_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline
    
    Steps:
    1. Convert to long format
    2. Aggregate to national level
    3. Calculate daily new cases/deaths
    4. Merge with mobility data (if provided)
    5. Handle missing values
    
    Returns:
        National-level DataFrame ready for modeling
    """
```

**Usage**:
```python
from src.data_processing.preprocess import preprocess_national_data
from src.data_processing.load_data import DataLoader

loader = DataLoader()
cases_df = loader.load_cases()
deaths_df = loader.load_deaths()
mobility_df = loader.load_mobility()

# Full preprocessing
national_df = preprocess_national_data(cases_df, deaths_df, mobility_df)

# Saves to: data/processed/national_data.csv
```

**Output**: `data/processed/national_data.csv`

---

#### `prepare_gluonts.py` - GluonTS Data Preparation

**Purpose**: Convert preprocessed data to GluonTS ListDataset format

**Key Functions**:

```python
def create_gluonts_dataset(
    df: pd.DataFrame,
    target_col: str = 'daily_cases',
    freq: str = 'D',
    prediction_length: int = 14,
    train_test_split: float = 0.8,
    use_covariates: bool = True
) -> tuple:
    """
    Create GluonTS train/test datasets
    
    Args:
        df: Preprocessed national data
        target_col: Column to forecast (e.g., 'daily_cases')
        freq: Frequency ('D' for daily)
        prediction_length: Forecast horizon
        train_test_split: Train/test ratio
        use_covariates: Include mobility features
        
    Returns:
        (train_dataset, test_dataset, scaler)
    """
```

**Usage**:
```python
from src.data_processing.prepare_gluonts import create_gluonts_dataset
import pandas as pd

df = pd.read_csv('data/processed/national_data.csv')
train_ds, test_ds, scaler = create_gluonts_dataset(
    df,
    target_col='daily_cases',
    prediction_length=14
)

# Saves to:
#   data/gluonts/train/data.json
#   data/gluonts/test/data.json
```

**Output**:
- `data/gluonts/train/data.json` - Training ListDataset
- `data/gluonts/test/data.json` - Test ListDataset

---

### 2. `src/models/` - Model Training

#### `train_baseline.py` - Baseline Models

**Purpose**: Train simple baseline models for comparison

**Models Implemented**:
1. **Naive Forecast** - Last value repeated
2. **Seasonal Naive** - Value from 7 days ago (weekly seasonality)

**Key Functions**:

```python
def train_baseline_models(
    train_data: ListDataset,
    test_data: ListDataset,
    prediction_length: int = 14
) -> tuple:
    """
    Train Naive and Seasonal Naive models
    
    Returns:
        (models_dict, forecasts_dict)
    """
```

**Usage**:
```bash
python src/models/train_baseline.py
```

**Output**:
- `results/baseline_forecasts.png` - Visualization
- `models/baseline_model.pkl` - Saved predictors

---

#### `train_deepar.py` - DeepAR Model

**Purpose**: Train advanced probabilistic forecasting model

**Model**: DeepAR (RNN-based autoregressive model)

**Key Functions**:

```python
def train_deepar_model(
    train_data: ListDataset,
    prediction_length: int = 14,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    num_layers: int = 2,
    num_cells: int = 40,
    dropout_rate: float = 0.1,
    use_covariates: bool = True
) -> Predictor:
    """
    Train DeepAR model with specified hyperparameters
    
    Returns:
        Trained GluonTS Predictor
    """
```

**Usage**:
```bash
python src/models/train_deepar.py
```

**Output**:
- `results/deepar_forecast.png` - Visualization
- `models/deepar_model/` - Model checkpoint directory

---

### 3. `src/evaluation/` - Metrics & Evaluation

#### `metrics.py` - Evaluation Metrics

**Purpose**: Calculate forecast accuracy metrics

**Key Functions**:

```python
def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error"""
    
def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error"""
    
def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    
def calculate_crps(
    actual: np.ndarray,
    forecast: Forecast
) -> float:
    """Continuous Ranked Probability Score (for probabilistic forecasts)"""
    
def calculate_metrics(
    actual_values: np.ndarray,
    forecast_object: Forecast,
    metric_names: list = ['mae', 'rmse', 'mape', 'crps']
) -> dict:
    """
    Calculate multiple metrics at once
    
    Returns:
        Dictionary with metric names and values
    """
```

**Usage**:
```python
from src.evaluation.metrics import calculate_metrics

metrics = calculate_metrics(
    actual_values=test_data,
    forecast_object=forecast,
    metric_names=['mae', 'rmse', 'crps']
)

print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"CRPS: {metrics['crps']:.2f}")
```

---

### 4. `src/visualization/` - Plotting Utilities

#### `plot_utils.py` - Reusable Plot Functions

**Purpose**: Create consistent, high-quality visualizations

**Key Functions**:

```python
def plot_forecast(
    forecast: Forecast,
    actual: np.ndarray,
    title: str = "Forecast",
    save_path: str = None,
    show_intervals: bool = True
) -> None:
    """
    Plot forecast with confidence intervals
    
    Args:
        forecast: GluonTS Forecast object
        actual: Ground truth values
        title: Plot title
        save_path: Path to save figure
        show_intervals: Show 10th, 50th, 90th percentiles
    """
```

```python
def plot_multiple_forecasts(
    forecasts_dict: dict,
    actual: np.ndarray,
    title: str = "Model Comparison",
    save_path: str = None
) -> None:
    """
    Compare multiple model forecasts on one plot
    
    Args:
        forecasts_dict: {'Model Name': forecast_object, ...}
        actual: Ground truth
    """
```

**Usage**:
```python
from src.visualization.plot_utils import plot_forecast

plot_forecast(
    forecast=deepar_forecast,
    actual=test_data,
    title="DeepAR 14-Day Forecast",
    save_path="results/my_forecast.png"
)
```

---

#### `create_eda_plots.py` - Exploratory Data Analysis

**Purpose**: Create comprehensive EDA visualizations

**Key Functions**:

```python
def create_comprehensive_eda_plot(
    cases_ts: pd.DataFrame,
    deaths_ts: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    Create 4-panel EDA plot:
    1. Cumulative cases
    2. Daily new cases (with 7-day MA)
    3. Cumulative deaths
    4. Daily new deaths (with 7-day MA)
    
    Returns:
        Path to saved plot
    """
```

```python
def print_statistics(
    cases_ts: pd.DataFrame,
    deaths_ts: pd.DataFrame
) -> None:
    """Print comprehensive statistics"""
```

**Usage**:
```bash
python src/visualization/create_eda_plots.py
```

**Output**: `results/eda_visualization.png`

---

### 5. `src/utils/` - Utilities

#### `config.py` - Central Configuration

**Purpose**: Centralize project configuration

**Constants**:

```python
# Paths
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')

# Model parameters
PREDICTION_LENGTH = 14
CONTEXT_LENGTH = 30
TRAIN_TEST_SPLIT = 0.8

# Training
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# Target columns
TARGET_COL = 'daily_cases'
FREQ = 'D'
```

**Usage**:
```python
from src.utils.config import PREDICTION_LENGTH, EPOCHS

estimator = DeepAREstimator(
    prediction_length=PREDICTION_LENGTH,
    trainer=Trainer(epochs=EPOCHS)
)
```

---

## 🎯 Main Pipeline: `run_pipeline.py`

**Purpose**: Run entire workflow end-to-end

**Steps**:
1. Load data
2. Preprocess to national level
3. Create EDA visualizations
4. Prepare GluonTS datasets
5. Train baseline models
6. Train DeepAR model
7. Evaluate and compare
8. Save results

**Usage**:
```bash
python run_pipeline.py
```

**Options**:
```bash
# Run with specific steps
python run_pipeline.py --steps preprocess train_baseline

# Skip EDA
python run_pipeline.py --skip-eda

# Custom prediction length
python run_pipeline.py --prediction-length 30
```

**Output**:
```
results/
├── eda_visualization.png
├── baseline_forecasts.png
├── deepar_forecast.png
└── metrics.json
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Optional: GPU settings
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=NaiveEngine

# Optional: Paths
export DATA_DIR=/path/to/data
export RESULTS_DIR=/path/to/results
```

### Modifying Hyperparameters

Edit `src/utils/config.py` or pass directly:

```python
# In src/models/train_deepar.py
estimator = DeepAREstimator(
    prediction_length=14,      # Forecast horizon
    context_length=30,         # Historical context
    num_layers=2,              # LSTM layers
    num_cells=40,              # Hidden units
    dropout_rate=0.1,          # Regularization
    epochs=10,                 # Training iterations
    learning_rate=1e-3,        # Step size
)
```

---

## 🎨 Usage Examples

### Example 1: Custom Preprocessing

```python
from src.data_processing.load_data import DataLoader
from src.data_processing.preprocess import aggregate_to_national

loader = DataLoader()
cases_df = loader.load_cases()
deaths_df = loader.load_deaths()

# Custom aggregation (e.g., state-level)
state_df = cases_df.groupby(['Province_State', 'Date'])['Confirmed'].sum()

# Continue with modeling...
```

### Example 2: Train with Different Target

```python
from src.data_processing.prepare_gluonts import create_gluonts_dataset

# Forecast deaths instead of cases
train_ds, test_ds, scaler = create_gluonts_dataset(
    df,
    target_col='daily_deaths',  # Changed from 'daily_cases'
    prediction_length=14
)

# Train model as usual
```

### Example 3: Ensemble Forecasting

```python
from src.models.train_baseline import train_baseline_models
from src.models.train_deepar import train_deepar_model

# Train multiple models
baseline_forecasts = train_baseline_models(train_ds, test_ds)
deepar_forecast = train_deepar_model(train_ds)

# Average predictions
ensemble = (baseline_forecasts['seasonal_naive'] + deepar_forecast) / 2
```

---

## 📊 Data Flow

```
Raw Data (CSV)
      ↓
load_data.py (DataLoader)
      ↓
Pandas DataFrames
      ↓
preprocess.py
      ↓
National-level CSV
      ↓
prepare_gluonts.py
      ↓
GluonTS ListDataset
      ↓
train_*.py (Models)
      ↓
Forecasts
      ↓
metrics.py (Evaluation)
      ↓
plot_utils.py (Visualization)
      ↓
Results (PNG, JSON)
```

---

## 🚀 Quick Commands

```bash
# Full pipeline
python run_pipeline.py

# Individual steps
python src/data_processing/preprocess.py
python src/data_processing/prepare_gluonts.py
python src/visualization/create_eda_plots.py
python src/models/train_baseline.py
python src/models/train_deepar.py

# Docker
docker-compose exec covid-forecasting python run_pipeline.py
```

---

## 🐛 Debugging

### Enable Verbose Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Data

```python
from src.data_processing.load_data import DataLoader

loader = DataLoader()
cases_df = loader.load_cases()

print(f"Shape: {cases_df.shape}")
print(f"Columns: {cases_df.columns.tolist()}")
print(f"Date range: {cases_df.columns[11]} to {cases_df.columns[-1]}")
```

### Inspect GluonTS Dataset

```python
from gluonts.dataset.common import load_datasets

train_ds, test_ds = load_datasets(
    'data/gluonts',
    train_file='train/data.json',
    test_file='test/data.json'
)

for entry in train_ds:
    print(f"Start: {entry['start']}")
    print(f"Target length: {len(entry['target'])}")
    break
```

---

## 📚 Adding New Components

### Add New Model

1. Create `src/models/train_mymodel.py`
2. Import GluonTS estimator
3. Follow pattern from `train_deepar.py`
4. Update `run_pipeline.py`

### Add New Metric

1. Edit `src/evaluation/metrics.py`
2. Add function following existing pattern
3. Update `calculate_metrics()` function

### Add New Visualization

1. Edit `src/visualization/plot_utils.py`
2. Create function with signature: `plot_*(data, save_path)`
3. Use consistent styling

---

## 🎓 Best Practices

1. **Imports**: Use absolute imports (`from src.module import func`)
2. **Docstrings**: Document all functions with Args, Returns
3. **Type Hints**: Use type hints for function signatures
4. **Error Handling**: Add try-except for file I/O
5. **Logging**: Use logging instead of print() for production
6. **Configuration**: Centralize parameters in `src/utils/config.py`
7. **Testing**: Add unit tests in `tests/` (future)

---

**For detailed model information, see [MODEL_GUIDE.md](MODEL_GUIDE.md)**

**For dataset details, see [DATASET_GUIDE.md](DATASET_GUIDE.md)**

