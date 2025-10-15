"""
Configuration settings for COVID-19 forecasting project
"""

from pathlib import Path


# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data files
DATA_FILES = {
    'cases': DATA_DIR / 'time_series_covid19_confirmed_US.csv',
    'deaths': DATA_DIR / 'time_series_covid19_deaths_US.csv',
    'vaccines': DATA_DIR / 'time_series_covid19_vaccine_us.csv',
    'mobility': DATA_DIR / 'mobility_report_US.csv'
}

# Model parameters
MODEL_PARAMS = {
    'prediction_length': 14,  # Forecast 14 days ahead
    'context_length': 56,     # Use 8 weeks of history
    'test_days': 60,          # Reserve last 60 days for testing
    'freq': 'D'               # Daily frequency
}

# Training parameters
TRAINING_PARAMS = {
    'epochs': 50,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_batches_per_epoch': 50
}

# DeepAR specific
DEEPAR_PARAMS = {
    'num_layers': 2,
    'num_cells': 40,
    'dropout_rate': 0.1
}

# Transformer specific
TRANSFORMER_PARAMS = {
    'model_dim': 32,
    'num_heads': 4,
    'num_layers': 3,
    'dropout_rate': 0.1
}


if __name__ == "__main__":
    print("Configuration Settings:")
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"\nModel parameters: {MODEL_PARAMS}")
    print(f"Training parameters: {TRAINING_PARAMS}")

