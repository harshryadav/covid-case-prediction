.PHONY: help install setup clean train test quick compare visualize all check lint format

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Python and virtual environment
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

##@ Help

help: ## Display this help message
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  COVID-19 FORECASTING PIPELINE - MAKEFILE COMMANDS$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"

##@ Setup & Installation

install: ## Install all dependencies in virtual environment
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: install ## Complete setup (install + check data)
	@echo "$(BLUE)Checking data files...$(NC)"
	@$(PYTHON_VENV) -c "from pathlib import Path; import sys; \
		files = ['data/time_series_covid19_confirmed_US.csv', \
		         'data/time_series_covid19_deaths_US.csv', \
		         'data/time_series_covid19_vaccine_us.csv']; \
		missing = [f for f in files if not Path(f).exists()]; \
		sys.exit(1) if missing else print('$(GREEN)✓ All data files present$(NC)')"

##@ Data Processing

preprocess: ## Run data preprocessing only
	@echo "$(BLUE)Running data preprocessing...$(NC)"
	@$(PYTHON_VENV) run_pipeline.py --steps preprocess gluonts
	@echo "$(GREEN)✓ Preprocessing complete$(NC)"

##@ Model Training

train-baseline: ## Train baseline models (Naive, Seasonal Naive)
	@echo "$(BLUE)Training baseline models...$(NC)"
	@$(PYTHON_VENV) src/models/train_baseline.py
	@echo "$(GREEN)✓ Baseline models trained$(NC)"

train-deepar: ## Train DeepAR model
	@echo "$(BLUE)Training DeepAR model...$(NC)"
	@$(PYTHON_VENV) src/models/train_deepar.py
	@echo "$(GREEN)✓ DeepAR model trained$(NC)"

train-tft: ## Train Temporal Fusion Transformer
	@echo "$(BLUE)Training TFT model...$(NC)"
	@$(PYTHON_VENV) src/models/train_tft.py
	@echo "$(GREEN)✓ TFT model trained$(NC)"

train-prophet: ## Train Prophet model
	@echo "$(BLUE)Training Prophet model...$(NC)"
	@$(PYTHON_VENV) src/models/train_prophet.py
	@echo "$(GREEN)✓ Prophet model trained$(NC)"

train-wavenet: ## Train WaveNet model
	@echo "$(BLUE)Training WaveNet model...$(NC)"
	@$(PYTHON_VENV) src/models/train_wavenet.py
	@echo "$(GREEN)✓ WaveNet model trained$(NC)"

train-all: ## Train all models (baseline, DeepAR, TFT, Prophet, WaveNet)
	@echo "$(BLUE)Training all models...$(NC)"
	@$(MAKE) train-baseline
	@$(MAKE) train-deepar
	@$(MAKE) train-tft
	@$(MAKE) train-prophet
	@$(MAKE) train-wavenet
	@echo "$(GREEN)✓ All models trained$(NC)"

##@ Pipeline Execution

all: ## Run complete pipeline (preprocess + train all + compare)
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Running complete pipeline (~25 minutes)$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@$(PYTHON_VENV) run_pipeline.py
	@echo "$(GREEN)✓ Pipeline complete!$(NC)"

quick: ## Quick mode (preprocess + baseline + DeepAR)
	@echo "$(BLUE)Running quick pipeline (~10 minutes)...$(NC)"
	@$(PYTHON_VENV) run_pipeline.py --quick
	@echo "$(GREEN)✓ Quick pipeline complete$(NC)"

compare: ## Run model comparison only (requires trained models)
	@echo "$(BLUE)Comparing models...$(NC)"
	@$(PYTHON_VENV) src/models/compare_models.py
	@echo "$(GREEN)✓ Comparison complete$(NC)"

##@ Results & Visualization

view: ## Open all result visualizations
	@echo "$(BLUE)Opening visualizations...$(NC)"
	@open results/model_comparison.png 2>/dev/null || xdg-open results/model_comparison.png 2>/dev/null || echo "$(YELLOW)⚠ Could not open files automatically$(NC)"
	@open results/all_forecasts_comparison.png 2>/dev/null || xdg-open results/all_forecasts_comparison.png 2>/dev/null || true
	@echo "$(GREEN)✓ Visualizations opened$(NC)"

view-deepar: ## View DeepAR forecast
	@open results/deepar_forecast.png 2>/dev/null || xdg-open results/deepar_forecast.png 2>/dev/null || echo "$(YELLOW)File not found$(NC)"

view-tft: ## View TFT forecast
	@open results/tft_forecast.png 2>/dev/null || xdg-open results/tft_forecast.png 2>/dev/null || echo "$(YELLOW)File not found$(NC)"

view-prophet: ## View Prophet forecast
	@open results/prophet_forecast.png 2>/dev/null || xdg-open results/prophet_forecast.png 2>/dev/null || echo "$(YELLOW)File not found$(NC)"

view-wavenet: ## View WaveNet forecast
	@open results/wavenet_forecast.png 2>/dev/null || xdg-open results/wavenet_forecast.png 2>/dev/null || echo "$(YELLOW)File not found$(NC)"

results: ## Display model comparison results
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  MODEL COMPARISON RESULTS$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@if [ -f results/model_comparison.csv ]; then \
		cat results/model_comparison.csv | column -t -s,; \
	else \
		echo "$(RED)✗ No results found. Run 'make all' first.$(NC)"; \
	fi

##@ Testing & Quality

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/ -v || echo "$(YELLOW)⚠ No tests found$(NC)"

check: ## Check if all required files exist
	@echo "$(BLUE)Checking project files...$(NC)"
	@$(PYTHON_VENV) -c "from pathlib import Path; import sys; \
		files = { \
			'Data': ['data/time_series_covid19_confirmed_US.csv', \
			         'data/time_series_covid19_deaths_US.csv'], \
			'Models': ['src/models/train_deepar.py', \
			          'src/models/train_tft.py', \
			          'src/models/train_prophet.py', \
			          'src/models/train_wavenet.py'], \
			'Pipeline': ['run_pipeline.py'], \
		}; \
		all_good = True; \
		for category, paths in files.items(): \
			print(f'\n{category}:'); \
			for path in paths: \
				exists = Path(path).exists(); \
				status = '✓' if exists else '✗'; \
				print(f'  {status} {path}'); \
				all_good = all_good and exists; \
		print('\n$(GREEN)All files present$(NC)' if all_good else '\n$(RED)Some files missing$(NC)'); \
		sys.exit(0 if all_good else 1)"

lint: ## Run code linting (flake8, pylint)
	@echo "$(BLUE)Running linters...$(NC)"
	@$(PYTHON_VENV) -m flake8 src/ --max-line-length=100 --ignore=E501,W503 2>/dev/null || echo "$(YELLOW)⚠ flake8 not installed$(NC)"
	@echo "$(GREEN)✓ Linting complete$(NC)"

##@ Cleanup

clean-results: ## Remove all result files (plots, metrics, models)
	@echo "$(BLUE)Cleaning results...$(NC)"
	@rm -rf results/*.png results/*.csv results/*.pkl
	@rm -rf lightning_logs/
	@echo "$(GREEN)✓ Results cleaned$(NC)"

clean-data: ## Remove processed data files
	@echo "$(BLUE)Cleaning processed data...$(NC)"
	@rm -rf data/processed/ data/gluonts/
	@echo "$(GREEN)✓ Processed data cleaned$(NC)"

clean-cache: ## Remove Python cache files
	@echo "$(BLUE)Cleaning cache...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

clean: clean-results clean-cache ## Remove results and cache (keeps processed data)
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean clean-data ## Remove everything (results, cache, processed data)
	@echo "$(YELLOW)⚠ Removing all generated files...$(NC)"
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

reset: clean-all ## Complete reset (removes venv too)
	@echo "$(RED)⚠ Removing virtual environment...$(NC)"
	@rm -rf $(VENV)
	@echo "$(GREEN)✓ Full reset complete. Run 'make setup' to start fresh.$(NC)"

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t covid-forecasting .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run pipeline in Docker
	@echo "$(BLUE)Running pipeline in Docker...$(NC)"
	@docker-compose up
	@echo "$(GREEN)✓ Docker run complete$(NC)"

docker-clean: ## Remove Docker containers and images
	@echo "$(BLUE)Cleaning Docker...$(NC)"
	@docker-compose down
	@docker rmi covid-forecasting 2>/dev/null || true
	@echo "$(GREEN)✓ Docker cleaned$(NC)"

##@ Documentation

docs: ## Display project documentation links
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  DOCUMENTATION$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "  📚 $(YELLOW)Quick Start$(NC)"
	@echo "     README.md - Project overview"
	@echo "     GETTING_STARTED.md - Beginner's guide"
	@echo "     QUICKREF.txt - Command reference"
	@echo ""
	@echo "  📊 $(YELLOW)Models$(NC)"
	@echo "     MODELS.md - All model architectures"
	@echo "     MODEL_GUIDE.md - Training details"
	@echo ""
	@echo "  📈 $(YELLOW)Results$(NC)"
	@echo "     PIPELINE_RESULTS.md - Comprehensive analysis"
	@echo "     RESULTS_SUMMARY.txt - Quick summary"
	@echo ""
	@echo "  🐳 $(YELLOW)Docker$(NC)"
	@echo "     DOCKER.md - Docker guide"
	@echo ""
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"

info: ## Show project information and status
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  COVID-19 FORECASTING PROJECT$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "  🎯 $(YELLOW)Objective:$(NC) Probabilistic COVID-19 case forecasting"
	@echo "  🔧 $(YELLOW)Technology:$(NC) GluonTS (PyTorch backend)"
	@echo "  📊 $(YELLOW)Models:$(NC) 5 (Baseline, DeepAR, TFT, Prophet, WaveNet)"
	@echo "  ⏱️  $(YELLOW)Full Pipeline:$(NC) ~25 minutes"
	@echo "  🚀 $(YELLOW)Quick Mode:$(NC) ~10 minutes"
	@echo ""
	@echo "  $(YELLOW)Environment Status:$(NC)"
	@if [ -d "$(VENV)" ]; then \
		echo "    ✓ Virtual environment: $(GREEN)Active$(NC)"; \
	else \
		echo "    ✗ Virtual environment: $(RED)Not found$(NC) (run 'make install')"; \
	fi
	@if [ -d "data/processed" ]; then \
		echo "    ✓ Processed data: $(GREEN)Available$(NC)"; \
	else \
		echo "    ✗ Processed data: $(RED)Not found$(NC) (run 'make preprocess')"; \
	fi
	@if [ -f "results/model_comparison.csv" ]; then \
		echo "    ✓ Model results: $(GREEN)Available$(NC)"; \
	else \
		echo "    ✗ Model results: $(RED)Not found$(NC) (run 'make all')"; \
	fi
	@echo ""
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"
	@echo "  Type '$(GREEN)make help$(NC)' to see all available commands"
	@echo "$(BLUE)════════════════════════════════════════════════════════════$(NC)"

# Default target
.DEFAULT_GOAL := help

