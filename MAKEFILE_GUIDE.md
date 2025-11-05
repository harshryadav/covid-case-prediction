# Makefile Guide - Quick Reference

This project includes a comprehensive Makefile for easy command execution. Below is a guide to all available commands.

---

## 🚀 Quick Start

```bash
# First time setup
make setup          # Install dependencies + check data files

# Run complete pipeline
make all            # Train all models + comparison (~25 min)

# Or run quick mode
make quick          # Train baseline + DeepAR only (~10 min)

# View results
make results        # Display metrics table
make view           # Open visualization files
```

---

## 📋 All Available Commands

### Setup & Installation

| Command | Description | Time |
|---------|-------------|------|
| `make install` | Install all dependencies in virtual environment | 2-3 min |
| `make setup` | Complete setup (install + check data) | 2-3 min |
| `make check` | Verify all required files exist | <1 sec |

###  Data Processing

| Command | Description | Time |
|---------|-------------|------|
| `make preprocess` | Run data preprocessing only | <1 min |

### Model Training

| Command | Description | Time |
|---------|-------------|------|
| `make train-baseline` | Train baseline models (Naive, Seasonal Naive) | <1 sec |
| `make train-deepar` | Train DeepAR model | ~8 min |
| `make train-tft` | Train Temporal Fusion Transformer | ~10 min |
| `make train-prophet` | Train Prophet model | ~30 sec |
| `make train-wavenet` | Train WaveNet model | ~8 min |
| `make train-all` | Train all models | ~25 min |

### Pipeline Execution

| Command | Description | Time |
|---------|-------------|------|
| `make all` | Run complete pipeline (preprocess + train all + compare) | ~25 min |
| `make quick` | Quick mode (preprocess + baseline + DeepAR) | ~10 min |
| `make compare` | Run model comparison only (requires trained models) | <1 min |

### Results & Visualization

| Command | Description |
|---------|-------------|
| `make view` | Open all result visualizations |
| `make view-deepar` | View DeepAR forecast |
| `make view-tft` | View TFT forecast |
| `make view-prophet` | View Prophet forecast |
| `make view-wavenet` | View WaveNet forecast |
| `make results` | Display model comparison table in terminal |

### Testing & Quality

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make lint` | Run code linting |

### Cleanup

| Command | Description | Warning |
|---------|-------------|---------|
| `make clean` | Remove results and cache (keeps processed data) | ⚠️ Removes model outputs |
| `make clean-results` | Remove all result files (plots, metrics, models) | ⚠️ Removes visualizations |
| `make clean-data` | Remove processed data files | ⚠️ Need to reprocess |
| `make clean-cache` | Remove Python cache files | ✓ Safe |
| `make clean-all` | Remove everything (results, cache, processed data) | ⚠️⚠️ Full cleanup |
| `make reset` | Complete reset (removes venv too) | ⚠️⚠️⚠️ Nuclear option |

### Docker

| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker image |
| `make docker-run` | Run pipeline in Docker |
| `make docker-clean` | Remove Docker containers and images |

### Documentation

| Command | Description |
|---------|-------------|
| `make docs` | Display all documentation links |
| `make info` | Show project information and status |
| `make help` | Display all available commands (default) |

---

## 💡 Common Workflows

### First Time Setup

```bash
# 1. Install dependencies
make setup

# 2. Run complete pipeline
make all

# 3. View results
make results
make view
```

### Daily Development

```bash
# Quick iteration with one model
make train-deepar
make view-deepar

# Or test multiple models
make train-deepar train-tft
make compare
make view
```

### Clean & Rebuild

```bash
# Clean only results (keeps processed data)
make clean

# Re-run pipeline
make all

# Full clean & rebuild
make clean-all
make setup
make all
```

### Testing Individual Models

```bash
# Train specific model
make train-wavenet

# View its output
make view-wavenet

# Compare with others
make compare
```

---

## 🎯 Recommended Commands by Use Case

### **For Group Project Presentation:**

```bash
# 1. Initial setup
make setup

# 2. Run all models
make all

# 3. View comprehensive results
make results
make view

# 4. Generate report from:
#    - PIPELINE_RESULTS.md (detailed analysis)
#    - results/model_comparison.csv (metrics)
#    - results/*.png (visualizations)
```

### **For Quick Testing:**

```bash
# Quick mode (baseline + DeepAR only)
make quick

# Or individual model
make train-prophet view-prophet
```

### **For Development/Debugging:**

```bash
# Check everything is set up
make check
make info

# Train incrementally
make preprocess
make train-baseline
make train-deepar
make compare
```

---

## 📊 Output Files

After running `make all`, you'll have:

```
results/
├── model_comparison.png          # Bar charts comparing metrics
├── all_forecasts_comparison.png  # Side-by-side forecasts
├── deepar_forecast.png           # DeepAR individual forecast
├── tft_forecast.png              # TFT individual forecast
├── prophet_forecast.png          # Prophet individual forecast
├── wavenet_forecast.png          # WaveNet individual forecast
├── baseline_forecasts.png        # Baseline models
├── model_comparison.csv          # Metrics table
├── model_rankings.csv            # Ranked by metric
├── baseline_metrics.csv          # Baseline detailed metrics
├── deepar_metrics.csv            # DeepAR detailed metrics
├── tft_metrics.csv               # TFT detailed metrics
├── prophet_metrics.csv           # Prophet detailed metrics
└── wavenet_metrics.csv           # WaveNet detailed metrics
```

---

## ⚙️ Configuration

The Makefile uses these settings:

- **Python**: `python3` (from your `venv/`)
- **Virtual Environment**: `venv/`
- **Results Directory**: `results/`
- **Data Directory**: `data/`

---

## 🐛 Troubleshooting

### "make: command not found"

```bash
# Install make (if on macOS)
xcode-select --install

# Or use direct Python commands
python run_pipeline.py
```

### "Virtual environment not found"

```bash
# Recreate venv
make reset
make setup
```

### "Data files missing"

```bash
# Check which files are missing
make check

# Ensure you have:
# - data/time_series_covid19_confirmed_US.csv
# - data/time_series_covid19_deaths_US.csv
# - data/time_series_covid19_vaccine_us.csv
```

### "Models not found for comparison"

```bash
# Train models first
make train-all

# Then compare
make compare
```

---

## 📚 Related Documentation

- **README.md** - Project overview
- **GETTING_STARTED.md** - Beginner's guide
- **MODELS.md** - Model details
- **PIPELINE_RESULTS.md** - Results analysis
- **DOCKER.md** - Docker guide

---

## 🎓 Tips & Tricks

1. **Chain Commands**: `make clean && make all`
2. **Parallel Training**: Train models individually in separate terminals
3. **Quick Iteration**: Use `make clean-results` (keeps processed data)
4. **View Help Anytime**: `make help` or just `make`
5. **Check Status**: `make info` shows current project state

---

**Need more help?** Run `make help` to see all commands with descriptions.

