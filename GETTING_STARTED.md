# Getting Started - COVID-19 Forecasting

**New to this project?** Follow these simple steps to run everything from start to finish.

---

## 🚀 Quick Start (5 Minutes)

### 1. **Clone the Repository** (if not already done)
```bash
git clone <your-repo-url>
cd covid-case-prediction
```

### 2. **Set Up Python Environment**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. **Run the Complete Pipeline**
```bash
python run_pipeline.py
```

That's it! ☕ Grab a coffee while it runs (~10-15 minutes).

---

## 📋 What Happens When You Run the Pipeline?

The pipeline automatically runs 5 steps:

1. **Download Data** → Downloads COVID-19 datasets from JHU CSSE
2. **Preprocess** → Cleans and merges the data
3. **Prepare GluonTS Data** → Converts to time series format
4. **Visualize** → Creates exploratory data analysis plots
5. **Train Models** → Trains baseline and DeepAR models

### Expected Output:
```
results/
├── eda_visualization.png       # EDA plots
├── baseline_forecasts.png      # Naive model forecasts
├── baseline_metrics.csv        # Baseline performance
├── deepar_forecast.png         # DeepAR neural network forecast
└── deepar_metrics.csv          # DeepAR performance
```

---

## 🔧 Run Individual Steps (Optional)

If you want to run steps one at a time:

```bash
# Always activate environment first!
source venv/bin/activate

# Step 1: Download and preprocess data
python src/data_processing/preprocess.py

# Step 2: Prepare for GluonTS
python src/data_processing/prepare_gluonts.py

# Step 3: Create visualizations
python src/visualization/create_eda_plots.py

# Step 4: Train baseline models
python src/models/train_baseline.py

# Step 5: Train DeepAR model
python src/models/train_deepar.py
```

---

## 📊 View Your Results

After running the pipeline:

```bash
# View all results (macOS)
open results/

# View specific plot
open results/deepar_forecast.png

# View metrics
cat results/deepar_metrics.csv
```

On Windows:
```bash
explorer results\
```

On Linux:
```bash
xdg-open results/
```

---

## ⚙️ System Requirements

- **Python**: 3.8 - 3.12 (tested on 3.10)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: ~500MB for data and models
- **Time**: ~10-15 minutes for full pipeline

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "No module named 'gluonts'"
```bash
# Install specific version
pip install gluonts==0.16.2 torch lightning
```

### Pipeline hangs or crashes
```bash
# Check available memory
free -h  # Linux/Mac
# Close other applications and try again
```

---

## 📚 Next Steps

After your first successful run:

1. **Understand the Results** → Read `MODEL_GUIDE.md`
2. **Improve Performance** → See `SETUP_COMPLETE.md` for tuning tips
3. **Modify Models** → Edit `src/models/train_deepar.py`
4. **Add Features** → Incorporate mobility data (see `DATASET_GUIDE.md`)

---

## 🎯 Common Commands Reference

```bash
# Activate environment (ALWAYS DO THIS FIRST!)
source venv/bin/activate

# Run full pipeline
python run_pipeline.py

# Run single model
python src/models/train_deepar.py

# Check what's installed
pip list | grep gluonts

# Deactivate environment when done
deactivate
```

---

## 📖 Full Documentation

| File | Purpose |
|------|---------|
| **GETTING_STARTED.md** | 👈 You are here! |
| `README.md` | Project overview |
| `QUICKSTART.md` | Detailed 30-min tutorial |
| `SETUP_COMPLETE.md` | Complete setup & troubleshooting |
| `MODEL_GUIDE.md` | Model architecture explained |
| `DATASET_GUIDE.md` | Data sources & structure |
| `REFERENCE.md` | Quick command reference |

---

## ✅ Success Checklist

After running the pipeline, you should have:

- [x] Virtual environment activated
- [x] All dependencies installed
- [x] Data downloaded to `data/`
- [x] Processed data in `data/processed/`
- [x] Visualizations in `results/`
- [x] Trained models completed
- [x] Metrics saved to CSV files

---

## 💡 Pro Tips

1. **Always activate venv first** → `source venv/bin/activate`
2. **Pipeline failed?** → Run individual steps to find the issue
3. **Want faster training?** → Edit `EPOCHS = 10` in `src/models/train_deepar.py`
4. **Need help?** → Check `SETUP_COMPLETE.md` for detailed troubleshooting

---

## 🎉 You're Ready!

Run this and you're good to go:

```bash
source venv/bin/activate
python run_pipeline.py
```

**Questions?** Check the documentation files above or the comprehensive guides in the repo.

---

**Happy Forecasting! 📈**

