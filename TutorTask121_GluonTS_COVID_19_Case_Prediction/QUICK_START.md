# 🚀 Quick Start Guide

## Get Running in 3 Steps!

### Step 1: Build Docker Image

```bash
./docker_build.sh
```

This will:
- Install all dependencies
- Set up Python environment
- Install Jupyter
- Copy all project files

**Time**: ~2-3 minutes

---

### Step 2: Start Jupyter

```bash
./docker_jupyter.sh
```

This will:
- Start Jupyter Notebook server
- Expose on port 8888
- Display access token in terminal

**Look for output like**:
```
http://127.0.0.1:8888/tree?token=abc123...
```

---

### Step 3: Open in Browser

1. Copy the URL from terminal (with token)
2. Paste into your browser
3. You'll see all project files

---

## 📖 Which Notebook to Run First?

### For Beginners: Start Here! 👇

**`GluonTS_SimpleFeedForward.API.ipynb`**
- Simplest model
- Learn basics
- Fast (1 minute)

Then: **`GluonTS_SimpleFeedForward.example.ipynb`**
- Apply to real COVID data
- See complete workflow

### For Complete Pipeline: 👇

**`GluonTS_DeepAR.example.ipynb`**
- Full COVID-19 forecasting
- Load real data
- Train DeepAR
- Comprehensive analysis

---

## 🎯 Notebook Types

### API Notebooks (`.API.ipynb`)
- **Purpose**: Learn model API
- **Data**: Toy/synthetic
- **Time**: 1-2 minutes
- **Best for**: Understanding how models work

### Example Notebooks (`.example.ipynb`)
- **Purpose**: Complete pipelines
- **Data**: Real COVID-19
- **Time**: 2-4 minutes
- **Best for**: Production-ready forecasting

---

## 🔧 Running a Notebook

1. **Click on notebook** in Jupyter file browser
2. **Run cells** top to bottom:
   - Click cell
   - Press `Shift + Enter`
   - Or click "Run" button
3. **Read explanations** in markdown cells
4. **Watch output** as models train

---

## 📊 Expected Outputs

### Training
```
Training DeepAR model...
================================================
📚 Model is learning patterns from the data...
✓ Training complete!
```

### Forecasts
```
🔮 Generating forecasts...
✓ Forecasts generated!
Mean prediction: 45,234 cases
```

### Visualizations
- Forecast vs actual plots
- Confidence intervals
- Error analysis
- Saved as PNG files

---

## 🆘 Troubleshooting

### Docker won't build?
```bash
# Check Docker is running
docker ps

# If not installed: install Docker Desktop
```

### Jupyter won't start?
```bash
# Check if port 8888 is in use
lsof -i :8888

# Stop conflicting process or change port in docker_jupyter.sh
```

### Import errors in notebooks?
```bash
# Restart Jupyter kernel
# In Jupyter: Kernel → Restart Kernel

# Or rebuild Docker
./docker_build.sh
```

### Data not found?
```bash
# Check data directory exists
ls data/

# Should see: cases.csv, deaths.csv, mobility.csv, vaccine.csv
```

---

## 💡 Tips

1. **Run cells in order** - Don't skip ahead
2. **Read markdown cells** - They explain what's happening
3. **Watch training progress** - See the model learn
4. **Save plots** - They're saved automatically as PNG
5. **Experiment** - Change parameters and rerun!

---

## 📚 Next Steps

After running notebooks:

1. **Compare models**
   - Run all 3 models (DeepAR, SimpleFeedForward, DeepNPTS)
   - Compare metrics (MAE, RMSE, MAPE)
   - See which works best

2. **Experiment**
   - Change `hidden_size` parameter
   - Try different `context_length`
   - Add/remove features

3. **Read documentation**
   - `docs/NOTEBOOKS_COMPLETE.md` - Full overview
   - `docs/ARCHITECTURE.md` - Design details
   - `docs/DOCKER_GUIDE.md` - Docker help

---

## 🎓 Learning Path

### Beginner (20 minutes)
1. SimpleFeedForward.API
2. SimpleFeedForward.example
3. DeepAR.API

### Intermediate (40 minutes)
4. DeepAR.example
5. DeepNPTS.API
6. DeepNPTS.example

### Advanced
- Modify parameters
- Add new features
- Try different data periods
- Implement scenario analysis

---

## ✅ Success Checklist

- [ ] Docker built successfully
- [ ] Jupyter started
- [ ] Opened notebook in browser
- [ ] Ran first notebook (SimpleFeedForward.API)
- [ ] Saw training complete
- [ ] Saw forecast plot
- [ ] Understood the workflow

---

## 🚀 You're Ready!

Your COVID-19 forecasting project is fully set up and ready to use!

**Start with**: `GluonTS_SimpleFeedForward.API.ipynb`

**Have fun forecasting!** 📈

