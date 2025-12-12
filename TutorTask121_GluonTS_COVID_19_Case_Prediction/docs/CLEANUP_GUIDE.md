# 🧹 Project Cleanup Guide

## Temporary/Generated Files (Safe to Delete)

These files are automatically generated during development and should **not** be included in your submission:

### 1. Lightning Logs ⚡
```bash
lightning_logs/
```
- **What**: PyTorch Lightning training logs and checkpoints
- **Created by**: GluonTS models during training
- **Contains**: Model checkpoints (`.ckpt`), training metrics, version folders
- **Needed?**: ❌ No - just training artifacts
- **Already in `.gitignore`**: ✅ Yes (line 30)

### 2. Python Cache 🐍
```bash
__pycache__/
*.pyc
```
- **What**: Compiled Python bytecode
- **Created by**: Python interpreter
- **Needed?**: ❌ No
- **Already in `.gitignore`**: ✅ Yes

### 3. Jupyter Checkpoints 📓
```bash
.ipynb_checkpoints/
```
- **What**: Jupyter notebook autosaves
- **Created by**: Jupyter
- **Needed?**: ❌ No
- **Already in `.gitignore`**: ✅ Yes

### 4. IDE Files 💻
```bash
.vscode/
.idea/
.DS_Store
```
- **What**: Editor/IDE configuration and macOS metadata
- **Created by**: VS Code, PyCharm, macOS
- **Needed?**: ❌ No
- **Already in `.gitignore`**: ✅ Yes

## Quick Cleanup Commands

### Clean All Temporary Files
```bash
cd TutorTask121_GluonTS_COVID_19_Case_Prediction

# Remove PyTorch Lightning logs
rm -rf lightning_logs/

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +

# Remove macOS metadata
find . -name ".DS_Store" -delete

echo "✓ Cleanup complete!"
```

### Verify Clean State
```bash
# Check what would be committed to git
git status

# Should NOT see:
# - lightning_logs/
# - __pycache__/
# - .ipynb_checkpoints/
# - .DS_Store
```

## Files That SHOULD Be Included in Submission

### ✅ Required Files (Keep These!)

**Root Level**:
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container setup
- `docker_*.sh` - Helper scripts
- `GluonTS.API.md` - API documentation
- `GluonTS.example.md` - Example documentation
- `GluonTS_*.ipynb` - All 6 Jupyter notebooks (3 API + 3 example)

**Data Folder** (`data/`):
- `cases.csv` - COVID-19 cases
- `deaths.csv` - COVID-19 deaths
- `vaccine.csv` - Vaccination data
- `mobility.csv` - Mobility data

**Utils Folder** (`utils/`):
- `load_data_utils.py` - Data loading functions
- `preprocess_data_utils.py` - Data preprocessing
- `gluonts_utils.py` - GluonTS data preparation
- `evaluation_utils.py` - Metrics and plotting
- `data_loader_for_notebooks.py` - Notebook data loader

**Docs Folder** (`docs/`):
- Any documentation you create
- Planning files, architecture notes, etc.

## Automated Cleanup (Recommended)

Add this to your workflow before submission:

```bash
#!/bin/bash
# cleanup_before_submission.sh

cd TutorTask121_GluonTS_COVID_19_Case_Prediction

echo "🧹 Cleaning up temporary files..."

# Remove all ignored files
rm -rf lightning_logs/
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type d -name ".ipynb_checkpoints" -exec rm -r {} + 2>/dev/null
find . -name ".DS_Store" -delete 2>/dev/null

echo "✅ Cleanup complete!"
echo ""
echo "Verifying git status..."
git status --short
```

## .gitignore Already Configured ✅

Your `.gitignore` already handles all these cases:

```gitignore
# PyTorch Lightning
lightning_logs/        ← Ignores training logs
checkpoints/
*.ckpt

# Python
__pycache__/          ← Ignores Python cache
*.pyc

# Jupyter
.ipynb_checkpoints    ← Ignores notebook checkpoints

# IDEs
.DS_Store             ← Ignores macOS files
.vscode/
.idea/
```

## Summary

**Safe to delete anytime**:
- `lightning_logs/` - Training artifacts (regenerated each run)
- `__pycache__/` - Python bytecode (regenerated automatically)
- `.ipynb_checkpoints/` - Notebook autosaves (recreated by Jupyter)
- `.DS_Store` - macOS metadata (useless outside macOS)

**Keep for submission**:
- Source code (`.py`, `.ipynb`)
- Documentation (`.md`)
- Data files (`.csv`)
- Configuration (`requirements.txt`, `Dockerfile`)

---

**Pro Tip**: Before submitting, run `git status` to see what files would be committed. If you see `lightning_logs/` or `__pycache__/`, something went wrong with `.gitignore`!

