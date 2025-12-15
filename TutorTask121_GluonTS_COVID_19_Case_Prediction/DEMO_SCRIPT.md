# COVID-19 Case Prediction with GluonTS - Demo Script

**Total Time: 15-20 minutes**
**Presenters: Harsh, Utkrisht, Deepika**

---

## HARSH - Steps 1-4 (2 minutes)

### Step 1: Introduction (20 seconds)

"Hello! I'm Harsh Yadav, UID [your UID]. We're presenting **COVID-19 Case Prediction using GluonTS** - a probabilistic time series forecasting library. We chose **Hard difficulty**. Our project forecasts COVID-19 cases 14 days ahead using three models: DeepAR, SimpleFeedForward, and DeepNPTS."

### Step 2: File Structure (30 seconds)

**[Show file browser]:**

"Our project follows all naming conventions:
- Notebooks: `GluonTS.API.ipynb`, `GluonTS.example.ipynb`
- Documentation: `README.md`, `GluonTS.API.md`, `GluonTS.example.md`
- Utilities: `GluonTS_utils_*.py` (7 modules for data, preprocessing, models, evaluation)
- Docker: `Dockerfile`, build/run scripts, `requirements.txt`
- Data: Auto-downloads from Google Drive if missing"

### Step 3: Docker Execution (30 seconds)

**[Terminal]:**

```bash
./docker_build.sh
./docker_jupyter.sh
```

"Docker builds with Python 3.10, PyTorch, GluonTS. Container runs on port 8888 with workspace mounted. MPS fallback enabled for Apple Silicon."

### Step 4: Open Notebooks (40 seconds)

**[Browser at localhost:8888]:**

"Two main notebooks:
- **API notebook**: Demonstrates each model separately
- **Example notebook**: Complete COVID-19 forecasting application

Starting with API notebook - each team member covers their assigned model:
- Part 1: DeepAR - Me
- Part 2: SimpleFeedForward - Utkrisht  
- Part 3: DeepNPTS - Deepika

Let's begin."

---

## Step 5: Full Project Walkthrough (12-14 minutes)

### HARSH - DeepAR (GluonTS.API.ipynb) - 4 minutes

#### Setup (30 seconds)

**[Run Cells 0-5]:**

"Notebook intro explains we're using COVID-19 data with multiple waves, seasonality, and external factors.

Cell 5 loads data automatically:
- Training: 1,123 days (Jan 2020 - Feb 2023)
- Testing: 14 days (late Feb - early Mar 2023)
- Target: Daily cases (7-day moving average)
- Features: 3 (deaths data, CFR)"

**[Run Cell 7 - Visualization]:**

"Timeline shows multiple COVID waves, weekly seasonality, and declining trend. This complexity requires sophisticated models."

#### DeepAR Configuration (30 seconds)

**[Run Cell 10]:**

"DeepAR uses RNNs with memory to learn patterns.

Key parameters:
- Forecast: 14 days ahead, using 60 days history
- Features: 3 external features (deaths, CFR)
- Architecture: 2 RNN layers, 40 hidden units
- Training: 10 epochs for demo (use 20-30 in production)"

#### Training & Forecasting (1 minute)

**[Run Cell 12]:**

"Training with PyTorch Lightning. Loss decreasing from 11.7 to 11.1 - model is learning. Takes 2-3 minutes on CPU."

**[Run Cell 14]:**

"Generating probabilistic forecasts with 100 samples for uncertainty.
- Mean: ~49,000 cases/day
- 90% CI: 15,000 - 81,000 cases

Wide confidence intervals reflect COVID's unpredictability."

#### Visualization & Evaluation (1 minute)

**[Run Cell 16]:**

"Forecast visualization shows:
- Green line: DeepAR prediction
- Green bands: 80% and 90% confidence intervals  
- Orange dots: Actual values
- DeepAR captures declining trend, provides realistic uncertainty"

**[Run Cell 18]:**

"Performance metrics:
- MAE: 13,905 cases
- MAPE: 40.2%

For 14-day COVID forecasts, this is reasonable. Real deployment would tune parameters further."

**Key takeaway:** "DeepAR learns complex patterns, quantifies uncertainty appropriately."

---

### UTKRISHT - SimpleFeedForward (GluonTS.API.ipynb) - 3 minutes

#### Introduction & Configuration (30 seconds)

"SimpleFeedForward is our fast baseline - no RNN memory, direct mapping from history to future.

**[Run Cell 21]:**

Configuration:
- Same forecast horizon: 14 days, 60-day context
- Simpler: 2 feedforward layers
- Key difference: NO external features, NO frequency parameter
- Training: 20 epochs since it's fast"

#### Training & Forecasting (45 seconds)

**[Run Cell 23]:**

"Watch the training speed - completes in ~30 seconds vs 2-3 minutes for DeepAR. That's 10x faster!

Loss decreasing from 12.9 to 12.8 - still learning despite simplicity."

**[Run Cell 25]:**

"Forecasts:
- Mean: ~48,000 cases (similar to DeepAR)
- But wider confidence intervals - less certain"

#### Visualization & Evaluation (45 seconds)

**[Run Cell 27]:**

"Purple forecast line is smoother than DeepAR:
- Doesn't capture weekly seasonality
- Extrapolates smooth trend
- Some actual values outside confidence bands"

**[Run Cell 29]:**

"Metrics:
- MAE: 14,055 (only 150 worse than DeepAR)
- MAPE: 40.2% (same as DeepAR!)

Only marginally worse accuracy but 10x faster training."

**Key takeaway:** "SimpleFeedForward excellent for quick experiments and baselines. Speed vs accuracy tradeoff."

---

### DEEPIKA - DeepNPTS (GluonTS.API.ipynb) - 3 minutes

#### Introduction & Configuration (30 seconds)

"DeepNPTS is non-parametric - learns the data distribution without assumptions.

Perfect for COVID because each variant behaves differently. No rigid distribution assumptions.

**[Run Cell 37]:**

Configuration similar to DeepAR but:
- `epochs=10` passed directly (not via trainer_kwargs)
- `num_hidden_nodes=[40,40]` (not hidden_size)
- Still uses 3 external features"

#### Training & Forecasting (45 seconds)

**[Run Cell 39]:**

"Training takes 1-2 minutes - between SimpleFeedForward and DeepAR.

Loss: 12.6 to 12.4 - learning the distribution shape, not just the mean."

**[Run Cell 41]:**

"Forecasts show interesting differences:
- Mean: ~45,000 (lower than others!)
- 90% CI: 11,000 - 76,000 (much wider)

DeepNPTS predicting lower cases, wider uncertainty - adapting to recent declining trend."

#### Visualization & Evaluation (45 seconds)

**[Run Cell 43]:**

"Orange forecast more aggressive decline:
- Widest confidence intervals (honest about uncertainty)
- Follows recent regime shift
- Some actual values closer to DeepNPTS in later days"

**[Run Cell 45]:**

"Performance - **Best of all three!**
- MAE: 12,785 (best)
- MAPE: 36.6% (best)

Non-parametric flexibility helped adapt to test period patterns."

**Key takeaway:** "DeepNPTS excels during regime changes. Best for volatile, changing data."

---

### ALL THREE - Example Notebook Walkthrough (2-3 minutes)

**[Switch to GluonTS.example.ipynb]:**

#### Problem & Data (Harsh - 45 seconds)

"Example notebook addresses real problem: hospitals need 14-day forecasts for resource allocation.

**[Run setup cells, show data viz]:**

Complete COVID timeline with multiple waves. Features: deaths and mobility data correlate with cases."

#### Model Training & Comparison (Utkrisht - 45 seconds)

"Trains all three models with same configuration.

**[Show comparison table]:**

Side-by-side metrics:
- All achieve 36-40% MAPE
- DeepNPTS slightly better for this period
- SimpleFeedForward fastest
- Choose based on needs: accuracy, speed, or adaptability"

#### Scenario Analysis (Deepika - 45 seconds)

"Most powerful feature - simulating interventions:

**[Show scenario outputs]:**

Three scenarios:
- Baseline: 65,000 cases
- Moderate intervention (-20% mobility): 52,000 cases (20% reduction)
- Strong intervention (-40% mobility): 38,000 cases (42% reduction)

Quantifies policy impact - helps decision-makers balance health vs economic costs."

---

## UTKRISHT - Step 6: Results Discussion (2 minutes)

"Key findings:

**Performance:**
- All three models achieved 36-40% MAPE for 14-day forecasts
- DeepNPTS best (36.6%), DeepAR and SimpleFeedForward tied (40.2%)
- Reasonable accuracy for volatile COVID data

**Model Selection:**
- DeepAR: Best for complex patterns, seasonality
- SimpleFeedForward: Best for speed (10x faster), good baseline
- DeepNPTS: Best for regime changes, adaptive

**Uncertainty Quantification:**
- 90% confidence intervals: ±15,000 cases
- Critical for hospital planning - prepare for variance
- DeepNPTS most honest with widest intervals

**Scenario Analysis Value:**
- Quantified intervention impacts: 20-42% case reduction
- Directly supports policy decisions
- Better than simple point forecasts

**How GluonTS Solved the Problem:**

Hospitals needed 14-day forecasts → We delivered:
- Multiple model options for different scenarios
- Probabilistic forecasts with uncertainty
- Feature integration (deaths, mobility)
- Scenario simulation capability
- Fast enough for daily retraining (<10 min total)

**Real-World Impact:**
- ICU bed allocation 2 weeks ahead
- Staff scheduling optimization
- Supply chain management (PPE, ventilators)
- Policy evaluation before implementation
- Transparent public communication

**Technical Achievement:**
- Automated data pipeline with Google Drive download
- Three models trained in <10 minutes on CPU
- Comprehensive evaluation framework
- Production-ready Docker deployment
- Modular, extensible architecture

**Limitations:**
- 14-day horizon only (longer less accurate)
- Requires daily retraining for best results
- Can't predict entirely new variants
- National-level only (need more data for state/county)

This demonstrates hard difficulty through multiple complex models, sophisticated uncertainty quantification, scenario analysis, and production deployment."

---

## DEEPIKA - Step 7: Documentation Review (2 minutes)

"Our documentation serves all audiences:

### README.md - Entry Point

**[Show file]:**

"First thing anyone sees:
- Quick start: 5 minutes to running notebooks
- Mermaid diagrams: Visual workflow (no code reading needed)
- Data setup: Automatic download instructions
- Expected outputs: No surprises

Works for: project managers, developers, stakeholders, students"

### GluonTS.API.md - Tool Reference  

**[Show file]:**

"Tool-focused, no COVID specifics:
- Model overview: When to use each
- Parameter reference: All options explained
- Basic usage patterns: Step-by-step examples
- Troubleshooting: Common issues and fixes

Generic - works for ANY time series project (sales, traffic, weather).

Works for: data scientists, developers, students learning GluonTS"

### GluonTS.example.md - Project Documentation

**[Show file]:**

"Project-focused, all COVID-19 content:
- Problem statement: Why hospitals need this
- Data sources: Cases, deaths, mobility - why each matters
- Feature engineering: Why 7-day MA, CFR, etc.
- Model selection: Why these three for COVID
- Results interpretation: What metrics mean practically
- Real-world application: How to use forecasts

Works for: hospital administrators, public health officials, policy makers, technical reviewers"

### Documentation Quality

"Three key aspects:

**Completeness:** Every file, function, parameter documented. All results interpreted.

**Clarity:** Plain language, visual aids, progressive complexity. Technical AND non-technical explanations.

**Accessibility:** Multiple entry points. README → API or Example depending on needs."

### How They Work Together

"Different users, different paths:

**Student learning GluonTS:**
README → API.md → API.ipynb → Apply to own project

**Hospital administrator:**
README (diagrams!) → example.md → example.ipynb → Evaluate for their needs

**Developer extending work:**
README → API.md → Utilities → Build own application

Everyone can understand and use our work appropriately."

---

## CLOSING (All 3) - 30 seconds

**[Harsh concludes]:**

"Summary of achievements:

- **Complete forecasting system** with three GluonTS models
- **36-40% MAPE** for 14-day COVID forecasts
- **Probabilistic predictions** with uncertainty quantification  
- **Scenario analysis** quantifying intervention impacts (20-42% reduction)
- **Production-ready** Docker deployment, <10 min training
- **Comprehensive documentation** for all audiences
- **Modular architecture** for easy extension

GluonTS enables sophisticated probabilistic forecasting that directly supports real-world public health decision-making.

Questions?"

---

## TIMING SUMMARY

- Steps 1-4 (Harsh): 2 minutes
- Step 5 (All): 12-14 minutes
  - DeepAR (Harsh): 4 min
  - SimpleFeedForward (Utkrisht): 3 min
  - DeepNPTS (Deepika): 3 min
  - Example notebook (All): 2-3 min
- Step 6 (Utkrisht): 2 min
- Step 7 (Deepika): 2 min
- Closing: 0.5 min

**Total: 18-20 minutes**

---

## DEMO TIPS

**Before Recording:**
- Practice for smooth 20-minute delivery
- Pre-run notebooks (saves time)
- Test audio/video quality
- Clean desktop

**During Recording:**
- Speak clearly, pace yourself
- Zoom in on important outputs
- Point with cursor to highlight
- Explain outputs, don't just read

**Hand-offs:**
- Harsh → Utkrisht: "Now Utkrisht will demonstrate SimpleFeedForward"
- Utkrisht → Deepika: "Deepika will show DeepNPTS"
- After example notebook: "Let's interpret results - Utkrisht?"
- Before documentation: "Finally, our documentation - Deepika?"

Good luck! 🎓
