# COVID-19 Case Prediction with GluonTS - Demo Script

**Total Time: 15-20 minutes**
**Presenters: Harsh, Utkrisht, Deepika**

---

## HARSH - Steps 1-4 (2 minutes)

### Step 1: Introduction (20 seconds)

"Hello! I'm Harsh Yadav, UID [your UID]. We're presenting **COVID-19 Case Prediction using GluonTS** - a probabilistic time series forecasting library. We chose **Hard difficulty**. Our project forecasts COVID-19 cases 14 days ahead using three models: DeepAR, SimpleFeedForward, and DeepNPTS."

---

### Step 2: File Structure & Naming Conventions (30 seconds)

**[Show file browser]:**

"Our project follows all required naming conventions:

**Notebooks:**
- `GluonTS.API.ipynb` - API demonstrations
- `GluonTS.example.ipynb` - Complete application

**Documentation:**
- `README.md` - Project overview and setup
- `GluonTS.API.md` - Tool reference guide
- `GluonTS.example.md` - Project documentation

**Utility modules** following `{Project}_utils_{purpose}.py`:
- `GluonTS_utils_data_io.py` - Raw data loading
- `GluonTS_utils_preprocessing.py` - Data aggregation and merging
- `GluonTS_utils_gluonts.py` - GluonTS format conversion
- `GluonTS_utils_evaluation.py` - Metrics and visualization
- `GluonTS_utils_notebook_loader.py` - Simplified data loading interface
- `GluonTS_utils_models.py` - Model training wrappers
- `GluonTS_utils_data_download.py` - Automatic data download

**Docker setup:**
- `Dockerfile` - Container specification
- `docker_build.sh`, `docker_jupyter.sh`, `docker_bash.sh` - Helper scripts
- `requirements.txt` - Python dependencies

**Data directory:**
- `data/` - CSV files (auto-downloaded from Google Drive if missing)"

---

### Step 3: Docker Execution (30 seconds)

**[Terminal]:**

```bash
./docker_build.sh
```

"Building Docker image with Python 3.10, PyTorch, and GluonTS. Installs all dependencies from requirements.txt for a reproducible environment."

**[After build completes]:**

```bash
./docker_jupyter.sh
```

"Starting container:
- Port 8888 mapped to localhost
- Workspace mounted to /workspace
- MPS fallback enabled for Apple Silicon compatibility
- Jupyter server ready at localhost:8888"

---

### Step 4: Open Notebooks (40 seconds)

**[Browser at localhost:8888]:**

"Two main notebooks demonstrate our work:

**GluonTS.API.ipynb:**
- Shows how to use each GluonTS model
- Demonstrates configuration, training, and forecasting
- Educational focus on the tool itself
- Deepika will walk through this

**GluonTS.example.ipynb:**
- Complete COVID-19 forecasting application
- End-to-end workflow from data to decisions
- Demonstrates real-world problem solving
- Utkrisht will walk through this

Let me hand it over to Deepika to demonstrate the API notebook."

---

## DEEPIKA - Step 5A: API Notebook Walkthrough (6-7 minutes)

**[Opens GluonTS.API.ipynb]:**

### Introduction (30 seconds)

"This notebook teaches how to use GluonTS for time series forecasting. We're using COVID-19 data because it has:
- Complex patterns (multiple waves)
- Seasonality (weekly reporting cycles)
- External factors (mobility, deaths)
- Real-world importance

Let me run through the setup and then demonstrate each model."

---

### Setup & Data Loading (45 seconds)

**[Run Cells 1-5]:**

"Cell 3 imports:
- GluonTS models: DeepAR, SimpleFeedForward, DeepNPTS
- Our utility functions for data and evaluation

Cell 5 loads COVID-19 data:
- Automatically downloads from Google Drive if missing
- Loads cases, deaths, and mobility data
- Preprocesses to national level with 7-day moving averages
- Splits into training (3+ years) and testing (14 days)
- Converts to GluonTS format
- Extracts 3 features: deaths metrics and case fatality rate"

**[Run Cell 7 - Visualization]:**

"This shows the full US COVID timeline:
- Multiple distinct waves visible
- Weekly seasonality in the pattern
- Red line marks forecast starting point
- We're predicting the orange section ahead"

---

### Part 1: DeepAR (1.5 minutes)

**[Scroll to DeepAR section]:**

"DeepAR uses recurrent neural networks with memory - learns from past patterns to predict future.

**[Run Cell 10 - Configuration]:**

Key parameters:
- `freq='D'`: Daily data
- `prediction_length=14`: Forecast 2 weeks ahead
- `context_length=60`: Use 2 months of history
- `num_feat_dynamic_real=3`: Include external features
- Network: 2 RNN layers with 40 units
- Training: 10 epochs for speed

**[Run Cell 12 - Training]:**

Training with PyTorch Lightning backend. Loss decreases as model learns COVID patterns. Takes a few minutes.

**[Run Cell 14 - Forecasts]:**

Generates probabilistic forecasts with 100 samples. Output shows mean prediction and confidence intervals.

**[Run Cell 16 - Visualization]:**

Forecast plot shows:
- Mean prediction line
- Confidence bands (80% and 90%)
- Actual future values
- DeepAR captures the trend with appropriate uncertainty

**[Run Cell 18 - Evaluation]:**

Metrics show forecast accuracy:
- MAE: Average error in number of cases
- RMSE: Penalizes large errors
- MAPE: Percentage error

DeepAR provides good baseline for complex COVID patterns."

---

### Part 2: SimpleFeedForward (1.5 minutes)

**[Scroll to SimpleFeedForward section]:**

"SimpleFeedForward is simpler - direct mapping from history to future, no RNN memory.

**[Run Cell 21 - Configuration]:**

Differences from DeepAR:
- No `freq` parameter
- No external features
- Just uses historical cases
- Simpler architecture: 2 feedforward layers
- Trains much faster

**[Run Cell 23 - Training]:**

Watch the speed - completes in seconds versus minutes for DeepAR. That's the key advantage.

**[Run Cell 25 - Forecasts]:**

Similar probabilistic output but without feature integration.

**[Run Cell 27 - Visualization]:**

Smoother predictions:
- Doesn't capture weekly seasonality as well
- Good general trend
- Wider uncertainty bounds

**[Run Cell 29 - Evaluation]:**

Performance comparable to DeepAR but trained 10x faster. Excellent for quick baselines and experiments."

---

### Part 3: DeepNPTS (1.5 minutes)

**[Scroll to DeepNPTS section]:**

"DeepNPTS is non-parametric - learns the data distribution without rigid assumptions. Perfect for COVID where each wave behaves differently.

**[Run Cell 37 - Configuration]:**

Similar to DeepAR but:
- `epochs=10` passed directly (not via trainer_kwargs)
- `num_hidden_nodes=[40,40]` (not hidden_size)
- Still uses external features

**[Run Cell 39 - Training]:**

Learns distribution shape, not just the mean. Moderate training time.

**[Run Cell 41 - Forecasts]:**

Often shows different predictions than DeepAR - adapts to recent patterns differently.

**[Run Cell 43 - Visualization]:**

May have wider confidence intervals - honest about uncertainty during distribution changes.

**[Run Cell 45 - Evaluation]:**

Performance depends on whether test period has regime shifts. Excels during transitions."

---

### API Notebook Summary (30 seconds)

"Key takeaways from API notebook:

**DeepAR:**
- Best for complex patterns and seasonality
- Slower training but comprehensive
- Use when accuracy is critical

**SimpleFeedForward:**
- Best for speed (10x faster)
- Good baseline performance
- Use for quick experiments

**DeepNPTS:**
- Best for changing distributions
- Adapts to regime shifts
- Use during volatile periods

All three provide probabilistic forecasts with uncertainty quantification.

Now Utkrisht will demonstrate the complete application in the example notebook."

---

## UTKRISHT - Step 5B: Example Notebook Walkthrough (6-7 minutes)

**[Opens GluonTS.example.ipynb]:**

### Introduction & Problem Statement (45 seconds)

"This notebook demonstrates a complete COVID-19 forecasting application.

**Real-world problem:**
Hospitals need to predict case surges 14 days ahead to:
- Allocate ICU beds and ventilators
- Schedule healthcare staff
- Plan intervention strategies
- Communicate risk to public

**Our solution:**
End-to-end forecasting system using all three GluonTS models with comprehensive evaluation and scenario analysis.

Let me walk through the workflow."

---

### Data Pipeline (1 minute)

**[Run Cells 1-3]:**

"Setup imports all models and utilities.

**[Run Cell 4 - Load & Explore Data]:**

Data loading:
- Same automated pipeline as API notebook
- Cases, deaths, mobility merged to national level
- 7-day moving averages applied
- Training/test split created

**[Show data visualization]:**

Timeline shows:
- Original 2020 surge
- Delta variant peak
- Omicron wave
- Subsequent patterns
- Features: deaths trend and mobility changes correlate with cases"

---

### Feature Engineering (45 seconds)

**[Scroll through feature cells]:**

"Key features created:
- `Daily_Cases_MA7`: Target variable (smoothed for training)
- `Daily_Deaths_MA7`: Leading indicator
- `Cumulative_Deaths`: Overall severity measure
- `CFR`: Case fatality rate (healthcare strain)
- Mobility metrics: Behavioral response indicators

Each feature chosen to improve forecast accuracy."

---

### Model Training (1 minute)

**[Run training cells for all three models]:**

"Training all three models with consistent configuration:
- Context: 60 days
- Prediction: 14 days
- Features: 3 (deaths and CFR metrics)

**DeepAR:**
Training with RNN architecture. Learns complex temporal patterns.

**SimpleFeedForward:**
Fast baseline training. Direct pattern mapping.

**DeepNPTS:**
Non-parametric learning. Adapts to distribution changes.

All models complete training in under 10 minutes total on CPU."

---

### Model Evaluation & Comparison (1.5 minutes)

**[Run evaluation cells]:**

"Each model evaluated with multiple metrics:

**Accuracy metrics:**
- MAE: Average prediction error
- RMSE: Penalizes large errors
- MAPE: Scale-independent percentage

**Probabilistic metric:**
- CRPS: Evaluates full forecast distribution

**[Show comparison table/visualization]:**

Side-by-side comparison shows:
- Performance varies by test period
- All models achieve reasonable accuracy for 14-day COVID forecasts
- Trade-offs between accuracy, speed, and adaptability

**[Show forecast visualizations]:**

Visual comparison:
- Different models capture trends differently
- Confidence intervals vary in width
- Actual values help validate predictions

Model selection depends on your priority:
- Need accuracy? Consider DeepAR or DeepNPTS
- Need speed? Use SimpleFeedForward
- Volatile period? Try DeepNPTS"

---

### Scenario Analysis Preview (30 seconds)

**[Scroll to scenario analysis section]:**

"The most powerful feature - simulating public health interventions.

We test three scenarios:
1. **Baseline**: No intervention, current trends continue
2. **Moderate intervention**: Reduce mobility (mask mandates, capacity limits)
3. **Strong intervention**: Significant mobility reduction (lockdowns, closures)

Each scenario generates different forecast trajectories by adjusting mobility features.

This directly supports policy decision-making. I'll explain the results in Step 6."

---

## UTKRISHT - Step 6: Results Discussion (3-4 minutes)

### Scenario Analysis Results (1.5 minutes)

**[Return to scenario analysis section, show outputs]:**

"Looking at the three scenarios:

**Baseline (No Intervention):**
- Projection shows expected case trajectory
- Assumes current behaviors continue
- Establishes reference point

**Moderate Intervention:**
- Simulates 20% mobility reduction
- Cases decrease compared to baseline
- Represents policies like mask mandates, capacity limits

**Strong Intervention:**
- Simulates 40% mobility reduction  
- Significant case decrease
- Represents lockdowns, school closures

**Key insight:** Interventions show measurable impact on case projections. The model quantifies what previously was qualitative - 'stronger interventions reduce cases' becomes 'this intervention reduces cases by this amount.'"

---

## HARSH - Step 7: Documentation Review (3-4 minutes)

**[Return to file browser]:**

"Now I'll show how our documentation is organized for both technical and non-technical readers.

We have three main documentation files, each serving different audiences."

---

### README.md - Project Entry Point (1 minute)

**[Open README.md, scroll through]:**

"README.md is the first thing anyone sees. Organized for immediate understanding:

**Quick Start Section:**
- Clear 3-step process: Build Docker, Run Jupyter, Open notebooks
- Takes 5 minutes from clone to running
- Terminal commands provided
- Expected outputs shown

**Visual Documentation:**
- Mermaid diagrams show data flow and project structure
- **Non-technical readers** can see the workflow without reading code
- Arrows show how data moves from sources → preprocessing → models → forecasts

**Project Overview:**
- Brief problem statement
- What we're solving and why it matters
- Target audience identified

**Data Setup:**
- Explains automatic download from Google Drive
- Manual instructions if download fails
- Direct links to each required file
- Transparent about data sources

**Expected Outputs:**
- Shows what successful Docker build looks like
- Shows Jupyter startup messages
- Eliminates confusion - users know what to expect

**Who benefits:**
- **Project managers:** Understand deliverables without technical depth
- **Developers:** Quick setup instructions
- **Stakeholders:** Visual workflow shows system architecture
- **Students:** Easy to replicate and learn from

Visual aids make this accessible to non-technical readers - they can understand the project flow without understanding Python code."

---

### GluonTS.API.md - Tool Reference (1 minute)

**[Open GluonTS.API.md, scroll through sections]:**

"API.md is tool-focused documentation - explains GluonTS, NOT our COVID project.

**Model Overview:**
- What each model does in plain English
- DeepAR: 'Uses recurrent networks with memory'
- SimpleFeedForward: 'Direct mapping, fast and simple'
- DeepNPTS: 'Learns distribution without assumptions'

**When to Use Each Model:**
- Decision guide based on data characteristics
- 'Complex patterns → DeepAR'
- 'Need speed → SimpleFeedForward'
- 'Regime changes → DeepNPTS'

**Parameter Reference:**
- Every parameter explained
- What it controls and typical values
- Examples of how changing it affects results

**Basic Usage Pattern:**
- Step-by-step code examples
- Prepare data → Configure → Train → Forecast → Interpret
- Generic examples - no COVID specifics

**Troubleshooting:**
- Common errors and solutions
- 'Training too slow' → solutions provided
- 'Wrong features' → how to diagnose and fix

**Key point:** This is generic. A **non-technical reader** learning about time series forecasting can understand:
- What each model type does conceptually
- When to choose one over another
- Without needing to understand implementation details

Someone could use this to apply GluonTS to sales forecasting, traffic prediction, or any time series problem."

---

### GluonTS.example.md - Project Documentation (1 minute)

**[Open GluonTS.example.md, scroll through]:**

"Example.md is project-focused - everything specific to COVID-19 forecasting.

**Project Overview:**
- Clear problem statement: 'Hospitals need 14-day forecasts'
- Why it matters: Resource allocation, staffing, interventions
- Real-world context and stakeholders

**Data Sources:**
- What data we use: Cases from JHU, Deaths from JHU, Mobility from Google
- Why each dataset matters for forecasting
- How they correlate with case trends

**Feature Engineering:**
- Why 7-day moving average: 'Smooths weekend reporting artifacts'
- Why CFR: 'Indicates healthcare system strain'
- Why mobility: 'Captures behavioral response to pandemic'
- Technical decisions explained in domain terms

**Model Selection for COVID-19:**
- Why these three models for pandemic data
- DeepAR: 'COVID has multiple complex waves with seasonality'
- DeepNPTS: 'Each variant behaves differently, need flexibility'
- Domain-specific rationale

**Results Interpretation:**
- What metrics mean practically
- 'MAE tells hospitals average error for bed planning'
- 'Confidence intervals show range to prepare for'
- Numbers translated to operational decisions

**Scenario Analysis:**
- How interventions affect forecasts
- Policy implications
- Decision-making framework

**Key point:** A **non-technical hospital administrator** can read this and understand:
- Why this forecasting system helps their work
- What the predictions mean for planning
- How to use scenario analysis for decisions
- No code knowledge required - focus on application

The documentation translates technical implementation into domain value."

---

### How Non-Technical Readers Navigate (45 seconds)

"Let me show how a non-technical person uses our documentation:

**Hospital Administrator scenario:**

**Step 1 - README.md:**
- Sees Mermaid diagram: 'Oh, data comes in, gets processed, models make predictions'
- Reads problem statement: 'This solves our resource planning problem'
- Looks at visual workflow: Understands system without code

**Step 2 - GluonTS.example.md:**
- Reads project overview: 'We need 14-day forecasts - this provides exactly that'
- Sees feature explanations: 'They're using deaths and mobility - that makes sense'
- Reviews scenario analysis: 'We can test intervention policies - very useful'
- Checks results interpretation: 'Confidence intervals help us plan for uncertainty'

**Step 3 - View visualizations in notebook:**
- Opens example.ipynb (no code execution needed)
- Sees forecast plots: 'Visual representation of predictions'
- Sees scenario comparisons: 'Clear impact of different policies'

**Result:** They understand:
- What the system does
- How it helps their work
- What forecasts mean operationally
- How to interpret uncertainty
- How to use scenario analysis

**Never needed to:**
- Read Python code
- Understand model mathematics
- Know machine learning concepts
- Understand GluonTS API details

The three-layer documentation structure (README → Example → API) lets different audiences find what they need without drowning in unnecessary details."

---

### Documentation Quality Summary (30 seconds)

"Three key qualities make our documentation effective:

**1. Completeness:**
- Every component documented
- Setup, data, models, evaluation, deployment all covered
- No black boxes - transparent throughout

**2. Clarity:**
- Plain language throughout
- Technical terms explained when introduced
- Visual aids supplement text
- Progressive complexity - simple concepts first

**3. Accessibility:**
- Multiple entry points for different audiences
- Technical readers: Go deep with API.md and code
- Non-technical readers: Stay with README and example.md
- Everyone gets what they need

This ensures our project is usable by hospitals, public health departments, policy makers, and future developers - not just data scientists."

---

## CLOSING (All 3) - 30 seconds

**[Harsh concludes]:**

"Thank you for watching our demonstration.

**What we delivered:**
- Complete COVID-19 forecasting system with three GluonTS models
- Probabilistic predictions with uncertainty quantification
- Scenario analysis for intervention planning
- Production-ready Docker deployment
- Comprehensive documentation for all audiences
- Modular architecture for easy extension

**Key achievement:**
GluonTS enabled sophisticated time series forecasting that directly supports real-world public health decision-making - from hospital resource planning to policy evaluation.

Our hard difficulty project demonstrates advanced probabilistic modeling, comprehensive uncertainty quantification, scenario-based analysis, and professional deployment practices.

Questions?"

---

## TIMING SUMMARY

- **Harsh - Steps 1-4:** 2 minutes
- **Deepika - Step 5A (API Notebook):** 6-7 minutes
- **Utkrisht - Step 5B (Example Notebook):** 6-7 minutes
- **Utkrisht - Step 6 (Results):** 3-4 minutes
- **Harsh - Step 7 (Documentation):** 3-4 minutes
- **Closing:** 0.5 minutes

**Total: 20-25 minutes**

---

## PRESENTATION TIPS

**General:**
- Speak clearly and maintain good pace
- Use cursor to highlight important information
- Explain what you're showing, don't just read
- Smooth transitions between presenters

**Hand-offs:**
- Harsh → Deepika: "Deepika will demonstrate the API notebook"
- Deepika → Utkrisht: "Utkrisht will show the complete application"
- Utkrisht → Harsh (after Step 6): "Harsh will review our documentation"

**If something fails:**
- Pre-run notebooks and show outputs
- If download fails, explain fallback to manual download
- Stay calm, explain the issue and solution

Good luck! 🎓
