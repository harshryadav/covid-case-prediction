# Predicting the Unpredictable: COVID-19 Case Forecasting with GluonTS

**A Hands-On Guide to Probabilistic Time Series Forecasting**

*Reading time: 10-15 minutes*

---

## The Challenge: When Tomorrow's Numbers Matter

Picture this: It's March 2020. Dr. Sarah Chen, an emergency room director at a major hospital, stares at her computer screen. Cases are rising, but by how much? Should she cancel elective surgeries next week? Order more ventilators? Call in extra staff?

She doesn't need to know the exact number of cases two weeks from now. What she *needs* is a reliable range—a forecast that says, "Prepare for somewhere between 500 and 800 new cases, but it could spike to 1,200 if nothing changes."

This is the world of **probabilistic forecasting**—and it's exactly what we'll learn to build in this tutorial.

### Why Pandemic Forecasting is Hard

COVID-19 didn't follow a simple pattern. Looking back at the U.S. case data from 2020-2023, we see:

- **Multiple waves**: The original strain, Delta, Omicron—each behaved differently
- **Weekly cycles**: Cases dropped every weekend (fewer tests, delayed reporting)
- **Human behavior**: Lockdowns worked, reopenings sparked surges
- **Variants**: Just when models learned one pattern, a new variant changed everything

Traditional forecasting methods that work for sales or weather often fail here. They assume patterns repeat predictably. Pandemics don't.

### What We'll Build

By the end of this tutorial, you'll have:

1. **A working forecasting system** that predicts COVID-19 cases 14 days ahead
2. **Three different models** to compare (each with its strengths)
3. **Uncertainty estimates** telling you the range of possible outcomes
4. **Scenario analysis** showing what happens under different intervention policies

All of this runs in a Docker container—no complex setup, no dependency headaches.

---

## Meet GluonTS: Your Forecasting Toolkit

### What is GluonTS?

GluonTS is a Python library developed by Amazon Web Services (AWS) for **probabilistic time series forecasting**. Think of it as a Swiss Army knife for prediction problems where:

- You have data that changes over time (cases per day, sales per week, temperature per hour)
- You need to predict future values
- You want to know *how confident* you should be in those predictions

The "probabilistic" part is key. Instead of saying "there will be 1,000 cases tomorrow," GluonTS says "there's a 90% chance cases will be between 800 and 1,400, with 1,000 being most likely."

For a hospital administrator, that range is gold. It's the difference between being caught off guard and being prepared.

### Why Not Just Use [Insert Other Tool Here]?

You might wonder: why GluonTS when there's Prophet, ARIMA, or plain PyTorch?

| Tool | Strength | Limitation |
|------|----------|------------|
| **ARIMA** | Well-understood, interpretable | Struggles with complex patterns |
| **Prophet** | Easy to use, handles seasonality | Less flexible for deep learning |
| **Raw PyTorch** | Maximum flexibility | You build everything from scratch |
| **GluonTS** | Pre-built models + probabilistic output + external features | Steeper learning curve (but that's why you're here!) |

GluonTS shines when you have:
- Complex, multi-pattern data (like COVID-19)
- External factors that influence your target (mobility, deaths, policy changes)
- A need for uncertainty quantification (not just point forecasts)

---

## The Three Musketeers: Our Forecasting Models

In this tutorial, we use three GluonTS models. Each approaches forecasting differently, like three colleagues with different problem-solving styles.

### 1. DeepAR: The Historian with a Memory

**Analogy**: Imagine a historian who reads through years of records, noting patterns like "cases rose after every holiday" or "this wave looks like the one from last year." DeepAR uses **Recurrent Neural Networks (RNNs)** to remember long sequences and learn temporal dependencies.

**Best for**:
- Complex patterns with seasonality
- When you have external features (like mobility data)
- Maximum accuracy (if you have time to train)

**Trade-off**: Slower to train (3-4 minutes on CPU)

### 2. SimpleFeedForward: The Quick Estimator

**Analogy**: Think of a colleague who glances at the last few weeks of data and makes a quick estimate. They don't analyze deep history—just recent trends. That's SimpleFeedForward: a straightforward neural network that maps recent context to future predictions.

**Best for**:
- Fast baselines (trains in under a minute)
- Stable, predictable trends
- Quick experiments and prototyping

**Trade-off**: Doesn't use external features, may miss complex patterns

### 3. DeepNPTS: The Pattern Spotter

**Analogy**: Imagine someone who doesn't assume COVID-19 follows any particular distribution. They just observe: "During Omicron, cases behaved *this* way; during Delta, they behaved *that* way." DeepNPTS is **non-parametric**—it learns the data's distribution without forcing it into a predetermined shape.

**Best for**:
- Data with regime changes (new variants!)
- When standard distributions don't fit
- Unusual, non-standard patterns

**Trade-off**: May have wider uncertainty when data is volatile

### Which Model Should You Choose?

```
┌─────────────────────────────────────────────────────────┐
│                  Model Selection Guide                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Need results quickly?                                  │
│     └── YES → SimpleFeedForward                         │
│     └── NO  → Continue...                               │
│                                                         │
│  Have external features (mobility, deaths)?             │
│     └── NO  → SimpleFeedForward                         │
│     └── YES → Continue...                               │
│                                                         │
│  Expect sudden changes (new variants, policy shifts)?   │
│     └── YES → DeepNPTS                                  │
│     └── NO  → DeepAR                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

In practice, we often train all three and compare. That's exactly what this tutorial does.

---

## Our Data Story: Cases, Deaths, and How People Moved

### The Datasets

We use three real-world data sources, all publicly available:

#### 1. COVID-19 Cases (Johns Hopkins University)
- **What**: Daily confirmed COVID-19 cases for every U.S. county
- **Period**: January 2020 – March 2023
- **We aggregate**: County → National level
- **We calculate**: Daily new cases, 7-day moving average

The 7-day moving average is crucial—it smooths out weekend reporting artifacts where fewer tests are processed.

#### 2. COVID-19 Deaths (Johns Hopkins University)
- **What**: Daily COVID-19 deaths
- **Why useful**: Deaths lag cases by 2-3 weeks. If deaths are rising, cases were likely high recently.
- **We calculate**: Daily deaths, cumulative deaths, Case Fatality Ratio (CFR)

**CFR (Case Fatality Ratio)** = Deaths ÷ Cases. A rising CFR often signals healthcare strain.

#### 3. Google Mobility Reports
- **What**: How much people moved compared to baseline (pre-pandemic)
- **Categories tracked**:
  - 🛒 Retail & recreation
  - 🛒 Grocery & pharmacy
  - 🌳 Parks
  - 🚇 Transit stations
  - 🏢 Workplaces
  - 🏠 Residential

**Why mobility matters**: When people stay home, transmission drops. When mobility increases, cases often follow (with a lag).

### What the Data Looks Like

Here's a simplified view of our combined dataset:

| Date | Daily_Cases_MA7 | Daily_Deaths_MA7 | CFR | Retail_Mobility | ... |
|------|-----------------|------------------|-----|-----------------|-----|
| 2020-03-01 | 50 | 1 | 2.0% | -5% | ... |
| 2020-04-15 | 30,000 | 2,000 | 6.0% | -45% | ... |
| 2021-01-10 | 250,000 | 3,500 | 1.5% | -25% | ... |
| 2022-01-15 | 800,000 | 2,500 | 0.3% | -15% | ... |

Notice the patterns:
- **April 2020**: High CFR (healthcare overwhelmed), very low mobility (lockdowns)
- **January 2021**: Massive case surge (winter wave), moderate mobility
- **January 2022**: Omicron—record cases, but lower CFR (milder variant + vaccines)

These are the patterns our models learn to recognize and forecast.

### The Train/Test Split

We split our data like this:

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING DATA                             │
│                    (~1,100+ days)                                │
│     January 2020 ─────────────────────────────► February 2023    │
└──────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                              ┌────────────────────┐
                                              │    TEST DATA       │
                                              │    (14 days)       │
                                              │  Feb 24 - Mar 9    │
                                              │      2023          │
                                              └────────────────────┘
```

The models learn from 3+ years of history, then we test: "Can you predict the next 2 weeks?"

---

## The Forecast Journey: From Raw Data to Predictions

### Step 1: Data Loading and Preprocessing

Our pipeline handles this automatically, but here's what happens:

```
Raw CSV Files
     │
     ▼
┌─────────────────┐
│ Load cases.csv  │──┐
│ Load deaths.csv │  │
│ Load mobility   │  │
└─────────────────┘  │
                     ▼
          ┌─────────────────────┐
          │ Aggregate to        │
          │ National Level      │
          │ (sum all counties)  │
          └─────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Calculate Features  │
          │ - 7-day MA          │
          │ - CFR               │
          │ - Merge mobility    │
          └─────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Convert to GluonTS  │
          │ ListDataset format  │
          └─────────────────────┘
                     │
                     ▼
              Ready to Train!
```

### Step 2: Training the Models

Each model is configured with:
- **Prediction length**: 14 days (our forecast horizon)
- **Context length**: 60 days (how much history the model sees)
- **Features**: Deaths data and CFR (for models that support them)

Training looks like this:

```python
# DeepAR with external features
estimator = DeepAREstimator(
    freq='D',                    # Daily data
    prediction_length=14,        # Forecast 2 weeks ahead
    context_length=60,           # Use 2 months of history
    num_feat_dynamic_real=3,     # 3 external features
    num_layers=2,                # RNN depth
    hidden_size=40,              # Network capacity
    trainer_kwargs={"max_epochs": 10}
)

predictor = estimator.train(train_dataset)
```

The model sees thousands of examples of "given 60 days of history, what happened in the next 14 days?" and learns the patterns.

### Step 3: Generating Probabilistic Forecasts

This is where the magic happens. Instead of one prediction, we get a *distribution*:

```python
# Generate 100 possible forecast trajectories
forecasts = predictor.predict(test_dataset, num_samples=100)
```

Each of those 100 samples is a plausible future. From them, we calculate:
- **Mean**: The average prediction
- **Median**: The middle prediction
- **Quantiles**: "There's a 90% chance cases will be below X"

### Step 4: Evaluating the Forecasts

We measure how well our models did using:

| Metric | What It Measures | Lower is Better? |
|--------|------------------|------------------|
| **MAE** (Mean Absolute Error) | Average prediction error | ✅ Yes |
| **RMSE** (Root Mean Square Error) | Penalizes large errors more | ✅ Yes |
| **MAPE** (Mean Absolute % Error) | Error as percentage (scale-free) | ✅ Yes |
| **CRPS** | How well uncertainty is calibrated | ✅ Yes |

A model with low MAPE but poorly calibrated uncertainty might be overconfident—it says "I'm 90% sure" but is wrong 30% of the time. CRPS catches this.

### Sample Results

When we run our models on COVID-19 data, we typically see:

```
Model Comparison Results
══════════════════════════════════════════════════════════════
  Rank   Model              MAPE    RMSE       MAE      Time
──────────────────────────────────────────────────────────────
   1     DeepAR             5.2%    3,200      2,500    180s
   2     DeepNPTS           5.7%    3,500      2,700    150s
   3     SimpleFeedForward  6.5%    4,000      3,100    30s
══════════════════════════════════════════════════════════════
```

DeepAR often wins on accuracy, but SimpleFeedForward is 6x faster. The "best" model depends on your constraints.

---

## Scenario Analysis: What If We Had Done Differently?

This is where forecasting becomes *actionable*.

### The Question

Policymakers in 2020-2022 faced impossible choices:
- "If we mandate masks, how many cases will we prevent?"
- "If we reopen schools, what's the risk?"
- "What happens if we do nothing?"

Our scenario analysis simulates these questions by modifying the **mobility features** and re-running forecasts.

### The Scenarios

We test five scenarios:

| Scenario | Description | Mobility Change | What It Simulates |
|----------|-------------|-----------------|-------------------|
| **Baseline** | No intervention | 0% | Status quo continues |
| **Moderate** | Mild restrictions | -15% | Mask mandates, capacity limits |
| **Strong** | Significant restrictions | -30% | Lockdowns, closures |
| **Relaxation** | Restrictions lifted | +20% | Reopening, holidays |
| **Healthcare Strain** | Elevated CFR | CFR +15% | Hospitals overwhelmed |

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      SCENARIO ANALYSIS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Original Data ──► Trained Model ──► Baseline Forecast         │
│                           │                                     │
│   Modified Data           │                                     │
│   (mobility -30%) ────────┴──────► Intervention Forecast        │
│                                                                 │
│   Compare: How many cases were prevented?                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sample Results

```
Scenario Comparison (14-day forecast)
══════════════════════════════════════════════════════════════════
  Scenario              Avg Daily Cases    Total Cases    vs Baseline
──────────────────────────────────────────────────────────────────────
  Baseline              47,000             658,000        --
  Moderate (-15%)       42,000             588,000        -10.6%
  Strong (-30%)         37,000             518,000        -21.3%
  Relaxation (+20%)     54,000             756,000        +14.9%
  Healthcare Strain     47,000             658,000        (same cases)
══════════════════════════════════════════════════════════════════════
```

**Key Insight**: A strong intervention (30% mobility reduction) could prevent ~140,000 cases over two weeks—but at significant economic and social cost. Policymakers must weigh these tradeoffs.

### Important Caveats

Scenario analysis is powerful but has limitations:

1. **Correlation ≠ Causation**: Just because mobility correlates with cases doesn't mean reducing mobility will reduce cases by exactly that amount
2. **Model Assumptions**: We assume the relationship learned from history holds in the future
3. **Simplification**: Real interventions are complex—a "30% mobility reduction" is a simplified proxy

Think of scenarios as *illustrative*, not *prescriptive*. They inform discussion, not dictate policy.

---

## Try It Yourself: 5 Minutes to Your First Forecast

Ready to run the code? Here's how to get started.

### Prerequisites

- **Docker** installed and running (that's it!)
- ~8GB RAM available
- ~15 minutes for first run (Docker build + model training)

### Quick Start

```bash
# 1. Clone the repository (or navigate to the folder)
cd TutorTask121_GluonTS_COVID_19_Case_Prediction

# 2. Build the Docker image
./docker_build.sh

# 3. Start Jupyter Notebook
./docker_jupyter.sh

# 4. Open your browser to http://localhost:8888
```

### What to Explore

| Resource | Purpose | Time |
|----------|---------|------|
| `GluonTS.API.ipynb` | Learn model APIs with guided examples | 20 min |
| `GluonTS.example.ipynb` | Complete end-to-end forecasting application | 25 min |
| `GluonTS.API.md` | Reference guide for GluonTS parameters | 10 min |
| `GluonTS.example.md` | Detailed explanation of the example notebook | 10 min |

### Suggested Learning Path

```
┌─────────────────────────────────────────────────────────────────┐
│                    60-MINUTE LEARNING PATH                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   0-5 min    Setup Docker, verify it works                      │
│   5-15 min   Read this blog post (you're almost done!)          │
│   15-35 min  Work through GluonTS.API.ipynb                     │
│   35-60 min  Run GluonTS.example.ipynb                          │
│                                                                 │
│   After 60 minutes, you'll have:                                │
│   ✓ Trained 3 models                                            │
│   ✓ Generated probabilistic forecasts                           │
│   ✓ Compared model performance                                  │
│   ✓ Run scenario analysis                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## What's Next: Taking It Further

Once you've mastered the basics, consider these extensions:

### Immediate Improvements
- **More features**: Add vaccination data (available in `data/vaccine.csv`)
- **Longer horizons**: Try 28-day forecasts instead of 14
- **State-level**: Adapt the code for individual state forecasting

### Advanced Techniques
- **Ensemble models**: Combine predictions from all three models
- **Hyperparameter tuning**: Optimize context_length, hidden_size, etc.
- **Real-time updates**: Automate daily data fetching and retraining

### Apply to Other Domains
The same techniques work for:
- 📈 Sales forecasting
- 🚗 Traffic prediction
- 🌡️ Energy demand forecasting
- 📊 Financial time series

GluonTS is domain-agnostic—once you understand the pattern, you can forecast almost anything.

---

## Key Takeaways

1. **Probabilistic > Point forecasts**: Knowing the range of possible outcomes is more valuable than a single number

2. **External features matter**: Mobility data significantly improves COVID-19 case predictions

3. **No single best model**: DeepAR wins on accuracy, SimpleFeedForward wins on speed, DeepNPTS handles regime changes

4. **Scenario analysis enables decisions**: Quantifying "what if" helps policymakers weigh tradeoffs

5. **Docker makes reproducibility easy**: Anyone can run this code and get the same results

---

## References and Further Reading

### GluonTS Resources
- [GluonTS Official Documentation](https://ts.gluon.ai/)
- [GluonTS GitHub Repository](https://github.com/awslabs/gluonts)
- [GluonTS Research Paper (arXiv)](https://arxiv.org/abs/1906.05264)

### Model Papers
- [DeepAR Paper](https://arxiv.org/abs/1704.04110) - Salinas et al., 2019
- [Deep Non-Parametric Time Series (DeepNPTS)](https://arxiv.org/abs/1906.05264)

### COVID-19 Data Sources
- [Johns Hopkins COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)
- [Google COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility/)
- [CDC COVID Data Tracker](https://covid.cdc.gov/covid-data-tracker/)

### Time Series Forecasting (General)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) - Hyndman & Athanasopoulos (free online textbook)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

---

## About This Tutorial

This tutorial is part of the **"Learn X in 60 Minutes"** series, designed to give you hands-on experience with data science tools in under an hour.

**What makes this tutorial different**:
- Everything runs in Docker (no dependency hell)
- Real-world data, not toy examples
- Focus on understanding, not just code copying
- Probabilistic forecasting, not just point predictions

**Feedback?** We'd love to hear how we can improve. Open an issue or submit a PR!

---

*Happy forecasting!* 🚀📈
