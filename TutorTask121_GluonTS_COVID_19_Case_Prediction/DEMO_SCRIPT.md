# COVID-19 Case Prediction with GluonTS - Demo Script

**Total Time: 10-20 minutes**
**Presenters: Harsh, Utkrisht, Deepika**

---

## HARSH - Steps 1-4 (2-3 minutes)

### Step 1: Introduction (30 seconds)

**[Harsh speaks]:**

"Hello everyone! My name is Harsh Yadav, UID [your UID].

Today we're presenting our project on **COVID-19 Case Prediction using GluonTS**, a probabilistic time series forecasting library.

We chose **Hard difficulty** for this project.

Our project title is: **Time Series Forecasting for COVID-19 Case Prediction using GluonTS with PyTorch Backend**

We'll be demonstrating how GluonTS can forecast COVID-19 cases 14 days ahead using three different models: DeepAR, SimpleFeedForward, and DeepNPTS."

---

### Step 2: Showcase Files and Naming Conventions (30 seconds)

**[Harsh shares screen showing project directory]:**

**[Harsh speaks while showing file listing]:**

"Let me show you our project structure. All files follow the required naming conventions:

**Main notebooks:**
- `GluonTS.API.ipynb` - API demonstration notebook
- `GluonTS.example.ipynb` - Complete application example

**Documentation:**
- `GluonTS.API.md` - Tool documentation
- `GluonTS.example.md` - Project documentation
- `README.md` - Project overview and setup

**Utility modules** following the `{Project}_utils_{purpose}.py` pattern:
- `GluonTS_utils_data_io.py` - Data loading functions
- `GluonTS_utils_preprocessing.py` - Data preprocessing
- `GluonTS_utils_gluonts.py` - GluonTS formatting
- `GluonTS_utils_evaluation.py` - Metrics and plotting
- `GluonTS_utils_notebook_loader.py` - Simplified data loading
- `GluonTS_utils_models.py` - Model training wrappers
- `GluonTS_utils_data_download.py` - Automatic data download

**Docker files:**
- `Dockerfile` - Container specification
- `docker_build.sh` - Build script
- `docker_jupyter.sh` - Run Jupyter script
- `docker_bash.sh` - Interactive shell script
- `requirements.txt` - Python dependencies

**Data directory:**
- `data/` - Contains COVID-19 datasets (auto-downloaded if missing)

All naming conventions align with the class requirements."

---

### Step 3: Execute Docker Image (30 seconds)

**[Harsh switches to terminal]:**

**[Harsh speaks]:**

"Now let me demonstrate building and running the Docker container.

First, we build the image:"

**[Harsh types and runs]:**
```bash
./docker_build.sh
```

**[While building shows on screen, Harsh explains]:**

"This builds our Docker image with Python 3.10, PyTorch, GluonTS, and all dependencies from requirements.txt. This ensures a reproducible environment on any machine."

**[After build completes, Harsh points to screen]:**

"You can see the successful build message here. Now let's start Jupyter:"

**[Harsh types and runs]:**
```bash
./docker_jupyter.sh
```

**[Harsh points to output]:**

"The container is now running. Notice the output shows:
- Port 8888 mapped to localhost
- Jupyter server started successfully
- Access token for security
- Volume mounted to /workspace
- MPS fallback enabled for Apple Silicon compatibility

We can now access Jupyter at localhost:8888."

---

### Step 4: Open Jupyter Notebooks (30 seconds)

**[Harsh opens browser to localhost:8888]:**

**[Harsh speaks]:**

"Here's our Jupyter environment. You can see all our project files.

We have two main notebooks:
1. **GluonTS.API.ipynb** - Demonstrates the GluonTS API with each model
2. **GluonTS.example.ipynb** - Complete COVID-19 forecasting application

We'll start with the API notebook to show you how each model works, then move to the example notebook for the complete application.

Let me open **GluonTS.API.ipynb** first."

**[Opens GluonTS.API.ipynb]:**

"This notebook is organized by model - each team member will walk through their assigned model:
- Part 1: DeepAR - I'll demonstrate
- Part 2: SimpleFeedForward - Utkrisht will demonstrate
- Part 3: DeepNPTS - Deepika will demonstrate

Let's begin!"

---

## ALL 3 MEMBERS - Step 5: Full Project Walkthrough (12-15 minutes)

### PART A: API Notebook - Model Demonstrations

---

### HARSH - DeepAR (GluonTS.API.ipynb Cells 0-18) - 4-5 minutes

**[Harsh scrolls to top of API notebook]:**

#### Cell 0: Notebook Introduction

**[Harsh speaks]:**

"This is our API tutorial notebook. It teaches how to use GluonTS for time series forecasting.

We're using COVID-19 data because it has:
- Multiple waves showing complex patterns
- Weekly seasonality from reporting cycles
- External factors like mobility changes
- Real-world importance for resource planning

Let me run through the setup cells quickly."

---

#### Cells 1-5: Setup and Data Loading

**[Harsh runs Cell 3 - Imports]:**

"Cell 3 imports all necessary libraries:
- Standard libraries: pandas, numpy, matplotlib
- GluonTS models: DeepAR, SimpleFeedForward, DeepNPTS
- Our custom utilities for data loading and evaluation"

**[Output shows: "All imports successful"]:**

"Good, all imports successful."

**[Harsh runs Cell 5 - Load Data]:**

**[Harsh speaks while cell runs]:**

"Cell 5 loads our COVID-19 data using `quick_load_minimal()`. This function:
- Checks if data files exist in the data folder
- Downloads them automatically from Google Drive if missing
- Loads cases, deaths, and mobility data
- Preprocesses and aggregates to national level
- Applies 7-day moving averages
- Splits into training and testing sets
- Converts to GluonTS format"

**[Output shows data loading process]:**

"You can see the output:
- Training period: 1,123 days (January 2020 to February 2023)
- Testing period: 14 days (late February to early March 2023)
- Target variable: Daily_Cases_MA7 - the 7-day moving average of cases
- Features: 3 features - Daily_Deaths_MA7, Cumulative_Deaths, and CFR (Case Fatality Rate)

These features help predict case trends because deaths lag cases and indicate healthcare strain."

---

#### Cell 7: Visualize the Data

**[Harsh runs Cell 7]:**

**[Graph appears showing COVID-19 timeline]:**

**[Harsh speaks]:**

"This visualization shows the complete US COVID-19 timeline:
- The blue line is our training data - 3+ years of history
- The orange line is our test data - the future we're trying to predict
- The red vertical line marks 'today' - where our forecast begins

Notice the multiple distinct waves:
- The original 2020 surge
- The Delta variant peak in late 2021
- The massive Omicron wave in early 2022
- Subsequent smaller waves

You can also see weekly patterns - the jagged nature of the line shows lower reporting on weekends.

This complex pattern is exactly why we need sophisticated models like DeepAR."

---

#### Cells 8-10: DeepAR Configuration

**[Harsh scrolls to Part 1: DeepAR section]:**

**[Harsh speaks]:**

"Now let me demonstrate **DeepAR** - our first model.

DeepAR uses recurrent neural networks, which means it has 'memory' - it remembers what happened before to make better predictions.

Use DeepAR when you have:
- Complex patterns like COVID waves
- Seasonality like weekly cycles
- Long-term dependencies
- Multiple related time series"

**[Harsh runs Cell 10 - Configure DeepAR]:**

**[Harsh explains while pointing to code]:**

"Let me explain the key configuration parameters:

**Temporal parameters:**
- `freq='D'`: Daily data
- `prediction_length=14`: We're forecasting 14 days ahead (2 weeks)
- `context_length=60`: We use 60 days (2 months) of history to make predictions
  - Rule of thumb: context should be 2-4x the prediction length

**Feature configuration:**
- `num_feat_dynamic_real=3`: We're using 3 external features - deaths and CFR metrics
  - These help the model understand case trends better

**Network architecture:**
- `num_layers=2`: Two RNN layers to learn patterns
- `hidden_size=40`: 40 neurons per layer - enough capacity for COVID patterns
- `dropout_rate=0.1`: Light regularization to prevent overfitting

**Training settings:**
- `lr=0.001`: Learning rate
- `trainer_kwargs={'max_epochs': 10}`: Train for 10 epochs
  - This is fast for demos; production would use 20-30 epochs"

**[Output shows configuration summary]:**

"The output confirms: forecasting 14 days ahead using 60 days of history with 3 features."

---

#### Cells 11-12: Train DeepAR

**[Harsh runs Cell 12 - Train DeepAR]:**

**[Harsh speaks while training shows progress]:**

"Now we're training DeepAR on our COVID-19 data. 

Notice the output:
- PyTorch Lightning is managing the training under the hood
- You can see 'Epoch 0, Epoch 1, Epoch 2...' as it trains
- The 'train_loss' value is decreasing - this means the model is learning!
- Started at 11.71, now down to 11.13 - that's good progress

Training takes about 2-3 minutes on CPU. Each epoch processes 50 batches of data."

**[Training completes]:**

"Training complete! DeepAR has learned the COVID-19 patterns from 3+ years of data."

---

#### Cells 13-14: Generate Forecasts

**[Harsh runs Cell 14 - Generate Forecasts]:**

**[Harsh speaks]:**

"Now we generate forecasts. 

This is where GluonTS shines - it produces **probabilistic forecasts**, not just a single prediction.

We're generating 100 sample forecasts to quantify uncertainty. This gives us:
- Mean prediction: The average expected outcome
- Confidence intervals: The range of likely outcomes"

**[Output shows forecast summary]:**

"Look at the output:
- Mean prediction: ~49,000 cases per day
- Forecast range: 44,800 to 53,000 cases
- 90% confidence interval: 15,000 to 81,000 cases

This tells us: 'We expect around 49,000 cases, but there's significant uncertainty due to COVID's unpredictable nature.'"

---

#### Cell 16: Visualize DeepAR Predictions

**[Harsh runs Cell 16]:**

**[Graph appears showing forecast with confidence bands]:**

**[Harsh speaks while pointing to graph]:**

"This is a powerful visualization. Let me explain what you're seeing:

**Historical context (left side):**
- Blue line: Last 90 days of training data for context
- Shows the declining trend leading up to our forecast point

**The forecast (right side after red line):**
- Orange dots: Actual future values (what really happened)
- Green dashed line: DeepAR's mean prediction
- Light green shaded areas: Confidence intervals
  - Darker green: 80% confidence
  - Lighter green: 90% confidence

**What this tells us:**
- DeepAR captured the declining trend reasonably well
- The actual values (orange) mostly fall within our confidence intervals
- The uncertainty bands are wide, reflecting COVID's inherent unpredictability
- The model provides a realistic range of outcomes, not false precision"

---

#### Cell 18: Evaluate DeepAR Performance

**[Harsh runs Cell 18]:**

**[Harsh speaks]:**

"Now let's evaluate DeepAR's performance using standard metrics:

**The output shows:**
- MAE (Mean Absolute Error): 13,905 cases
  - On average, we're off by about 14,000 cases per day
  
- RMSE (Root Mean Square Error): 14,183 cases
  - Similar to MAE but penalizes large errors more
  
- MAPE (Mean Absolute Percentage Error): 40.2%
  - On average, we're off by about 40%
  
**Is this good?** 
The notebook says 'Decent baseline' - and that's fair. A 40% error might seem high, but remember:
- COVID-19 is extremely volatile
- We're forecasting 14 days ahead
- External events (new variants, policy changes) can shift patterns suddenly
- The confidence intervals captured most actual values

For real-world deployment, we'd tune hyperparameters more carefully - increase epochs to 30, adjust hidden size, optimize context length. But this demonstrates the model works!

**Key takeaway:** DeepAR learned COVID patterns, captured trends, and quantified uncertainty appropriately."

---

**[Harsh transitions]:**

"That's DeepAR! Now let me hand it over to Utkrisht, who will demonstrate SimpleFeedForward - our fast baseline model."

---

### UTKRISHT - SimpleFeedForward (GluonTS.API.ipynb Cells 19-34) - 3-4 minutes

**[Utkrisht takes over screen sharing]:**

**[Utkrisht scrolls to Part 2: SimpleFeedForward section]:**

#### Cells 19-21: SimpleFeedForward Introduction and Configuration

**[Utkrisht speaks]:**

"Thanks Harsh! Now I'll demonstrate **SimpleFeedForward** - the fast baseline model.

SimpleFeedForward is different from DeepAR:
- No recurrent connections, no memory
- Direct mapping from recent history to future
- Trains about 10x faster than DeepAR
- Simpler architecture with fewer parameters

Use SimpleFeedForward when you need:
- Fast training for quick experiments
- Testing different scenarios rapidly
- A simple baseline for comparison
- Your data has stable, predictable trends"

**[Utkrisht runs Cell 21 - Configure SimpleFeedForward]:**

**[Utkrisht explains]:**

"Let me explain the configuration:

**Key differences from DeepAR:**
- `prediction_length=14` and `context_length=60` - same as DeepAR
- `hidden_dimensions=[40, 40]` - two feedforward layers with 40 neurons each
  - This is simpler than DeepAR's RNN layers
  
**Important notes:**
- SimpleFeedForward does NOT use `freq` parameter
- SimpleFeedForward does NOT use external features (`num_feat_dynamic_real`)
- It only looks at the target variable (cases) history
- We're training for 20 epochs since it's so fast

This simplicity is both a strength (speed) and limitation (less information)."

**[Output confirms configuration]:**

"Configuration set. Notice the output mentions it doesn't use frequency or external features like DeepAR."

---

#### Cells 22-23: Train SimpleFeedForward

**[Utkrisht runs Cell 23 - Train SimpleFeedForward]:**

**[Utkrisht speaks while training runs]:**

"Watch how fast this trains compared to DeepAR!

Notice:
- Same PyTorch Lightning backend
- Same epoch structure
- But much faster - completing in about 30 seconds vs 2-3 minutes for DeepAR

The train_loss is decreasing: starting around 12.89, going down to 12.77.

This speed advantage is huge for:
- Hyperparameter tuning (try many configurations quickly)
- Production retraining (daily updates with new data)
- Rapid prototyping

And... done! See how fast that was?"

**[Training completes]:**

"SimpleFeedForward trained! About 10x faster than DeepAR."

---

#### Cells 24-25: Generate SimpleFeedForward Forecasts

**[Utkrisht runs Cell 25 - Generate Forecasts]:**

**[Utkrisht speaks]:**

"Now generating forecasts the same way - 100 samples for uncertainty quantification.

**Output shows:**
- Mean prediction: ~48,000 cases per day
- Forecast range: 44,000 to 51,000 cases
- 90% confidence interval: wider than DeepAR (more uncertainty)

Interesting comparison to DeepAR:
- Similar mean predictions (~48k vs ~49k)
- Slightly narrower forecast range
- But larger confidence intervals - less certain"

---

#### Cell 27: Visualize SimpleFeedForward Predictions

**[Utkrisht runs Cell 27]:**

**[Graph appears]:**

**[Utkrisht speaks while pointing to visualization]:**

"Here's SimpleFeedForward's forecast visualization:

**Comparing to DeepAR's graph:**
- Purple dashed line: SimpleFeedForward's mean prediction
- Purple shaded areas: Confidence intervals
- Orange dots: Still the actual values

**Key observations:**
1. **Smoother predictions:** Notice SimpleFeedForward's forecast is smoother than DeepAR
   - It doesn't capture the weekly seasonality (the jagged pattern)
   - It extrapolates a smooth trend
   
2. **Similar trend direction:** It got the general declining direction right

3. **Wider uncertainty at the end:** The confidence bands widen more as we go further out
   - This is realistic - uncertainty increases with time

4. **Some actual values outside confidence bands:** A few orange dots escape the purple shaded area
   - This suggests the model might be slightly underestimating uncertainty

**Why the differences?**
- No RNN memory → can't learn weekly patterns as well
- No external features → missing death/mobility signals
- Simpler architecture → less capacity for complex patterns

**But remember:** This trained 10x faster! That's the speed-accuracy tradeoff."

---

#### Cell 29: Evaluate SimpleFeedForward Performance

**[Utkrisht runs Cell 29]:**

**[Utkrisht speaks]:**

"Let's see the performance metrics:

**Output shows:**
- MAE: 14,055 cases (slightly worse than DeepAR's 13,905)
- RMSE: 14,261 cases (slightly worse than DeepAR's 14,183)
- MAPE: 40.2% (same as DeepAR!)

**Interesting findings:**
1. SimpleFeedForward is only marginally worse than DeepAR
   - About 150 cases more error on average
   - In percentage terms, same MAPE
   
2. For this particular test period, the simple model was nearly as good

3. But it trained 10x faster!

**The output says 'Decent baseline'** - and that's exactly what this is. In practice:
- Use SimpleFeedForward for quick experiments and baselines
- Use DeepAR when you need maximum accuracy
- The 10x speed advantage often matters more than 1% accuracy gain"

---

#### Cell 30: SimpleFeedForward Wrap-up

**[Utkrisht reads the key takeaways]:**

"The notebook summarizes SimpleFeedForward's value:

**When to use it:**
- Quick prototyping and experimentation
- Establishing baselines for comparison
- Production scenarios where retraining speed matters
- Data with stable, smooth trends

**Trade-offs:**
- ✓ 10x faster training
- ✓ Fewer hyperparameters to tune
- ✓ Good enough accuracy for many use cases
- ✗ Doesn't capture complex seasonality
- ✗ Can't use external features
- ✗ Less sophisticated pattern recognition

For COVID-19 forecasting in production, I'd recommend:
- Use SimpleFeedForward for daily quick forecasts
- Use DeepAR for weekly detailed analysis
- Compare both to catch discrepancies"

---

**[Utkrisht transitions]:**

"That's SimpleFeedForward! Now Deepika will demonstrate DeepNPTS - our most flexible model that adapts to changing distributions."

---

### DEEPIKA - DeepNPTS (GluonTS.API.ipynb Cells 35-50) - 3-4 minutes

**[Deepika takes over screen sharing]:**

**[Deepika scrolls to Part 3: DeepNPTS section]:**

#### Cells 35-37: DeepNPTS Introduction and Configuration

**[Deepika speaks]:**

"Thank you Utkrisht! Now I'll demonstrate **DeepNPTS** - Deep Non-Parametric Time Series.

DeepNPTS is special - it doesn't assume your data follows any particular distribution like normal or Poisson. Instead, it **learns the distribution directly from your data**.

This is crucial for COVID-19 because:
- Each wave behaves differently
- Delta variant ≠ Omicron variant
- The distribution of cases changes over time (regime shifts)
- We don't want to force-fit a Gaussian or Poisson when reality is more complex

Use DeepNPTS when:
- Your data distribution keeps changing
- You have regime shifts (patterns shift over time)
- You see unusual, non-standard distributions
- Standard assumptions don't fit your data"

**[Deepika runs Cell 37 - Configure DeepNPTS]:**

**[Deepika explains configuration]:**

"DeepNPTS configuration is similar to DeepAR but with key differences:

**Same as DeepAR:**
- `freq='D'`: Daily data
- `prediction_length=14`: 14-day forecast
- `context_length=60`: 60-day lookback
- `num_feat_dynamic_real=3`: Uses external features (deaths, CFR)

**Different from DeepAR:**
- `epochs=10` - passed directly, NOT via `trainer_kwargs`!
  - This is a quirk of DeepNPTS's API
  
- `num_hidden_nodes=[40, 40]` - NOT `hidden_size`!
  - Uses list of layer sizes like SimpleFeedForward
  - But unlike SimpleFeedForward, it DOES use features

**What makes it 'non-parametric'?**
- Most models say: 'I assume your errors are normally distributed'
- DeepNPTS says: 'Show me your data, I'll figure out the distribution'
- This flexibility helps with COVID's changing nature"

**[Output confirms configuration]:**

"Configuration set for 14-day forecasting with non-parametric distribution learning."

---

#### Cells 38-39: Train DeepNPTS

**[Deepika runs Cell 39 - Train DeepNPTS]:**

**[Deepika speaks during training]:**

"Training DeepNPTS now. Speed-wise, it's between DeepAR and SimpleFeedForward:
- Slower than SimpleFeedForward (has more complexity)
- Comparable to DeepAR (similar architecture)
- Takes about 1-2 minutes

Watch the training loss:
- Starting around 12.63
- Decreasing steadily to 12.42
- This shows the model is learning the data distribution

The non-parametric aspect means:
- It's learning not just 'what's the average' but 'what's the full shape of possible outcomes'
- More flexible but requires careful training"

**[Training completes]:**

"Training complete! DeepNPTS has learned the distribution of COVID-19 cases without making rigid assumptions."

---

#### Cells 40-41: Generate DeepNPTS Forecasts

**[Deepika runs Cell 41 - Generate Forecasts]:**

**[Deepika speaks]:**

"Generating 100 sample forecasts for uncertainty quantification.

**Output shows:**
- Mean prediction: ~45,000 cases per day
- Forecast range: 40,000 to 48,000 cases
- 90% confidence interval: much wider - 11,000 to 76,000

**Interesting comparison:**
- DeepAR: ~49k mean
- SimpleFeedForward: ~48k mean
- DeepNPTS: ~45k mean

DeepNPTS is predicting lower cases! Why?
- It might be adapting to the most recent declining trend
- The non-parametric nature makes it more responsive to recent changes
- Less anchored to historical average patterns

Also notice the very wide confidence interval:
- DeepNPTS is saying 'there's high uncertainty here'
- This honesty about uncertainty is valuable for planning"

---

#### Cell 43: Visualize DeepNPTS Predictions

**[Deepika runs Cell 43]:**

**[Graph appears with orange forecast line]:**

**[Deepika speaks while analyzing the visualization]:**

"This is DeepNPTS's forecast visualization. Let me highlight what's different:

**The forecast (orange):**
- Mean prediction: Lower than DeepAR and SimpleFeedForward
- Confidence bands: Noticeably wider, especially in later days
- Shape: Follows the declining trend more aggressively

**Comparing all three models visually:**

1. **Trend capturing:**
   - DeepNPTS: Most aggressive decline
   - DeepAR: Moderate decline
   - SimpleFeedForward: Smooth, stable decline

2. **Uncertainty quantification:**
   - DeepNPTS: Widest confidence intervals (most honest about uncertainty)
   - DeepAR: Moderate confidence intervals
   - SimpleFeedForward: Narrower intervals (perhaps overconfident)

3. **Actual values:**
   - Some orange dots are closer to DeepNPTS in later days
   - DeepNPTS might be capturing the recent regime shift better

**What this demonstrates:**
- Different models, different perspectives on the future
- DeepNPTS's flexibility shows in its adaptive predictions
- Wide uncertainty is appropriate for volatile data like COVID-19
- In practice, you'd ensemble multiple models for robust forecasts"

---

#### Cell 45: Evaluate DeepNPTS Performance

**[Deepika runs Cell 45]:**

**[Deepika speaks]:**

"Performance metrics for DeepNPTS:

**Output shows:**
- MAE: 12,785 cases - **Best of all three models!**
- RMSE: 13,303 cases - Also best!
- MAPE: 36.6% - Also best!

**This is surprising and informative:**
- DeepNPTS actually outperformed DeepAR and SimpleFeedForward
- Lower mean prediction turned out more accurate for this test period
- Non-parametric flexibility helped adapt to recent trends

**Why did it perform better?**
1. The test period (March 2023) was during a declining trend
2. DeepNPTS adapted more quickly to this regime
3. It wasn't constrained by distributional assumptions
4. The recent pattern was more important than historical average

**Output says 'Good performance!'** - and indeed, 36.6% MAPE is better than the others' 40.2%."

---

#### Cell 46: Model Comparison Summary

**[Deepika reads the comparison table]:**

"The notebook provides a helpful summary comparing all three:

**DeepAR:**
- Best for: Complex patterns, seasonality
- Speed: Slow (2-3 min)
- Accuracy: Good baseline
- Uncertainty: Well-calibrated
- When to use: Need maximum pattern recognition

**SimpleFeedForward:**
- Best for: Speed, simplicity
- Speed: Very fast (30 sec)
- Accuracy: Comparable to DeepAR
- Uncertainty: Adequate
- When to use: Quick experiments, rapid retraining

**DeepNPTS:**
- Best for: Regime changes, distribution shifts
- Speed: Medium (1-2 min)
- Accuracy: Best in our test!
- Uncertainty: Honest, wide intervals
- When to use: Volatile data, changing patterns

**In practice for COVID-19 forecasting:**
- Use all three and compare
- DeepAR for understanding historical patterns
- SimpleFeedForward for daily quick checks
- DeepNPTS during transitions (new variants, policy changes)
- Ensemble them for robust forecasts"

---

**[Deepika wraps up API notebook]:**

"That concludes our API demonstration! We've shown you:
- How to configure each model
- How to train and generate forecasts
- How to evaluate and visualize results
- When to use each model

Key takeaway: **Different models for different scenarios**. GluonTS gives you options!"

---

### PART B: Example Notebook - Complete Application

**[Deepika transitions]:**

"Now let's move to the **example notebook** - our complete COVID-19 forecasting application. This shows how to use GluonTS in a real-world scenario.

Let me open `GluonTS.example.ipynb`..."

**[Opens GluonTS.example.ipynb]:**

---

### ALL THREE - Example Notebook Walkthrough (3-4 minutes)

**[Harsh takes lead for introduction]:**

#### Cells 0-3: Problem Statement and Setup

**[Harsh speaks]:**

"The example notebook addresses a real-world problem:

**Problem:** Hospitals need to predict COVID-19 cases 14 days ahead to:
- Allocate ICU beds and ventilators
- Schedule healthcare staff appropriately
- Plan intervention strategies
- Communicate risk to the public

**Our solution:** A complete forecasting system using all three GluonTS models.

Let me run the setup cells..."

**[Harsh runs Cells 1-3 quickly]:**

"Same imports and data loading. The difference is this notebook focuses on the complete application workflow, not teaching the API."

---

#### Cells 4-5: Data Exploration

**[Utkrisht takes over]:**

**[Utkrisht runs Cell 4 - Data visualization]:**

"This comprehensive visualization shows:
- Multiple COVID waves over 3 years
- Weekly seasonality patterns
- The features we're using: deaths and mobility
- Clear correlation between decreased mobility and case declines

This justifies our feature selection - deaths and mobility DO help predict cases."

---

#### Cells 6-12: Training All Three Models

**[Deepika takes over]:**

**[Deepika speaks]:**

"For time, I'll show the training outputs we've already run. In a live demo, this would take about 6-7 minutes total.

The notebook trains:
1. DeepAR with full features
2. SimpleFeedForward as baseline
3. DeepNPTS for distribution flexibility

All use the same 60-day context, 14-day forecast configuration for fair comparison."

**[Scrolls through training outputs]:**

"You can see each model trained successfully, with decreasing loss values indicating learning."

---

#### Cells 13-18: Model Comparison

**[Harsh takes over]:**

**[Harsh runs comparison cells]:**

**[A comparison table appears]:**

**[Harsh speaks]:**

"This is the key result - comparing all three models side by side:

**Metrics comparison:**
- MAE, RMSE, MAPE for accuracy
- CRPS for probabilistic forecast quality
- Training time for practical considerations

**Typical findings:**
- DeepAR: Best for seasonal patterns, moderate training time
- SimpleFeedForward: Fastest, good baseline accuracy
- DeepNPTS: Best for this test period, adaptive to recent trends

**Visualization:**
A bar chart shows these metrics visually - easy to communicate to stakeholders."

---

#### Cells 19-22: Scenario Analysis

**[Utkrisht takes over]:**

**[Utkrisht speaks]:**

"Now the most powerful feature - scenario analysis. This is where probabilistic forecasting shines for public health decision-making.

We simulate three scenarios:
1. **Baseline:** No intervention, current trends continue
2. **Moderate:** 20% mobility reduction (mask mandates, capacity limits)
3. **Strong:** 40% mobility reduction (lockdowns, school closures)"

**[Runs scenario cells]:**

**[Output shows three different forecast trajectories]:**

**[Utkrisht explains]:**

"Look at the results:
- Baseline: 65,000 cases expected
- Moderate intervention: 52,000 cases (20% reduction)
- Strong intervention: 38,000 cases (42% reduction)

**Real-world value:**
- Decision-makers can see quantified intervention impact
- Balance health benefits vs economic costs
- Plan resources based on chosen scenario
- Communicate tradeoffs clearly to public

This is why GluonTS is powerful - not just predicting one future, but exploring multiple possible futures."

---

**[Deepika wraps up Step 5]:**

"That completes our project walkthrough! We've demonstrated:
- How each GluonTS model works (API notebook)
- A complete COVID-19 forecasting application (example notebook)
- Model comparison and selection
- Scenario analysis for decision support

Key achievement: **Production-ready forecasting system in under 10 minutes of code execution.**"

---

## UTKRISHT - Step 6: Discuss Results (2-3 minutes)

**[Utkrisht takes over]:**

**[Utkrisht speaks]:**

"Let me interpret what these results mean for our problem statement.

### Key Findings

**1. Model Performance:**
- DeepAR, SimpleFeedForward, and DeepNPTS all achieved MAPE between 36-40%
- For 14-day COVID forecasts, this is reasonable given the data's volatility
- DeepNPTS performed best on our test period (36.6% MAPE)
- SimpleFeedForward was competitive despite being 10x faster

**2. Uncertainty Quantification:**
- All models provided confidence intervals
- DeepNPTS had widest intervals (most honest about uncertainty)
- 90% confidence intervals typically ±15,000 cases
- This tells hospitals: 'Plan for average, prepare for variance'

**3. Speed vs Accuracy Trade-offs:**
- SimpleFeedForward: 30 seconds, 40% MAPE
- DeepNPTS: 1-2 minutes, 37% MAPE
- DeepAR: 2-3 minutes, 40% MAPE
- For daily retraining, SimpleFeedForward's speed advantage matters

**4. Scenario Analysis Impact:**
- Moderate interventions: 20% case reduction
- Strong interventions: 42% case reduction
- Quantifies policy options for decision-makers

### How GluonTS Solved Our Problem

**Problem:** Hospitals need 14-day forecasts for resource planning

**GluonTS provided:**

1. **Multiple modeling approaches:** Compare and choose best for situation

2. **Probabilistic forecasts:** Not just point predictions but full uncertainty quantification

3. **Feature integration:** Leverage deaths and mobility data for better accuracy

4. **Scenario simulation:** Test interventions before implementing

5. **Production-ready speed:** Fast enough for daily retraining

### Real-World Impact

If deployed, this system enables:
- **ICU bed allocation:** 2 weeks advance notice for capacity planning
- **Staff scheduling:** Optimize healthcare worker assignments
- **Supply chain management:** Order PPE and ventilators proactively
- **Policy evaluation:** Quantify intervention effects before implementing
- **Public communication:** Transparent uncertainty bounds build trust

### Technical Achievements

1. **Automated data pipeline:** Downloads, preprocesses, merges three data sources
2. **Fast training:** All three models in under 10 minutes on CPU
3. **Comprehensive evaluation:** Multiple metrics and visualizations
4. **Modular design:** Easy to extend with new models or features
5. **Reproducibility:** Docker ensures consistent environment

### Limitations and Future Work

**Current limitations:**
- 14-day horizon only (longer forecasts would be less accurate)
- Requires daily retraining for best results
- Assumes historical patterns continue (can't predict entirely new variants)
- National-level only (state/county forecasts would need more data)

**Future improvements:**
- Add vaccination data (already in data/ folder)
- Implement automated daily retraining pipeline
- Extend to state and county-level forecasts
- Integrate real-time hospital capacity data
- Ensemble all three models for robust predictions
- Add external data: weather, genomic surveillance, policy indices

### Bottom Line

GluonTS enabled a complete, production-ready COVID-19 forecasting system that:
- Provides actionable 14-day forecasts with uncertainty
- Runs fast enough for operational use
- Supports scenario-based decision making
- Can be deployed immediately for real-world impact

The hard difficulty was justified by:
- Multiple complex models integrated
- Sophisticated data pipeline with auto-download
- Probabilistic forecasting with uncertainty
- Scenario analysis capabilities
- Production-ready Docker deployment"

---

## DEEPIKA - Step 7: Documentation Review (2-3 minutes)

**[Deepika takes over, shares file browser]:**

**[Deepika speaks]:**

"Now let me show how our documentation serves both technical and non-technical audiences.

### Documentation Structure

We have three main documentation files:

**[Opens README.md]:**

### README.md - The Entry Point

"This is what anyone sees first. It's organized for quick access:

**1. Quick Start (top section):**
- Clone repo
- Build Docker: `./docker_build.sh`
- Run Jupyter: `./docker_jupyter.sh`
- Open notebooks
- 5-minute setup, anyone can do it

**2. Visual Documentation:**
Scroll down... see these Mermaid diagrams:
- Data pipeline flow: Shows how data moves from files → preprocessing → models
- Project structure: Clear hierarchy of components
- **Non-technical readers** can understand the workflow without reading code

**3. Data Setup Section:**
Explains automatic data download:
- System checks for files
- Downloads from Google Drive if missing
- Manual instructions if automatic fails
- Clear, step-by-step

**4. Expected Outputs:**
Shows what you should see in terminal:
- Docker build success messages
- Jupyter server startup
- No confusion or surprises

**Who can use this?**
- Project managers: Understand what we built
- Developers: Get up and running quickly
- Stakeholders: See the workflow visually
- Students: Learn and replicate our work"

---

**[Opens GluonTS.API.md]:**

### GluonTS.API.md - Tool Reference

**[Deepika scrolls through]:**

"This is **tool-focused documentation** - explains GluonTS itself, NOT our COVID project.

**Structure:**

**1. Model Overview:**
- DeepAR: 'Uses RNNs for complex patterns'
- SimpleFeedForward: 'Direct mapping, fast and simple'
- DeepNPTS: 'Learns distribution, adapts to changes'
- Plain English, no jargon

**2. When to Use Each Model:**
Decision guide:
- Complex seasonality → DeepAR
- Need speed → SimpleFeedForward
- Regime changes → DeepNPTS

**3. Parameter Reference:**
Every parameter explained:
- `context_length`: 'How much history to use'
- `prediction_length`: 'How far ahead to forecast'
- `num_feat_dynamic_real`: 'Number of external features'
- Includes typical values and effects

**4. Basic Usage Pattern:**
Step-by-step code examples:
- Prepare data
- Configure model
- Train
- Forecast
- Interpret
Generic examples - work for ANY time series problem

**5. Troubleshooting:**
Common issues and solutions:
- 'Training too slow' → reduce epochs
- 'Wrong number of features' → check num_feat_dynamic_real
- Practical fixes

**Who can use this?**
- Data scientists applying GluonTS to their own problems
- Students learning time series forecasting
- Developers building forecasting systems
- Anyone needing a GluonTS reference

**Key point:** No COVID-19 specifics here. This is reusable for sales, traffic, weather, anything!"

---

**[Opens GluonTS.example.md]:**

### GluonTS.example.md - Project Documentation

**[Deepika scrolls through]:**

"This is **project-focused** - explains our specific COVID-19 application.

**Structure:**

**1. Project Overview:**
- Problem statement: 'Hospitals need 14-day forecasts'
- Why it matters: Resource allocation, staff planning
- Real-world context

**2. Data Sources:**
- Cases from JHU
- Deaths from JHU
- Mobility from Google
- Explains WHY each matters for forecasting

**3. Feature Engineering Rationale:**
- 7-day moving average: 'Smooths weekend reporting artifacts'
- CFR: 'Indicates healthcare strain'
- Mobility: 'Captures behavioral changes'
- Technical decisions in plain language

**4. Model Selection for COVID-19:**
Why we chose these three:
- DeepAR: 'COVID has multiple complex waves'
- SimpleFeedForward: 'Fast baseline, good for stable periods'
- DeepNPTS: 'Each variant behaves differently'
- Domain-specific reasoning

**5. Notebook Walkthrough:**
Section-by-section explanation:
- What each cell does
- Expected outputs
- How to interpret results
- Complete guide for running

**6. Results Interpretation:**
- 'MAE of 13,000 means we're off by 13k cases on average'
- '90% CI of ±15k tells hospitals the range to prepare for'
- 'Scenario analysis shows 20% reduction with moderate intervention'
- Translates numbers to actionable insights

**7. Real-World Application:**
- How hospitals would use this
- Policy implications
- Communication to public
- Practical value demonstrated

**Who can use this?**
- Hospital administrators: Understand forecasts for their work
- Public health officials: Evaluate interventions quantitatively
- Policy makers: See cost-benefit of different scenarios
- Technical reviewers: Understand complete methodology
- Students: Learn applied forecasting in real domain

**Key point:** Everything specific to COVID-19 and this project is here, NOT in API.md"

---

### How Documentation Works Together

**[Deepika explains the flow]:**

"For different users:

**Scenario 1: Student wanting to learn GluonTS**
1. README.md → Get overview, set up environment
2. GluonTS.API.md → Learn the tool thoroughly
3. GluonTS.example.ipynb → See it applied to real problem
4. Adapt to their own project

**Scenario 2: Hospital administrator evaluating our work**
1. README.md → Understand what we built (Mermaid diagrams!)
2. GluonTS.example.md → See how it solves their problem
3. GluonTS.example.ipynb → View actual forecasts
4. Decide if it meets their needs

**Scenario 3: Developer extending our work**
1. README.md → Setup environment
2. GluonTS.API.md → Reference for parameters
3. Utility modules → Reuse our code
4. Build their own application

### Documentation Quality

**Completeness:**
- Every file explained
- Every function documented
- Every parameter described
- All results interpreted

**Clarity:**
- Plain language, minimal jargon
- Progressive complexity (simple → advanced)
- Visual aids (diagrams, charts)
- Real examples throughout

**Accessibility:**
- Multiple entry points (API vs example)
- 'What' and 'Why' before 'How'
- Troubleshooting sections
- Both technical and non-technical explanations

**Professional Standards:**
- Consistent formatting
- Logical organization
- Clear section headers
- Academic-appropriate tone
- No emojis or overly casual language"

---

**[Deepika concludes]:**

"In summary, our documentation architecture:
- **README.md:** Everyone - quick start and overview
- **GluonTS.API.md:** Technical users - tool reference
- **GluonTS.example.md:** All users - project explanation

Whether you're a hospital administrator, a data scientist, a student, or a policy maker, you can understand our work and use it appropriately."

---

## CLOSING (All 3) - 30 seconds

**[Harsh wraps up]:**

"Thank you for watching our demonstration!

**Summary of what we achieved:**

1. **Complete forecasting system** using three GluonTS models
2. **14-day COVID-19 forecasts** with 36-40% MAPE accuracy
3. **Probabilistic predictions** with uncertainty quantification
4. **Scenario analysis** for intervention planning
5. **Production-ready** Docker deployment
6. **Comprehensive documentation** for all audiences
7. **Modular, reusable** code architecture

**Key takeaway:** GluonTS enables sophisticated probabilistic forecasting that directly supports real-world public health decision-making.

Our hard-difficulty project demonstrates:
- Multiple complex models integrated and compared
- Automated data pipeline with error handling
- Advanced uncertainty quantification
- Scenario-based decision support
- Professional documentation and deployment

Questions?"

---

## TIMING BREAKDOWN

- **Harsh (Steps 1-4)**: 2-3 minutes
- **All 3 (Step 5)**:12-15 minutes
  - Harsh - DeepAR (API): 4-5 minutes
  - Utkrisht - SimpleFeedForward (API): 3-4 minutes
  - Deepika - DeepNPTS (API): 3-4 minutes
  - All - Example notebook: 3-4 minutes
- **Utkrisht (Step 6)**: 2-3 minutes
- **Deepika (Step 7)**: 2-3 minutes
- **Closing**: 0.5 minutes

**Total: 18-25 minutes** (aim for 20 minutes)

---

## TIPS FOR SUCCESSFUL DEMO

### Before Recording

1. **Practice multiple times:** Aim for smooth 20-minute delivery
2. **Pre-run notebooks:** Have outputs visible (saves waiting for training)
3. **Test screen recording:** Ensure good quality
4. **Check audio:** Clear microphone
5. **Clean desktop:** Professional appearance
6. **Backup outputs:** Screenshots if live demo fails

### During Recording

1. **Speak clearly:** Not too fast, pause between sections
2. **Zoom when needed:** Make code/outputs readable
3. **Point with cursor:** Highlight important information
4. **Explain, don't just read:** Interpret outputs
5. **Maintain energy:** Stay engaged throughout

### Common Pitfalls to Avoid

1. **Don't wait for training:** Use pre-run outputs
2. **Don't rush documentation:** It's as important as code
3. **Don't skip transitions:** Smooth handoffs between presenters
4. **Don't ignore errors:** Explain them gracefully if they occur
5. **Don't forget to conclude:** Summarize key achievements

### Hand-off Phrases

**Harsh → Utkrisht:**
"That's DeepAR! Now Utkrisht will demonstrate SimpleFeedForward."

**Utkrisht → Deepika:**
"Now Deepika will show DeepNPTS, our most flexible model."

**Deepika → All (after example notebook):**
"Now let's interpret these results. Utkrisht?"

**After Step 6 → Step 7:**
"And finally, let me review our documentation. Deepika?"

Good luck with your presentation!
