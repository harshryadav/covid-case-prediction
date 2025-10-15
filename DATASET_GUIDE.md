# COVID-19 Case Prediction - Dataset Guide

This document provides comprehensive information about all datasets required for the COVID-19 case prediction project, including download instructions, data formats, and usage guidelines.

---

## 📊 Dataset Overview

| Dataset | Status | Required | Size | Time Period | Update Frequency |
|---------|--------|----------|------|-------------|------------------|
| JHU CSSE Confirmed Cases (US) | ✅ Available | Required | ~15 MB | Jan 2020 - Mar 2023 | Archived |
| JHU CSSE Deaths (US) | ✅ Available | Required | ~15 MB | Jan 2020 - Mar 2023 | Archived |
| JHU CSSE Vaccines (US) | ✅ Available | Optional | ~5 MB | Dec 2020 - Mar 2023 | Archived |
| Google Mobility Reports | 🔄 To Download | Bonus | ~200 MB | Feb 2020 - Oct 2022 | Archived |
| Oxford Government Response | 🔄 Optional | Optional | ~10 MB | Jan 2020 - Present | Weekly |

---

## 1️⃣ JHU CSSE COVID-19 Data (✅ Available)

### Overview
The Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) maintained the most comprehensive COVID-19 dataset from January 2020 to March 2023.

### Important Note
⚠️ **The JHU CSSE data collection ceased on March 10, 2023.** The repository is now archived and read-only.

### Dataset Details

#### 1.1 Time Series Confirmed Cases - US

**Current Location:** `/data/time_series_covid19_confirmed_US.csv`

**Source URL:** 
```
https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv
```

**Description:**
- Daily cumulative confirmed COVID-19 cases in the United States
- County-level granularity (3,000+ rows)
- Time series from January 22, 2020 to March 9, 2023

**Columns:**
```
- UID: Unique identifier
- iso2, iso3: Country codes
- code3: Numeric country code
- FIPS: Federal Information Processing Standards code (county identifier)
- Admin2: County name
- Province_State: State name
- Country_Region: "US"
- Lat: Latitude
- Long_: Longitude
- Combined_Key: "County, State, US"
- 1/22/20, 1/23/20, ..., 3/9/23: Daily cumulative case counts
```

**Example Row:**
```
UID,iso2,iso3,code3,FIPS,Admin2,Province_State,Country_Region,Lat,Long_,Combined_Key,1/22/20,1/23/20,...
84001001,US,USA,840,1001.0,Autauga,Alabama,US,32.53952745,-86.64408227,"Autauga, Alabama, US",0,0,...
```

**Data Processing Notes:**
- Values are **cumulative** - need to calculate daily new cases
- Some counties have missing FIPS codes
- "Unassigned" and "Out of State" entries exist
- Weekend reporting delays may cause artificial drops/spikes

---

#### 1.2 Time Series Deaths - US

**Current Location:** `/data/time_series_covid19_deaths_US.csv`

**Source URL:**
```
https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv
```

**Description:**
- Daily cumulative COVID-19 deaths in the United States
- County-level granularity
- Includes population data for each county

**Columns:**
```
- UID: Unique identifier
- iso2, iso3: Country codes
- code3: Numeric country code
- FIPS: Federal Information Processing Standards code
- Admin2: County name
- Province_State: State name
- Country_Region: "US"
- Lat: Latitude
- Long_: Longitude
- Combined_Key: "County, State, US"
- Population: County population (2019 estimate)
- 1/22/20, 1/23/20, ..., 3/9/23: Daily cumulative death counts
```

**Example Row:**
```
UID,iso2,iso3,code3,FIPS,Admin2,Province_State,Country_Region,Lat,Long_,Combined_Key,Population,1/22/20,...
84001001,US,USA,840,1001.0,Autauga,Alabama,US,32.53952745,-86.64408227,"Autauga, Alabama, US",55869,0,0,...
```

**Use Cases:**
- Calculate case fatality rate (CFR)
- Analyze mortality trends
- Per capita death rate calculations using Population column
- Model deaths as secondary target variable

---

#### 1.3 Time Series Vaccines - US

**Current Location:** `/data/time_series_covid19_vaccine_us.csv`

**Source URL:**
```
https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_vaccine_us.csv
```

**Description:**
- Daily vaccination data for United States
- State-level granularity
- Includes multiple vaccination metrics

**Columns:**
```
- UID: Unique identifier
- iso2, iso3: Country codes
- code3: Numeric country code
- FIPS: State FIPS code
- Admin2: (empty for state-level)
- Province_State: State name
- Country_Region: "US"
- Lat: Latitude
- Long_: Longitude
- Combined_Key: "State, US"
- Date: YYYY-MM-DD
- People_fully_vaccinated: Number of people fully vaccinated
- (Additional columns may include: doses administered, partially vaccinated, etc.)
```

**Data Processing Notes:**
- Data starts from December 2020 (vaccine rollout)
- State-level only (not county-level)
- Vaccination definitions changed over time (2-dose vs. 3-dose "fully vaccinated")
- Some states have data quality issues

**Use Cases:**
- Model impact of vaccination on case reduction
- Include as covariate in forecasting models
- Scenario analysis: "What if vaccination rate was X%?"

---

### How to Update/Refresh JHU Data (If Needed)

Since the JHU CSSE repository is archived, you already have the complete dataset. However, if you need to re-download:

```bash
# Navigate to your data directory
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction/data

# Download from GitHub (raw file URL)
wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv

wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv

wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_vaccine_us.csv
```

Or using Python:
```python
import requests
import pandas as pd

base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

files = [
    "time_series_covid19_confirmed_US.csv",
    "time_series_covid19_deaths_US.csv",
    "time_series_covid19_vaccine_us.csv"
]

for file in files:
    url = base_url + file
    df = pd.read_csv(url)
    df.to_csv(f"data/{file}", index=False)
    print(f"Downloaded: {file}")
```

---

## 2️⃣ Google COVID-19 Community Mobility Reports (🔄 Bonus)

### Overview
Google's Community Mobility Reports show movement trends over time by geography, across different categories of places. This data is invaluable for understanding how human behavior changed during the pandemic and can serve as a leading indicator for case trends.

### Status: To Be Downloaded

### Why This Data Matters
- Mobility changes often **precede** changes in case counts by 1-2 weeks
- Can improve forecast accuracy significantly
- Enables "what-if" scenarios (e.g., "What if we reduce mobility by 20%?")

### Dataset Details

**Official Source:** 
- ~~Google COVID-19 Community Mobility Reports~~ (Google stopped publishing in October 2022)
- [Google Mobility Reports Archive](https://www.google.com/covid19/mobility/)

**Recommended Source:** ✅

**[ActiveConclusion COVID-19 Mobility Archive](https://github.com/ActiveConclusion/COVID19_mobility)** - Best option!
- Complete US mobility data archive
- Date range: February 15, 2020 - February 1, 2022
- Pre-filtered for United States only
- Direct file: [`mobility_report_US.csv`](https://github.com/ActiveConclusion/COVID19_mobility/blob/master/google_reports/mobility_report_US.csv)
- No additional processing needed!

### Recommended Download Method

**✅ Option 1: Download from ActiveConclusion (RECOMMENDED - Already in your data folder!)**

You already have this file as `mobility_report_US.csv` in your data directory! If you need to re-download:

```bash
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction/data

# Download directly from ActiveConclusion GitHub
wget https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv

# Or using curl
curl -O https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv
```

**Advantages of ActiveConclusion dataset:**
- ✅ Covers Feb 2020 - Feb 2022 (24 months vs 10.5 months)
- ✅ Already filtered for US only
- ✅ Clean format with columns: `state`, `county`, `date`, mobility metrics
- ✅ No additional processing required
- ✅ Includes all pandemic waves (2020-2021, Delta, early Omicron)

**Option 2: Download from Our World in Data (Alternative)**

```bash
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction/data

# Download the global mobility data
wget https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/google_mobility/Google%20Mobility%20Trends.csv -O google_mobility_global.csv

# Filter for US only (using Python)
python << EOF
import pandas as pd

# Read global data
df = pd.read_csv('google_mobility_global.csv')

# Filter for United States
us_df = df[df['country_code'] == 'USA'].copy()

# Save
us_df.to_csv('google_mobility_owid.csv', index=False)
print(f"Saved US mobility data: {len(us_df)} rows")
print(f"Date range: {us_df['date'].min()} to {us_df['date'].max()}")
EOF
```

**Note:** Our World in Data has shorter coverage than ActiveConclusion.

### Mobility Data Columns (ActiveConclusion Format)

**File:** `mobility_report_US.csv`

```
- state: State name or "Total" for national level
- county: County name or "Total" for state/national level
- date: YYYY-MM-DD format
- retail and recreation: % change in visits to retail & recreation
- grocery and pharmacy: % change in visits to grocery stores & pharmacies
- parks: % change in visits to parks
- transit stations: % change in visits to transit stations
- workplaces: % change in visits to workplaces
- residential: % change in time spent at residential locations
```

**Example rows:**
```
state,county,date,retail and recreation,grocery and pharmacy,parks,transit stations,workplaces,residential
Total,Total,2020-02-15,6.0,2.0,15.0,3.0,2.0,-1.0  # National level
California,Total,2020-02-15,8.0,3.0,20.0,5.0,3.0,-2.0  # State level
California,Los Angeles County,2020-02-15,7.0,2.0,18.0,4.0,2.0,-1.0  # County level
```

**Baseline:**
- Mobility baseline is the median value for the corresponding day of the week, during the 5-week period Jan 3 – Feb 6, 2020
- All values are **percent changes** from this baseline
- Example: `-20` means 20% decrease from baseline, `+15` means 15% increase

### Data Coverage (ActiveConclusion Dataset)
- **Geographic:** County, State, and National levels for US
- **Temporal:** February 15, 2020 to February 1, 2022 (717 days, ~24 months)
- **Frequency:** Daily
- **Granularity:** 
  - National level: 718 rows (`state='Total', county='Total'`)
  - State level: 36,618 rows (`state!=Total, county='Total'`)
  - County level: 1,809,874 rows
- **Missing Data:** Some rural counties have missing values (privacy thresholds)

### How to Use Mobility Data

**Load and Prepare the Data:**
```python
import pandas as pd
from gluonts.dataset.common import ListDataset

# Load ActiveConclusion mobility data
mobility_df = pd.read_csv('data/mobility_report_US.csv')

# Extract national-level data
us_mobility = mobility_df[mobility_df['state'] == 'Total'].copy()
us_mobility['date'] = pd.to_datetime(us_mobility['date'])
us_mobility = us_mobility.sort_values('date')

# Prepare mobility features (note the column names without underscores)
mobility_features = [
    'retail and recreation',
    'grocery and pharmacy',
    'transit stations',
    'workplaces',
    'residential'
]
```

**As GluonTS Covariates:**
```python
# Create dataset with dynamic features
train_ds = ListDataset(
    [
        {
            "start": "2020-02-15",
            "target": daily_cases_array,
            "feat_dynamic_real": us_mobility[mobility_features].values.T  # Shape: (num_features, time_steps)
        }
    ],
    freq="D"
)
```

**Preprocessing Steps:**
1. **Filter level:** Extract national (`state=='Total'`) or state-level data
2. **Handle missing values:** Forward fill or interpolate
3. **Normalize:** Optionally standardize features (already in % change)
4. **Create lags:** Shift by 7-14 days (mobility affects cases with delay)
5. **Align dates:** Match with COVID case data (both start Feb 15, 2020)

### Expected Impact
- Studies show mobility data can improve COVID forecast accuracy by **15-30%**
- Most predictive features: `workplaces` and `retail and recreation`
- Best lag period: 10-14 days before case changes
- **Data coverage:** Feb 15, 2020 - Feb 1, 2022 (covers 2 major pandemic years)

---

## 3️⃣ Oxford COVID-19 Government Response Tracker (🔄 Optional)

### Overview
The Oxford COVID-19 Government Response Tracker (OxCGRT) systematically collects information on policy measures that governments have taken to tackle COVID-19. This includes lockdowns, school closures, travel restrictions, and economic support.

### Why This Data Matters
- Enables **scenario analysis**: "What if lockdowns were 20% stricter?"
- Helps interpret historical case trends
- Can be used as categorical covariates in models

### Dataset Details

**Source:** https://github.com/OxCGRT/covid-policy-tracker

**Download Instructions:**

```bash
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction/data

# Download latest data
wget https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv

# Filter for United States
python << EOF
import pandas as pd

df = pd.read_csv('OxCGRT_latest.csv')
us_df = df[df['CountryCode'] == 'USA'].copy()
us_df.to_csv('oxford_government_response_us.csv', index=False)

print(f"Saved US policy data: {len(us_df)} rows")
EOF
```

### Key Columns

**Stringency Index:**
- `StringencyIndex`: Overall government response stringency (0-100)
- Composite measure of lockdown policies

**Individual Policies (0-4 scale typically):**
- `C1_School closing`
- `C2_Workplace closing`
- `C3_Cancel public events`
- `C4_Restrictions on gatherings`
- `C5_Close public transport`
- `C6_Stay at home requirements`
- `C7_Restrictions on internal movement`
- `C8_International travel controls`

**Economic Support:**
- `E1_Income support`
- `E2_Debt relief`

**Health System:**
- `H1_Public information campaigns`
- `H2_Testing policy`
- `H3_Contact tracing`
- `H6_Facial coverings`
- `H7_Vaccination policy`

### Usage in Models

**As Categorical Covariates:**
```python
# Discretize StringencyIndex into categories
def categorize_stringency(index):
    if index < 30:
        return 0  # Low
    elif index < 60:
        return 1  # Medium
    else:
        return 2  # High

policy_df['StringencyCategory'] = policy_df['StringencyIndex'].apply(categorize_stringency)

# Use as feat_static_cat or feat_dynamic_cat in GluonTS
```

**For Scenario Analysis:**
```python
# Define scenarios
scenarios = {
    'baseline': us_df['StringencyIndex'],
    'strict_lockdown': us_df['StringencyIndex'] + 20,  # Increase stringency by 20 points
    'relaxed': us_df['StringencyIndex'] - 15
}

# Generate forecasts for each scenario
```

### Data Coverage
- **Geographic:** Country and State level
- **Temporal:** January 1, 2020 to Present (still updating)
- **Frequency:** Daily
- **Quality:** High (manually curated by Oxford team)

---

## 📊 Data Integration Strategy

### Recommended Aggregation Level

For this project, recommend **National (US) level** aggregation:

**Pros:**
- Simpler to start with
- More stable trends (less noise)
- Sufficient data for deep learning models
- Easier to interpret

**Cons:**
- Loses regional heterogeneity
- May miss local outbreaks

**Alternative:** Focus on **top 5-10 states** by population

### Data Alignment

All datasets need to be aligned on:
1. **Date range:** Use intersection of available dates
2. **Geographic level:** Aggregate to same level (national/state)
3. **Frequency:** Daily (already aligned)

### Proposed Unified Dataset Structure

```
Date         | Daily_Cases | Daily_Deaths | Vaccination_Rate | Mobility_Retail | Mobility_Work | Stringency_Index
-------------|-------------|--------------|------------------|-----------------|---------------|------------------
2020-02-15   | 15          | 0            | 0.0              | 0               | 0             | 25.0
2020-02-16   | 18          | 0            | 0.0              | -2              | -1            | 25.0
...          | ...         | ...          | ...              | ...             | ...           | ...
```

---

## 🔍 Data Quality Checks

### Validation Checklist

- [ ] **No missing dates:** Time series should be continuous
- [ ] **Non-negative values:** Cases, deaths, vaccinations cannot be negative
- [ ] **Monotonic cumulative:** Cumulative counts should not decrease
- [ ] **Outliers:** Check for reporting anomalies (spikes/drops)
- [ ] **Alignment:** All datasets cover same date range
- [ ] **Data types:** Dates parsed correctly as datetime

### Data Cleaning Script Template

```python
import pandas as pd
import numpy as np

def validate_dataset(df, date_col='Date', value_col='Daily_Cases'):
    """Validate time series data quality"""
    
    # Check for missing dates
    date_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq='D')
    missing_dates = set(date_range) - set(df[date_col])
    if missing_dates:
        print(f"Warning: {len(missing_dates)} missing dates")
    
    # Check for negative values
    if (df[value_col] < 0).any():
        print(f"Warning: Negative values found in {value_col}")
        df[value_col] = df[value_col].clip(lower=0)
    
    # Check for outliers (values > 5 std from mean)
    mean = df[value_col].mean()
    std = df[value_col].std()
    outliers = df[df[value_col] > mean + 5*std]
    if len(outliers) > 0:
        print(f"Warning: {len(outliers)} outliers detected")
    
    return df
```

---

## 📥 Quick Start: Data Download Script

Create this script to automate downloading all datasets:

**File:** `src/data_processing/download_all_datasets.py`

```python
"""
Download all required datasets for COVID-19 prediction project
"""
import os
import requests
import pandas as pd
from pathlib import Path

# Create data directory
data_dir = Path("../../data")
data_dir.mkdir(exist_ok=True)

print("=" * 60)
print("COVID-19 DATA DOWNLOAD SCRIPT")
print("=" * 60)

# 1. JHU CSSE Data (refresh even if exists)
print("\n[1/3] Downloading JHU CSSE COVID-19 data...")
jhu_base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

jhu_files = {
    "time_series_covid19_confirmed_US.csv": "US confirmed cases",
    "time_series_covid19_deaths_US.csv": "US deaths",
    "time_series_covid19_vaccine_us.csv": "US vaccinations"
}

for filename, description in jhu_files.items():
    url = jhu_base_url + filename
    try:
        df = pd.read_csv(url)
        output_path = data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"  ✓ Downloaded {description}: {len(df)} rows")
    except Exception as e:
        print(f"  ✗ Error downloading {filename}: {e}")

# 2. Google Mobility Data (ActiveConclusion)
print("\n[2/3] Downloading Google Mobility data from ActiveConclusion...")
try:
    mobility_url = "https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv"
    mobility_df = pd.read_csv(mobility_url)
    
    output_path = data_dir / "mobility_report_US.csv"
    mobility_df.to_csv(output_path, index=False)
    
    # Get stats
    national_data = mobility_df[mobility_df['state'] == 'Total']
    print(f"  ✓ Downloaded US mobility data: {len(mobility_df):,} rows")
    print(f"    Date range: {national_data['date'].min()} to {national_data['date'].max()}")
    print(f"    Coverage: Feb 2020 - Feb 2022 (24 months)")
except Exception as e:
    print(f"  ✗ Error downloading mobility data: {e}")
    print("    Alternative: Download manually from: https://github.com/ActiveConclusion/COVID19_mobility")

# 3. Oxford Government Response (Optional)
print("\n[3/3] Downloading Oxford Government Response data...")
try:
    oxford_url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
    oxford_df = pd.read_csv(oxford_url)
    
    # Filter for US
    us_oxford = oxford_df[oxford_df['CountryCode'] == 'USA'].copy()
    output_path = data_dir / "oxford_government_response_us.csv"
    us_oxford.to_csv(output_path, index=False)
    print(f"  ✓ Downloaded US policy data: {len(us_oxford)} rows")
except Exception as e:
    print(f"  ⚠ Optional data not downloaded: {e}")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE!")
print("=" * 60)
print(f"\nAll data saved to: {data_dir.absolute()}")
print("\nNext steps:")
print("  1. Run EDA notebook: notebooks/01_exploratory_data_analysis.ipynb")
print("  2. Preprocess data: python src/data_processing/prepare_gluonts_format.py")
```

---

## 📚 Additional Data Sources (Future Enhancements)

### Weather Data
- **Source:** NOAA Climate Data Online
- **Use:** Temperature, humidity as covariates
- **URL:** https://www.ncdc.noaa.gov/cdo-web/

### Hospital Capacity
- **Source:** COVID-19 Reported Patient Impact and Hospital Capacity by State
- **Use:** Resource allocation, severity modeling
- **URL:** https://healthdata.gov/

### Testing Data
- **Source:** COVID Tracking Project (archived)
- **Use:** Adjust case counts for testing rates
- **URL:** https://covidtracking.com/data/download

### Genomic Surveillance (Variants)
- **Source:** GISAID
- **Use:** Variant prevalence as covariate
- **URL:** https://www.gisaid.org/

---

## ✅ Dataset Checklist

Before starting model development:

- [ ] JHU CSSE cases data downloaded
- [ ] JHU CSSE deaths data downloaded
- [ ] JHU CSSE vaccine data downloaded
- [ ] Google mobility data downloaded (bonus)
- [ ] Oxford policy data downloaded (optional)
- [ ] All files in `data/` directory
- [ ] Data validated (no missing dates, negative values)
- [ ] Date ranges aligned
- [ ] Geographic aggregation decided (national/state)

---

## 📞 Support

If you encounter issues downloading data:
- Check network connection
- Verify URLs haven't changed (repositories may update)
- Use alternative sources listed above
- Check GitHub repository: https://github.com/CSSEGISandData/COVID-19

---

**Last Updated:** October 2025
**Data Current As Of:** March 10, 2023 (JHU CSSE data collection end date)

