# COVID-19 Case Prediction Project

This project analyzes and visualizes COVID-19 case, death, and vaccination data in the United States. It includes data processing scripts and visualization tools to help understand the trends and patterns in the pandemic data.

## Project Structure

```
data/
    time_series_covid19_confirmed_US.csv
    time_series_covid19_deaths_US.csv
    time_series_covid19_vaccine_us.csv
src/
    model/
        prepare_time_series.py      # Data processing script
        visualize_covid_data_analysis.py  # Visualization script
        training/
            processed_us_cases.csv   # Processed cases data
            processed_us_deaths.csv  # Processed deaths data
            us_covid_analysis.png   # Data visualization
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/harshryadav/covid-case-prediction-project.git
cd covid-case-prediction-project
```

2. Install required Python packages:
```bash
pip install pandas matplotlib numpy
```

## Usage

1. Process the raw data:
```bash
python src/model/prepare_time_series.py
```

2. Generate visualizations:
```bash
python src/model/visualize_covid_data_analysis.py
```

## Data Sources

The data used in this project comes from:
- Johns Hopkins CSSE COVID-19 Data Repository
- CDC COVID-19 Vaccination Data

## Visualizations

The project generates comprehensive visualizations including:
- Daily new cases with 7-day moving average
- Daily deaths with 7-day moving average
- Cumulative cases over time
- Cumulative deaths over time

## License

MIT License