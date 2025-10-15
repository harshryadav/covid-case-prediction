import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set paths
data_dir = Path('../../data')
output_dir = Path('./training')

# Read and process the datasets
print("Reading and processing datasets...")
cases_df = pd.read_csv(data_dir / 'time_series_covid19_confirmed_US.csv')
deaths_df = pd.read_csv(data_dir / 'time_series_covid19_deaths_US.csv')

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

def process_time_series(df, start_col_index):
    # Get date columns (excluding metadata columns)
    date_columns = df.columns[start_col_index:]
    
    # Calculate daily total for entire US by summing across all states/counties
    us_total = df[date_columns].sum()
    
    # Convert series to dataframe with proper datetime index
    ts = pd.DataFrame({
        'Date': pd.to_datetime(us_total.index),
        'Total': us_total.values
    })
    
    return ts

# Process both datasets (deaths dataset has Population column, so it starts one column later)
cases_ts = process_time_series(cases_df, start_col_index=11)
deaths_ts = process_time_series(deaths_df, start_col_index=12)

# Create the visualization
plt.figure(figsize=(15, 10))

# Create two y-axes
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot cases on left y-axis with adjusted format
cases_line = ax1.plot(cases_ts['Date'], cases_ts['Total'] / 1_000_000, 
                      color='#3498db', linewidth=2.5, label='Confirmed Cases')
ax1.set_ylabel('Number of Cases (Millions)', color='#3498db', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#3498db')
ax1.grid(True, alpha=0.3)

# Plot deaths on right y-axis with adjusted format
deaths_line = ax2.plot(deaths_ts['Date'], deaths_ts['Total'] / 1_000, 
                       color='#e74c3c', linewidth=2.5, label='Deaths')
ax2.set_ylabel('Number of Deaths (Thousands)', color='#e74c3c', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#e74c3c')

# Customize the plot
plt.title('US COVID-19 Cases and Deaths Over Time\n(Cases in Millions, Deaths in Thousands)', 
          fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)

# Add both legends with better positioning
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
           bbox_to_anchor=(0.05, 0.95), framealpha=1)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid but keep it light
ax1.grid(True, alpha=0.3)

# Adjust layout and save with high resolution
plt.tight_layout()
plt.savefig(output_dir / 'us_covid_cases_deaths.png', 
            dpi=300, bbox_inches='tight', facecolor='white')

# Calculate and display statistics
print("\nCOVID-19 Statistics in the US:")
print("-" * 50)
print(f"Time period: {cases_ts['Date'].min().date()} to {cases_ts['Date'].max().date()}")
print(f"\nTotal Cases: {cases_ts['Total'].iloc[-1]:,.0f}")
print(f"Total Deaths: {deaths_ts['Total'].iloc[-1]:,.0f}")
print(f"Mortality Rate: {(deaths_ts['Total'].iloc[-1] / cases_ts['Total'].iloc[-1] * 100):.2f}%")

# Calculate daily numbers
cases_daily = cases_ts['Total'].diff()
deaths_daily = deaths_ts['Total'].diff()

print(f"\nDaily Averages (across entire period):")
print(f"Average Daily New Cases: {cases_daily.mean():,.0f}")
print(f"Average Daily Deaths: {deaths_daily.mean():,.0f}")

# Calculate peak statistics
peak_cases_day = cases_daily.idxmax()
peak_deaths_day = deaths_daily.idxmax()

print(f"\nPeak Statistics:")
print(f"Peak Daily Cases: {cases_daily.max():,.0f} (on {cases_ts['Date'].iloc[peak_cases_day].date()})")
print(f"Peak Daily Deaths: {deaths_daily.max():,.0f} (on {deaths_ts['Date'].iloc[peak_deaths_day].date()})")

plt.close()  # Close the figure to free memory
