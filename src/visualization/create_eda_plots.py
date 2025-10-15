"""
Create comprehensive EDA visualizations for COVID-19 data
Integrated from original visualize_covid_data_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def process_time_series(df, start_col_index):
    """
    Process time series data from JHU CSSE format
    
    Args:
        df: DataFrame with time series columns
        start_col_index: Index where date columns start
        
    Returns:
        DataFrame with Date, Cumulative, Daily, Daily_MA7 columns
    """
    # Get date columns (excluding metadata columns)
    date_columns = df.columns[start_col_index:]
    
    # Calculate daily total for entire US by summing across all states/counties
    cumulative = df[date_columns].sum()
    
    # Convert series to dataframe with proper datetime index
    ts = pd.DataFrame({
        'Date': pd.to_datetime(cumulative.index),
        'Cumulative': cumulative.values
    })
    
    # Calculate daily new cases/deaths
    ts['Daily'] = ts['Cumulative'].diff().fillna(0)
    
    # Calculate 7-day moving average for daily numbers
    ts['Daily_MA7'] = ts['Daily'].rolling(window=7).mean()
    
    return ts


def create_comprehensive_eda_plot(cases_ts, deaths_ts, output_path=None):
    """
    Create 4-panel EDA visualization
    
    Args:
        cases_ts: DataFrame with cases time series
        deaths_ts: DataFrame with deaths time series
        output_path: Where to save the plot (default: results/eda_visualization.png)
    """
    if output_path is None:
        output_path = Path('results') / 'eda_visualization.png'
    else:
        output_path = Path(output_path)
    
    # Create output directory
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('COVID-19 Cases and Deaths in the United States\n(With Daily Numbers and 7-Day Moving Average)', 
                 fontsize=16, y=0.95)
    
    # 1. Cumulative Cases
    ax1.plot(cases_ts['Date'], cases_ts['Cumulative'] / 1_000_000, 
             color='#3498db', linewidth=2.5, label='Total Cases')
    ax1.set_title('Cumulative COVID-19 Cases', fontsize=12, pad=10)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Total Cases (Millions)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # 2. Daily New Cases
    ax2.bar(cases_ts['Date'], cases_ts['Daily'] / 1_000, 
            color='#3498db', alpha=0.3, label='Daily New Cases')
    ax2.plot(cases_ts['Date'], cases_ts['Daily_MA7'] / 1_000, 
             color='#2980b9', linewidth=2, label='7-Day Moving Average')
    ax2.set_title('Daily New COVID-19 Cases', fontsize=12, pad=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Daily New Cases (Thousands)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # 3. Cumulative Deaths
    ax3.plot(deaths_ts['Date'], deaths_ts['Cumulative'] / 1_000, 
             color='#e74c3c', linewidth=2.5, label='Total Deaths')
    ax3.set_title('Cumulative COVID-19 Deaths', fontsize=12, pad=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Total Deaths (Thousands)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    # 4. Daily New Deaths
    ax4.bar(deaths_ts['Date'], deaths_ts['Daily'], 
            color='#e74c3c', alpha=0.3, label='Daily Deaths')
    ax4.plot(deaths_ts['Date'], deaths_ts['Daily_MA7'], 
             color='#c0392b', linewidth=2, label='7-Day Moving Average')
    ax4.set_title('Daily New COVID-19 Deaths', fontsize=12, pad=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Daily Deaths', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved EDA plot to {output_path}")
    
    plt.close()
    
    return output_path


def print_statistics(cases_ts, deaths_ts):
    """Print comprehensive statistics"""
    print("\n" + "="*60)
    print("COVID-19 STATISTICS IN THE US")
    print("="*60)
    
    print(f"\nTime period: {cases_ts['Date'].min().date()} to {cases_ts['Date'].max().date()}")
    
    print("\nCumulative Statistics:")
    print(f"  Total Cases: {cases_ts['Cumulative'].iloc[-1]:,.0f}")
    print(f"  Total Deaths: {deaths_ts['Cumulative'].iloc[-1]:,.0f}")
    print(f"  Overall Mortality Rate: {(deaths_ts['Cumulative'].iloc[-1] / cases_ts['Cumulative'].iloc[-1] * 100):.2f}%")
    
    print("\nDaily Statistics (7-day moving average at the end of the period):")
    print(f"  New Cases: {cases_ts['Daily_MA7'].iloc[-1]:,.0f}")
    print(f"  New Deaths: {deaths_ts['Daily_MA7'].iloc[-1]:,.0f}")
    
    print("\nPeak Statistics:")
    print("  Daily New Cases:")
    peak_cases_idx = cases_ts['Daily'].idxmax()
    print(f"    Peak: {cases_ts['Daily'].max():,.0f} (on {cases_ts['Date'].iloc[peak_cases_idx].date()})")
    print(f"    7-day Average at Peak: {cases_ts['Daily_MA7'].iloc[peak_cases_idx]:,.0f}")
    
    print("\n  Daily Deaths:")
    peak_deaths_idx = deaths_ts['Daily'].idxmax()
    print(f"    Peak: {deaths_ts['Daily'].max():,.0f} (on {deaths_ts['Date'].iloc[peak_deaths_idx].date()})")
    print(f"    7-day Average at Peak: {deaths_ts['Daily_MA7'].iloc[peak_deaths_idx]:,.0f}")


if __name__ == "__main__":
    print("="*60)
    print("CREATING EDA VISUALIZATIONS")
    print("="*60)
    
    # Load data
    from pathlib import Path
    data_dir = Path('data')
    
    print("\nReading and processing datasets...")
    cases_df = pd.read_csv(data_dir / 'time_series_covid19_confirmed_US.csv')
    deaths_df = pd.read_csv(data_dir / 'time_series_covid19_deaths_US.csv')
    
    # Process both datasets (deaths dataset has Population column, so it starts one column later)
    cases_ts = process_time_series(cases_df, start_col_index=11)
    deaths_ts = process_time_series(deaths_df, start_col_index=12)
    
    # Create visualization
    print("\nCreating comprehensive EDA plot...")
    output_path = create_comprehensive_eda_plot(cases_ts, deaths_ts)
    
    # Print statistics
    print_statistics(cases_ts, deaths_ts)
    
    # Save processed data
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    cases_ts.to_csv(output_dir / 'processed_us_cases.csv', index=False)
    deaths_ts.to_csv(output_dir / 'processed_us_deaths.csv', index=False)
    print(f"\n✓ Saved processed data to {output_dir}")
    
    print("\n" + "="*60)
    print("EDA VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nView the plot: {output_path}")

