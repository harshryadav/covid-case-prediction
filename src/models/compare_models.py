"""
Compare performance of all forecasting models

Loads metrics from all trained models and creates comparison visualizations.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean, professional style
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def load_all_metrics(results_dir='results'):
    """Load metrics from all trained models"""
    results_dir = Path(results_dir)
    
    models = {
        'Naive': 'baseline_metrics.csv',
        'DeepAR': 'deepar_metrics.csv',
        'TFT': 'tft_metrics.csv',
        'Prophet': 'prophet_metrics.csv',
        'WaveNet': 'wavenet_metrics.csv',
        'DeepVAR': 'deepvar_metrics.csv'
    }
    
    all_metrics = {}
    available_models = []
    
    for model_name, filename in models.items():
        filepath = results_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            all_metrics[model_name] = df.iloc[0].to_dict()
            available_models.append(model_name)
            print(f"✓ Loaded: {model_name}")
        else:
            print(f"⚠ Missing: {model_name} ({filename})")
    
    return all_metrics, available_models


def create_comparison_table(all_metrics, available_models):
    """Create a comparison table of key metrics"""
    
    # Key metrics to compare
    metric_names = {
        'RMSE': 'RMSE',
        'abs_error': 'MAE',
        'MAPE': 'MAPE (%)',
        'sMAPE': 'sMAPE (%)',
        'mean_wQuantileLoss': 'CRPS',
    }
    
    # Build comparison dataframe
    comparison_data = []
    
    for model_name in available_models:
        metrics = all_metrics[model_name]
        row = {'Model': model_name}
        
        for key, label in metric_names.items():
            value = None
            # Try different possible metric names
            for possible_key in [key, key.lower(), key.upper()]:
                if possible_key in metrics:
                    value = metrics[possible_key]
                    break
            
            if value is not None:
                # Format percentage metrics
                if 'MAPE' in label or 'sMAPE' in label:
                    row[label] = f"{value:.2%}"
                else:
                    row[label] = f"{value:.2f}"
            else:
                row[label] = 'N/A'
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def plot_metric_comparison(all_metrics, available_models, output_dir):
    """Create bar plots comparing key metrics across models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Metrics to plot
    metrics_to_plot = [
        ('RMSE', 'Root Mean Square Error (Lower is Better)', 0),
        ('abs_error', 'Mean Absolute Error (Lower is Better)', 1),
        ('MAPE', 'Mean Absolute Percentage Error (Lower is Better)', 2),
        ('mean_wQuantileLoss', 'CRPS (Lower is Better)', 3)
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for metric_key, title, idx in metrics_to_plot:
        ax = axes[idx // 2, idx % 2]
        
        # Extract metric values
        values = []
        labels = []
        
        for model_name in available_models:
            metrics = all_metrics[model_name]
            
            # Try to find the metric
            value = None
            for possible_key in [metric_key, metric_key.lower(), metric_key.upper()]:
                if possible_key in metrics:
                    value = metrics[possible_key]
                    break
            
            if value is not None:
                values.append(value)
                labels.append(model_name)
        
        if values:
            # Create bar plot
            bars = ax.bar(labels, values, color=colors[:len(labels)])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if 'MAPE' in metric_key or 'Percentage' in title:
                    label_text = f'{height:.1%}'
                else:
                    label_text = f'{height:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label_text, ha='center', va='bottom', fontsize=10)
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Highlight best model (lowest value)
            best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)
        else:
            ax.text(0.5, 0.5, 'Data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_dir / 'model_comparison.png'}")
    plt.close()


def plot_forecast_comparison(output_dir):
    """Create a side-by-side comparison of forecast plots"""
    
    forecast_files = {
        'Baseline': 'baseline_forecasts.png',
        'DeepAR': 'deepar_forecast.png',
        'TFT': 'tft_forecast.png',
        'Prophet': 'prophet_forecast.png',
        'DeepVAR': 'deepvar_forecast.png'
    }
    
    available_forecasts = {}
    for model_name, filename in forecast_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            available_forecasts[model_name] = filepath
    
    if len(available_forecasts) < 2:
        print("⚠ Not enough forecast plots to compare")
        return
    
    # Create grid of forecast plots
    n_models = len(available_forecasts)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    fig.suptitle('Forecast Comparison - All Models', fontsize=16, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, filepath) in enumerate(available_forecasts.items()):
        try:
            img = plt.imread(filepath)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{model_name} Model', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        except Exception as e:
            print(f"⚠ Could not load {model_name} forecast: {e}")
    
    # Hide unused subplots
    for idx in range(len(available_forecasts), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_forecasts_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved forecast comparison: {output_dir / 'all_forecasts_comparison.png'}")
    plt.close()


def create_ranking_table(all_metrics, available_models, output_dir):
    """Create a ranking table showing which model performs best on each metric"""
    
    metrics_to_rank = [
        ('RMSE', 'ascending'),
        ('abs_error', 'ascending'),
        ('MAPE', 'ascending'),
        ('sMAPE', 'ascending'),
        ('mean_wQuantileLoss', 'ascending'),
    ]
    
    rankings = []
    
    for metric_key, order in metrics_to_rank:
        values = []
        
        for model_name in available_models:
            metrics = all_metrics[model_name]
            
            # Try to find the metric
            value = None
            for possible_key in [metric_key, metric_key.lower(), metric_key.upper()]:
                if possible_key in metrics:
                    value = metrics[possible_key]
                    break
            
            if value is not None:
                values.append((model_name, value))
        
        if values:
            # Sort based on order
            values.sort(key=lambda x: x[1], reverse=(order == 'descending'))
            
            ranking_row = {
                'Metric': metric_key,
                '1st Place': f"{values[0][0]} ({values[0][1]:.2f})" if len(values) > 0 else 'N/A',
                '2nd Place': f"{values[1][0]} ({values[1][1]:.2f})" if len(values) > 1 else 'N/A',
                '3rd Place': f"{values[2][0]} ({values[2][1]:.2f})" if len(values) > 2 else 'N/A',
            }
            rankings.append(ranking_row)
    
    ranking_df = pd.DataFrame(rankings)
    
    # Save to CSV
    ranking_df.to_csv(output_dir / 'model_rankings.csv', index=False)
    print(f"✓ Saved rankings: {output_dir / 'model_rankings.csv'}")
    
    return ranking_df


if __name__ == "__main__":
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    output_dir = Path('results')
    if not output_dir.exists():
        print("\nError: No results directory found!")
        print("Train at least one model first.")
        exit(1)
    
    # Load metrics
    print("\n[1/5] Loading metrics...")
    all_metrics, available_models = load_all_metrics(output_dir)
    
    if len(available_models) < 2:
        print("\n⚠ Warning: Need at least 2 models to compare!")
        print("Train more models first.")
        exit(1)
    
    print(f"\n✓ Found {len(available_models)} models to compare")
    
    # Create comparison table
    print("\n[2/5] Creating comparison table...")
    comparison_df = create_comparison_table(all_metrics, available_models)
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"✓ Saved: {output_dir / 'model_comparison.csv'}")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Create metric comparison plots
    print("\n[3/5] Creating metric comparison plots...")
    plot_metric_comparison(all_metrics, available_models, output_dir)
    
    # Create forecast comparison
    print("\n[4/5] Creating forecast comparison...")
    plot_forecast_comparison(output_dir)
    
    # Create ranking table
    print("\n[5/5] Creating ranking table...")
    ranking_df = create_ranking_table(all_metrics, available_models, output_dir)
    
    print("\n" + "="*60)
    print("MODEL RANKINGS")
    print("="*60)
    print(ranking_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE! 🎉")
    print("="*60)
    
    print("\nGenerated files:")
    print(f"  📊 {output_dir / 'model_comparison.csv'} - Metrics table")
    print(f"  📊 {output_dir / 'model_comparison.png'} - Metric plots")
    print(f"  📊 {output_dir / 'all_forecasts_comparison.png'} - Forecast plots")
    print(f"  📊 {output_dir / 'model_rankings.csv'} - Rankings")

