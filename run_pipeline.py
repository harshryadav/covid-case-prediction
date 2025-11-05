#!/usr/bin/env python3
"""
COVID-19 Forecasting Pipeline Runner

This script runs the complete forecasting pipeline from data loading to model training.
"""

import sys
import argparse
from pathlib import Path


def check_data_exists():
    """Check if required data files exist"""
    data_dir = Path('data')
    required_files = [
        'time_series_covid19_confirmed_US.csv',
        'time_series_covid19_deaths_US.csv',
        'mobility_report_US.csv'
    ]
    
    missing = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing required data files:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    print("✅ All required data files found")
    return True


def run_preprocessing():
    """Run data preprocessing"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/data_processing/preprocess.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ Preprocessing failed!")
        return False
    
    print("✅ Preprocessing complete")
    return True


def run_gluonts_preparation():
    """Prepare data for GluonTS"""
    print("\n" + "="*60)
    print("STEP 2: GLUONTS DATA PREPARATION")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/data_processing/prepare_gluonts.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ GluonTS preparation failed!")
        return False
    
    print("✅ GluonTS data ready")
    return True


def run_baseline_models():
    """Train baseline models"""
    print("\n" + "="*60)
    print("STEP 3: BASELINE MODELS")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/models/train_baseline.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ Baseline training failed!")
        return False
    
    print("✅ Baseline models trained")
    return True


def run_deepar_model():
    """Train DeepAR model"""
    print("\n" + "="*60)
    print("STEP 4: DEEPAR MODEL")
    print("="*60)
    
    # Check if GluonTS is installed
    try:
        import gluonts
    except ImportError:
        print("⚠️  GluonTS not installed. Skipping DeepAR training.")
        print("    Install with: pip install gluonts torch lightning")
        return True  # Not a failure, just skipped
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/models/train_deepar.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ DeepAR training failed!")
        return False
    
    print("✅ DeepAR model trained")
    return True


def run_tft_model():
    """Train Temporal Fusion Transformer model"""
    print("\n" + "="*60)
    print("STEP 5: TEMPORAL FUSION TRANSFORMER (TFT)")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/models/train_tft.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ TFT training failed!")
        return False
    
    print("✅ TFT model trained")
    return True


def run_prophet_model():
    """Train Prophet model"""
    print("\n" + "="*60)
    print("STEP 6: PROPHET MODEL")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/models/train_prophet.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ Prophet training failed!")
        return False
    
    print("✅ Prophet model trained")
    return True


def run_deepvar_model():
    """Train DeepVAR multivariate model"""
    print("\n" + "="*60)
    print("STEP 7: DEEPVAR MODEL (Multivariate)")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/models/train_deepvar.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ DeepVAR training failed!")
        return False
    
    print("✅ DeepVAR model trained")
    return True


def run_model_comparison():
    """Compare all trained models"""
    print("\n" + "="*60)
    print("STEP 8: MODEL COMPARISON")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/models/compare_models.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("⚠️  Model comparison failed (but models trained)")
        return True  # Don't fail entire pipeline
    
    print("✅ Model comparison complete")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run COVID-19 forecasting pipeline')
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['preprocess', 'gluonts', 'baseline', 'deepar', 'tft', 'prophet', 'deepvar', 'compare', 'all'],
        default=['all'],
        help='Which steps to run (default: all)'
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip data existence check'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: only baseline and DeepAR'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("COVID-19 FORECASTING PIPELINE")
    print("="*60)
    
    # Check data exists
    if not args.skip_check:
        if not check_data_exists():
            print("\n❌ Pipeline aborted: missing data files")
            return 1
    
    # Determine which steps to run
    steps = args.steps
    if args.quick:
        steps = ['preprocess', 'gluonts', 'baseline', 'deepar', 'compare']
        print("\n🚀 Quick mode: Running baseline + DeepAR only")
    elif 'all' in steps:
        steps = ['preprocess', 'gluonts', 'baseline', 'deepar', 'tft', 'prophet', 'deepvar', 'compare']
    
    # Run steps
    success = True
    
    if 'preprocess' in steps:
        if not run_preprocessing():
            success = False
    
    if success and 'gluonts' in steps:
        if not run_gluonts_preparation():
            success = False
    
    if success and 'baseline' in steps:
        if not run_baseline_models():
            success = False
    
    if success and 'deepar' in steps:
        if not run_deepar_model():
            success = False
    
    if success and 'tft' in steps:
        if not run_tft_model():
            success = False
    
    if success and 'prophet' in steps:
        if not run_prophet_model():
            success = False
    
    if success and 'deepvar' in steps:
        if not run_deepvar_model():
            success = False
    
    if success and 'compare' in steps:
        run_model_comparison()  # Don't fail pipeline if comparison fails
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        print("\nResults saved to:")
        print("  - results/baseline_forecasts.png")
        print("  - results/baseline_metrics.csv")
        print("  - results/deepar_forecast.png")
        print("  - results/tft_forecast.png")
        print("  - results/prophet_forecast.png")
        print("  - results/deepvar_forecast.png")
        print("  - results/model_comparison.csv")
        print("  - results/model_comparison.png")
        print("\nNext steps:")
        print("  1. View comparison: open results/model_comparison.png")
        print("  2. Review rankings: cat results/model_rankings.csv")
        print("  3. Experiment with hyperparameters")
        return 0
    else:
        print("❌ PIPELINE FAILED")
        print("="*60)
        print("\nCheck error messages above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())

