"""
Load COVID-19 Data Utilities

This module provides functions to download and load COVID-19 data from various sources.
Data includes: cases, deaths, vaccines, and mobility reports.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def load_jhu_cases(data_dir: str = "data") -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 cases data from CSV.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        DataFrame with cases time series data
        
    Example:
        >>> cases_df = load_jhu_cases("data")
        >>> print(cases_df.shape)
    """
    filepath = Path(data_dir) / 'cases.csv'
    if not filepath.exists():
        raise FileNotFoundError(
            f"cases.csv not found in {data_dir}. "
            "Please ensure data files are present."
        )
    
    print(f"Loading cases data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_jhu_deaths(data_dir: str = "data") -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 deaths data from CSV.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        DataFrame with deaths time series data
        
    Example:
        >>> deaths_df = load_jhu_deaths("data")
        >>> print(deaths_df.shape)
    """
    filepath = Path(data_dir) / 'deaths.csv'
    if not filepath.exists():
        raise FileNotFoundError(
            f"deaths.csv not found in {data_dir}. "
            "Please ensure data files are present."
        )
    
    print(f"Loading deaths data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_jhu_vaccines(data_dir: str = "data") -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 vaccine data from CSV.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        DataFrame with vaccine time series data
        
    Example:
        >>> vaccines_df = load_jhu_vaccines("data")
        >>> print(vaccines_df.shape)
    """
    filepath = Path(data_dir) / 'vaccine.csv'
    if not filepath.exists():
        raise FileNotFoundError(
            f"vaccine.csv not found in {data_dir}. "
            "Please ensure data files are present."
        )
    
    print(f"Loading vaccine data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_google_mobility(data_dir: str = "data") -> pd.DataFrame:
    """
    Load Google COVID-19 Community Mobility Reports from CSV.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        DataFrame with mobility time series data
        
    Example:
        >>> mobility_df = load_google_mobility("data")
        >>> print(mobility_df.shape)
    """
    filepath = Path(data_dir) / 'mobility.csv'
    if not filepath.exists():
        raise FileNotFoundError(
            f"mobility.csv not found in {data_dir}. "
            "Please ensure data files are present."
        )
    
    print(f"Loading mobility data from {filepath}")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def load_all_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load all COVID-19 datasets at once.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Dictionary with keys: 'cases', 'deaths', 'vaccines', 'mobility'
        
    Example:
        >>> data = load_all_data("data")
        >>> cases_df = data['cases']
        >>> deaths_df = data['deaths']
    """
    print("Loading all COVID-19 datasets...")
    print("=" * 60)
    
    data = {}
    data['cases'] = load_jhu_cases(data_dir)
    data['deaths'] = load_jhu_deaths(data_dir)
    data['vaccines'] = load_jhu_vaccines(data_dir)
    data['mobility'] = load_google_mobility(data_dir)
    
    print("=" * 60)
    print("✓ All datasets loaded successfully!")
    return data


def verify_data_exists(data_dir: str = "data") -> bool:
    """
    Verify that all required data files exist.
    
    Args:
        data_dir: Directory to check for data files
        
    Returns:
        True if all files exist, False otherwise
        
    Example:
        >>> if verify_data_exists("data"):
        >>>     print("All data files present")
    """
    required_files = ['cases.csv', 'deaths.csv', 'vaccine.csv', 'mobility.csv']
    data_path = Path(data_dir)
    
    missing_files = []
    for filename in required_files:
        if not (data_path / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print(f"   Expected location: {data_path.absolute()}")
        return False
    
    print(f"✓ All required data files present in {data_dir}")
    return True


class DataLoader:
    """
    Convenience class for loading COVID-19 data.
    
    Example:
        >>> loader = DataLoader(data_dir="data")
        >>> cases_df = loader.load_cases()
        >>> deaths_df = loader.load_deaths()
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader with data directory."""
        self.data_dir = data_dir
    
    def load_cases(self) -> pd.DataFrame:
        """Load cases data."""
        return load_jhu_cases(self.data_dir)
    
    def load_deaths(self) -> pd.DataFrame:
        """Load deaths data."""
        return load_jhu_deaths(self.data_dir)
    
    def load_vaccines(self) -> pd.DataFrame:
        """Load vaccine data."""
        return load_jhu_vaccines(self.data_dir)
    
    def load_mobility(self) -> pd.DataFrame:
        """Load mobility data."""
        return load_google_mobility(self.data_dir)
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets."""
        return load_all_data(self.data_dir)

