"""
Data loading utilities for the carbon credit optimization problem.
Handles loading and initial preprocessing of all CSV data files.

Usage:
    from data_loader import load_all_data
    
    # Load from default location (./data)
    data = load_all_data()
    
    # Load from custom Kaggle input directory
    data = load_all_data("/kaggle/input/competition-name")
    
    # Load from any custom path
    data = load_all_data("C:/path/to/your/data")
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass

# Default data directory (relative to this file)
DEFAULT_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class DataContainer:
    """Container for all loaded datasets."""
    config: Dict[str, Any]
    stp_registry: pd.DataFrame
    farm_locations: pd.DataFrame
    daily_weather: pd.DataFrame
    daily_n_demand: pd.DataFrame
    planting_schedule: pd.DataFrame
    
    def __repr__(self) -> str:
        return (
            f"DataContainer(\n"
            f"  config: {len(self.config)} keys,\n"
            f"  stp_registry: {self.stp_registry.shape},\n"
            f"  farm_locations: {self.farm_locations.shape},\n"
            f"  daily_weather: {self.daily_weather.shape},\n"
            f"  daily_n_demand: {self.daily_n_demand.shape},\n"
            f"  planting_schedule: {self.planting_schedule.shape}\n"
            f")"
        )
    
    def print_summary(self):
        """Print detailed summary of all loaded data."""
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        
        # Config
        print("\n[CONFIG]")
        print(f"  Keys: {list(self.config.keys())}")
        
        # DataFrames
        datasets = [
            ("stp_registry", self.stp_registry),
            ("farm_locations", self.farm_locations),
            ("daily_weather", self.daily_weather),
            ("daily_n_demand", self.daily_n_demand),
            ("planting_schedule", self.planting_schedule),
        ]
        
        for name, df in datasets:
            print(f"\n[{name.upper()}]")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Dtypes:\n{df.dtypes.to_string().replace(chr(10), chr(10) + '    ')}")
            if not df.empty:
                print(f"  Sample (first 2 rows):")
                print(df.head(2).to_string().replace('\n', '\n    '))
        
        print("\n" + "=" * 60)


def load_config(base_dir: Path) -> Dict[str, Any]:
    """
    Load configuration from config.json file.
    
    Args:
        base_dir: Base directory containing config.json
        
    Returns:
        Dictionary containing all configuration parameters
    """
    config_path = base_dir / "config.json"
    
    if not config_path.exists():
        print(f"  Warning: config.json not found at {config_path}")
        print("  Using default empty configuration")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"  ✓ config.json: {len(config)} parameters")
    return config


def load_csv_safe(
    file_path: Path, 
    name: str,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Safely load a CSV file with error handling.
    
    Args:
        file_path: Path to the CSV file
        name: Name for logging
        parse_dates: Whether to auto-parse date columns
        
    Returns:
        Loaded DataFrame
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{name} not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Auto-parse date columns if requested
    if parse_dates:
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass  # Keep as-is if parsing fails
    
    print(f"  ✓ {name}: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def load_stp_registry(base_dir: Path) -> pd.DataFrame:
    """Load STP (Sewage Treatment Plant) registry data."""
    return load_csv_safe(base_dir / "stp_registry.csv", "stp_registry")


def load_farm_locations(base_dir: Path) -> pd.DataFrame:
    """Load farm locations data."""
    return load_csv_safe(base_dir / "farm_locations.csv", "farm_locations")


def load_daily_weather(base_dir: Path) -> pd.DataFrame:
    """Load daily weather data for 2025."""
    return load_csv_safe(base_dir / "daily_weather_2025.csv", "daily_weather_2025")


def load_daily_n_demand(base_dir: Path) -> pd.DataFrame:
    """Load daily nitrogen demand data."""
    return load_csv_safe(base_dir / "daily_n_demand.csv", "daily_n_demand")


def load_planting_schedule(base_dir: Path) -> pd.DataFrame:
    """Load planting schedule for 2025."""
    return load_csv_safe(base_dir / "planting_schedule_2025.csv", "planting_schedule_2025")


def load_all_data(base_dir: Union[str, Path] = None) -> DataContainer:
    """
    Load all datasets from a base directory.
    
    Args:
        base_dir: Directory containing the data files.
                  Can be:
                  - None: Uses default ./data directory
                  - String path: "/kaggle/input/competition-name"
                  - Path object
        
    Returns:
        DataContainer with all loaded datasets
        
    Example:
        # Default location
        data = load_all_data()
        
        # Kaggle input directory
        data = load_all_data("/kaggle/input/72-hr-hackathon-code-for-climate-action")
        
        # Custom path
        data = load_all_data("C:/Users/me/Downloads/data")
    """
    # Handle base_dir argument
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR
    else:
        base_dir = Path(base_dir)
    
    print(f"\nLoading data from: {base_dir}")
    print("-" * 40)
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {base_dir}")
    
    # List files in directory
    files = list(base_dir.glob("*"))
    print(f"Found {len(files)} files/folders:")
    for f in files[:10]:  # Show first 10
        print(f"  - {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
    print()
    
    # Load all datasets
    data = DataContainer(
        config=load_config(base_dir),
        stp_registry=load_stp_registry(base_dir),
        farm_locations=load_farm_locations(base_dir),
        daily_weather=load_daily_weather(base_dir),
        daily_n_demand=load_daily_n_demand(base_dir),
        planting_schedule=load_planting_schedule(base_dir)
    )
    
    print("-" * 40)
    print(f"All data loaded successfully!")
    
    return data


def validate_data(data: DataContainer) -> bool:
    """
    Validate loaded data for common issues.
    
    Args:
        data: DataContainer with loaded datasets
        
    Returns:
        True if validation passes
    """
    print("\nValidating data...")
    issues = []
    warnings = []
    
    datasets = [
        ("stp_registry", data.stp_registry),
        ("farm_locations", data.farm_locations),
        ("daily_weather", data.daily_weather),
        ("daily_n_demand", data.daily_n_demand),
        ("planting_schedule", data.planting_schedule),
    ]
    
    for name, df in datasets:
        # Check for empty dataframes
        if df.empty:
            issues.append(f"{name} is empty")
            continue
        
        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            cols_with_nulls = null_counts[null_counts > 0]
            for col, count in cols_with_nulls.items():
                pct = count / len(df) * 100
                if pct > 50:
                    issues.append(f"{name}.{col}: {count} nulls ({pct:.1f}%)")
                else:
                    warnings.append(f"{name}.{col}: {count} nulls ({pct:.1f}%)")
        
        # Check for duplicates if there's an ID column
        id_cols = [c for c in df.columns if c.endswith('_id') or c == 'id']
        for id_col in id_cols:
            dupes = df[id_col].duplicated().sum()
            if dupes > 0:
                warnings.append(f"{name}.{id_col}: {dupes} duplicate values")
    
    # Print results
    if issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("✓ All validation checks passed!")
    elif not issues:
        print("\n✓ No critical issues found (warnings only)")
    
    return len(issues) == 0


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    import sys
    
    # Allow passing base_dir as command line argument
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        print(f"Using custom base directory: {base_dir}")
    else:
        base_dir = None
        print("Using default data directory")
    
    try:
        # Load all data
        data = load_all_data(base_dir)
        
        # Print detailed summary
        data.print_summary()
        
        # Validate
        validate_data(data)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nExpected files:")
        print("  - config.json")
        print("  - stp_registry.csv")
        print("  - farm_locations.csv")
        print("  - daily_weather_2025.csv")
        print("  - daily_n_demand.csv")
        print("  - planting_schedule_2025.csv")
        sys.exit(1)
