"""
CSV export utilities for the solution submission.
Generates the solution.csv file in the required format.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from optimizer import Schedule
from config import OUTPUT_DIR


def ensure_output_dir(output_dir: Path = OUTPUT_DIR):
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_solution(
    schedule: Schedule,
    output_path: Optional[Path] = None,
    columns: Optional[List[str]] = None,
    date_format: str = '%Y-%m-%d'
) -> Path:
    """
    Export the schedule to solution.csv in the competition format.
    
    Args:
        schedule: The optimized delivery schedule
        output_path: Path for output file (default: output/solution.csv)
        columns: Columns to include in output (default: all)
        date_format: Format for date columns
        
    Returns:
        Path to the exported file
    """
    if output_path is None:
        ensure_output_dir()
        output_path = OUTPUT_DIR / "solution.csv"
    
    df = schedule.to_dataframe()
    
    if df.empty:
        print("Warning: Exporting empty schedule")
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['date', 'stp_id', 'farm_id', 'quantity'])
    
    # Format date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime(date_format)
    
    # Select specific columns if requested
    if columns is not None:
        df = df[columns]
    
    # Export to CSV
    df.to_csv(output_path, index=False)
    print(f"Solution exported to: {output_path}")
    print(f"  - {len(df)} rows")
    print(f"  - Columns: {list(df.columns)}")
    
    return output_path


def export_deliveries_to_csv(
    deliveries: List[tuple],
    output_path: Optional[Path] = None,
    date_format: str = '%Y-%m-%d'
) -> Path:
    """
    Convert delivery list to DataFrame and save as solution.csv.
    
    Args:
        deliveries: List of tuples (date, stp_id, farm_id, tons_delivered)
        output_path: Path for output file (default: output/solution.csv)
        date_format: Format for date column
        
    Returns:
        Path to the exported file
        
    Example:
        >>> deliveries = optimizer.optimize()
        >>> path = export_deliveries_to_csv(deliveries)
        >>> print(f"Saved to: {path}")
    """
    if output_path is None:
        ensure_output_dir()
        output_path = OUTPUT_DIR / "solution.csv"
    
    # Convert to DataFrame with correct column names
    df = pd.DataFrame(
        deliveries,
        columns=['date', 'stp_id', 'farm_id', 'tons_delivered']
    )
    
    if df.empty:
        print("Warning: Exporting empty delivery list")
    else:
        # Ensure date column is properly formatted
        df['date'] = pd.to_datetime(df['date']).dt.strftime(date_format)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
    
    # Save as CSV with no index
    df.to_csv(output_path, index=False)
    
    print(f"Solution exported to: {output_path}")
    print(f"  - {len(df)} deliveries")
    print(f"  - Columns: {list(df.columns)}")
    
    if not df.empty:
        total_tons = df['tons_delivered'].sum()
        print(f"  - Total biosolids: {total_tons:.2f} tons")
    
    return output_path


def export_with_timestamp(
    schedule: Schedule,
    prefix: str = "solution",
    output_dir: Path = OUTPUT_DIR
) -> Path:
    """
    Export solution with timestamp for versioning.
    
    Args:
        schedule: The optimized delivery schedule
        prefix: Filename prefix
        output_dir: Output directory
        
    Returns:
        Path to the exported file
    """
    ensure_output_dir(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{timestamp}.csv"
    output_path = output_dir / filename
    
    return export_solution(schedule, output_path)


def export_summary(
    schedule: Schedule,
    score_breakdown: dict = None,
    output_path: Optional[Path] = None
) -> Path:
    """
    Export a summary of the solution for review.
    
    Args:
        schedule: The delivery schedule
        score_breakdown: Score components (optional)
        output_path: Path for summary file
        
    Returns:
        Path to the summary file
    """
    if output_path is None:
        ensure_output_dir()
        output_path = OUTPUT_DIR / "solution_summary.txt"
    
    df = schedule.to_dataframe()
    
    lines = [
        "=" * 60,
        "SOLUTION SUMMARY",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "SCHEDULE STATISTICS",
        "-" * 40,
        f"Total deliveries: {len(df)}",
    ]
    
    if not df.empty:
        lines.extend([
            f"Total quantity: {df['quantity'].sum():.2f}",
            f"Total distance: {df['distance'].sum():.2f} km",
            f"Avg distance per delivery: {df['distance'].mean():.2f} km",
            f"Unique dates: {df['date'].nunique()}",
            f"Unique STPs used: {df['stp_id'].nunique()}",
            f"Unique farms served: {df['farm_id'].nunique()}",
            "",
            "DAILY BREAKDOWN",
            "-" * 40,
        ])
        
        daily_stats = df.groupby('date').agg({
            'quantity': 'sum',
            'distance': 'sum',
            'stp_id': 'count'
        }).rename(columns={'stp_id': 'deliveries'})
        
        for date, row in daily_stats.iterrows():
            lines.append(
                f"  {date}: {row['deliveries']} deliveries, "
                f"{row['quantity']:.0f} qty, {row['distance']:.1f} km"
            )
    
    if score_breakdown:
        lines.extend([
            "",
            "SCORE BREAKDOWN",
            "-" * 40,
        ])
        for key, value in score_breakdown.items():
            lines.append(f"  {key}: {value:.2f}")
    
    lines.append("=" * 60)
    
    content = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Summary exported to: {output_path}")
    return output_path


def validate_submission_format(
    csv_path: Path,
    required_columns: List[str] = None
) -> bool:
    """
    Validate that a CSV file meets submission requirements.
    
    Args:
        csv_path: Path to the CSV file
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if required_columns is None:
        required_columns = ['date', 'stp_id', 'farm_id', 'quantity']
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    errors = []
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for empty file
    if df.empty:
        errors.append("CSV file is empty")
    
    # Check for missing values in required columns
    for col in required_columns:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} missing values")
    
    # Check quantity is numeric and positive
    if 'quantity' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['quantity']):
            errors.append("'quantity' column is not numeric")
        elif (df['quantity'] <= 0).any():
            errors.append("'quantity' contains non-positive values")
    
    if errors:
        print("Submission validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Submission format validation passed!")
    print(f"  - {len(df)} rows, {len(df.columns)} columns")
    return True


if __name__ == "__main__":
    from data_loader import load_all_data
    from precompute import precompute_all
    from optimizer import DeliveryOptimizer
    
    try:
        data = load_all_data()
        precomputed = precompute_all(data)
        
        optimizer = DeliveryOptimizer(precomputed)
        schedule = optimizer.optimize(strategy='greedy')
        
        # Export solution
        solution_path = export_solution(schedule)
        
        # Export summary
        export_summary(schedule)
        
        # Validate submission
        validate_submission_format(solution_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
