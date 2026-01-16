"""
Precomputation utilities for the optimization problem.
Calculates distances, travel times, and other derived values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from functools import lru_cache

from data_loader import DataContainer
from config import config


@dataclass
class PrecomputedData:
    """Container for all precomputed values."""
    distance_matrix: np.ndarray  # STP to Farm distances
    travel_time_matrix: np.ndarray  # Travel times in hours
    stp_capacities: np.ndarray  # Daily capacity per STP
    farm_ids: List[str]  # Farm identifiers
    stp_ids: List[str]  # STP identifiers
    weather_factors: Dict[str, np.ndarray]  # Weather impact factors by date
    demand_by_farm_date: pd.DataFrame  # Pivoted demand data


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers.
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def compute_distance_matrix(
    stps: pd.DataFrame,
    farms: pd.DataFrame,
    stp_lat_col: str = 'lat',
    stp_lon_col: str = 'lon',
    farm_lat_col: str = 'lat',
    farm_lon_col: str = 'lon'
) -> np.ndarray:
    """
    Compute distance matrix between all STPs and farms.
    
    Args:
        stps: DataFrame with STP locations
        farms: DataFrame with farm locations
        
    Returns:
        2D numpy array of shape (n_stps, n_farms) with distances in km
    """
    n_stps = len(stps)
    n_farms = len(farms)
    
    distance_matrix = np.zeros((n_stps, n_farms))
    
    for i, stp in stps.iterrows():
        for j, farm in farms.iterrows():
            distance_matrix[i, j] = haversine_distance(
                stp[stp_lat_col], stp[stp_lon_col],
                farm[farm_lat_col], farm[farm_lon_col]
            )
    
    return distance_matrix


def compute_distance_dict(
    stps: pd.DataFrame,
    farms: pd.DataFrame,
    stp_id_col: str = 'stp_id',
    farm_id_col: str = 'farm_id',
    stp_lat_col: str = 'lat',
    stp_lon_col: str = 'lon',
    farm_lat_col: str = 'lat',
    farm_lon_col: str = 'lon'
) -> Dict[str, Dict[str, float]]:
    """
    Compute distance dictionary between all STPs and farms.
    
    Args:
        stps: DataFrame with STP locations (must have stp_id, lat, lon)
        farms: DataFrame with farm locations (must have farm_id, lat, lon)
        
    Returns:
        Nested dictionary: distances[stp_id][farm_id] = distance_km
        
    Example:
        >>> distances = compute_distance_dict(stp_registry, farm_locations)
        >>> distances['STP_TVM']['F_1000']  # Distance from STP_TVM to farm F_1000
        45.23
    """
    distances: Dict[str, Dict[str, float]] = {}
    
    for _, stp in stps.iterrows():
        stp_id = stp[stp_id_col]
        distances[stp_id] = {}
        
        for _, farm in farms.iterrows():
            farm_id = farm[farm_id_col]
            dist = haversine_distance(
                stp[stp_lat_col], stp[stp_lon_col],
                farm[farm_lat_col], farm[farm_lon_col]
            )
            distances[stp_id][farm_id] = dist
    
    return distances


def compute_rain_lock(
    farms: pd.DataFrame,
    weather: pd.DataFrame,
    farm_id_col: str = 'farm_id',
    zone_col: str = 'zone',
    date_col: str = 'date',
    threshold_mm: float = 30.0,
    forecast_window_days: int = 5
) -> Dict[str, Dict[str, bool]]:
    """
    Compute rain-lock status for each farm on each date.
    
    A farm-day is locked if the cumulative rainfall for the farm's weather zone
    over the current day and the next (forecast_window_days - 1) days exceeds
    the threshold.
    
    Args:
        farms: DataFrame with farm_id and zone columns
        weather: DataFrame with date and zone rainfall columns
        farm_id_col: Column name for farm ID
        zone_col: Column name for farm's weather zone
        date_col: Column name for date in weather data
        threshold_mm: Rainfall threshold in mm (default 30.0)
        forecast_window_days: Number of days to look ahead including current day (default 5)
        
    Returns:
        Nested dictionary: rain_lock[farm_id][date_str] -> True/False
        Where date_str is in YYYY-MM-DD format
        True means the farm is LOCKED (cannot deliver) on that date
        
    Example:
        >>> rain_lock = compute_rain_lock(farm_locations, daily_weather)
        >>> rain_lock['F_1000']['2025-06-15']  # Is F_1000 locked on June 15?
        True
    """
    # Build farm -> zone mapping
    farm_zone_map = dict(zip(farms[farm_id_col], farms[zone_col]))
    
    # Parse dates and sort weather data
    weather = weather.copy()
    weather[date_col] = pd.to_datetime(weather[date_col])
    weather = weather.sort_values(date_col).reset_index(drop=True)
    
    # Get all dates as strings
    all_dates = weather[date_col].tolist()
    date_strings = [d.strftime('%Y-%m-%d') for d in all_dates]
    
    # Get unique zones from weather columns (exclude date column)
    zone_columns = [col for col in weather.columns if col != date_col]
    
    # Precompute rolling sum for each zone (5-day forward window)
    zone_rolling_sums = {}
    for zone in zone_columns:
        # Create forward-looking rolling sum
        # We want sum of current day + next 4 days
        # Pandas rolling is backward-looking, so we reverse, roll, then reverse back
        values = weather[zone].values
        n = len(values)
        rolling_sum = np.zeros(n)
        
        for i in range(n):
            end_idx = min(i + forecast_window_days, n)
            rolling_sum[i] = np.sum(values[i:end_idx])
        
        zone_rolling_sums[zone] = rolling_sum
    
    # Build rain_lock dictionary
    rain_lock: Dict[str, Dict[str, bool]] = {}
    
    for farm_id in farms[farm_id_col]:
        zone = farm_zone_map[farm_id]
        rain_lock[farm_id] = {}
        
        if zone in zone_rolling_sums:
            rolling_sum = zone_rolling_sums[zone]
            for i, date_str in enumerate(date_strings):
                # Locked if cumulative rainfall exceeds threshold
                rain_lock[farm_id][date_str] = rolling_sum[i] > threshold_mm
        else:
            # If zone not found in weather data, assume not locked
            for date_str in date_strings:
                rain_lock[farm_id][date_str] = False
    
    return rain_lock


def compute_farm_active(
    planting_schedule: pd.DataFrame,
    all_farms: pd.DataFrame,
    weather: pd.DataFrame,
    farm_id_col: str = 'farm_id',
    plant_date_col: str = 'plant_date',
    harvest_date_col: str = 'harvest_date',
    date_col: str = 'date'
) -> Dict[str, Dict[str, bool]]:
    """
    Compute farm activity status for each farm on each date.
    
    A farm is active on a date if that date falls between any of its
    planting_date and harvest_date periods (inclusive). Farms can have
    multiple planting periods.
    
    Args:
        planting_schedule: DataFrame with farm_id, plant_date, harvest_date
        all_farms: DataFrame with all farm_ids (to include farms not in schedule)
        weather: DataFrame with date column (to get all dates in simulation)
        farm_id_col: Column name for farm ID
        plant_date_col: Column name for planting date
        harvest_date_col: Column name for harvest date
        date_col: Column name for date in weather data
        
    Returns:
        Nested dictionary: is_active[farm_id][date_str] -> True/False
        Where date_str is in YYYY-MM-DD format
        True means the farm is ACTIVE (has crops that need nitrogen)
        
    Example:
        >>> is_active = compute_farm_active(planting_schedule, farm_locations, daily_weather)
        >>> is_active['F_1000']['2025-06-15']  # Is F_1000 active on June 15?
        True
    """
    # Parse dates in planting schedule
    schedule = planting_schedule.copy()
    schedule[plant_date_col] = pd.to_datetime(schedule[plant_date_col])
    schedule[harvest_date_col] = pd.to_datetime(schedule[harvest_date_col])
    
    # Get all dates from weather data
    weather_copy = weather.copy()
    weather_copy[date_col] = pd.to_datetime(weather_copy[date_col])
    all_dates = sorted(weather_copy[date_col].unique())
    date_strings = [d.strftime('%Y-%m-%d') for d in all_dates]
    
    # Build farm -> list of (plant_date, harvest_date) mapping
    farm_periods: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for _, row in schedule.iterrows():
        farm_id = row[farm_id_col]
        plant_date = row[plant_date_col]
        harvest_date = row[harvest_date_col]
        
        if farm_id not in farm_periods:
            farm_periods[farm_id] = []
        farm_periods[farm_id].append((plant_date, harvest_date))
    
    # Get all farm IDs
    all_farm_ids = all_farms[farm_id_col].unique()
    
    # Build is_active dictionary
    is_active: Dict[str, Dict[str, bool]] = {}
    
    for farm_id in all_farm_ids:
        is_active[farm_id] = {}
        periods = farm_periods.get(farm_id, [])
        
        for i, date in enumerate(all_dates):
            date_str = date_strings[i]
            
            # Check if date falls within any planting period
            active = False
            for plant_date, harvest_date in periods:
                if plant_date <= date <= harvest_date:
                    active = True
                    break
            
            is_active[farm_id][date_str] = active
    
    return is_active


class STPStorageTracker:
    """
    Tracks daily storage levels for each STP.
    
    For each STP and each day:
    storage_today = storage_yesterday + incoming_biosolids - dispatched_biosolids
    
    If storage exceeds storage_max_tons, overflow is computed.
    Storage is never allowed to go negative.
    
    Attributes:
        stp_ids: List of STP identifiers
        daily_output: Dict mapping stp_id -> daily biosolid production (tons)
        storage_max: Dict mapping stp_id -> maximum storage capacity (tons)
        storage: Dict[stp_id][date_str] -> storage level at end of day
        overflow: Dict[stp_id][date_str] -> overflow amount (tons)
        dispatched: Dict[stp_id][date_str] -> dispatched amount (tons)
        
    Example:
        >>> tracker = STPStorageTracker(stp_registry, daily_weather)
        >>> tracker.dispatch('STP_TVM', '2025-01-15', 5.0)  # Dispatch 5 tons
        >>> tracker.simulate_day('2025-01-15')
        >>> print(tracker.storage['STP_TVM']['2025-01-15'])
    """
    
    def __init__(
        self,
        stp_registry: pd.DataFrame,
        weather: pd.DataFrame,
        stp_id_col: str = 'stp_id',
        daily_output_col: str = 'daily_output_tons',
        storage_max_col: str = 'storage_max_tons',
        date_col: str = 'date',
        initial_storage_fraction: float = 0.5
    ):
        """
        Initialize the storage tracker.
        
        Args:
            stp_registry: DataFrame with STP info
            weather: DataFrame with dates (to get simulation period)
            stp_id_col: Column name for STP ID
            daily_output_col: Column name for daily biosolid output
            storage_max_col: Column name for max storage capacity
            date_col: Column name for date
            initial_storage_fraction: Initial storage as fraction of max (default 0.5)
        """
        # Extract STP info
        self.stp_ids = stp_registry[stp_id_col].tolist()
        self.daily_output = dict(zip(
            stp_registry[stp_id_col], 
            stp_registry[daily_output_col]
        ))
        self.storage_max = dict(zip(
            stp_registry[stp_id_col], 
            stp_registry[storage_max_col]
        ))
        
        # Get all dates
        weather_copy = weather.copy()
        weather_copy[date_col] = pd.to_datetime(weather_copy[date_col])
        self.all_dates = sorted(weather_copy[date_col].unique())
        self.date_strings = [d.strftime('%Y-%m-%d') for d in self.all_dates]
        self.date_to_idx = {d: i for i, d in enumerate(self.date_strings)}
        
        # Initialize storage tracking dictionaries
        self.storage: Dict[str, Dict[str, float]] = {}
        self.overflow: Dict[str, Dict[str, float]] = {}
        self.dispatched: Dict[str, Dict[str, float]] = {}
        
        for stp_id in self.stp_ids:
            self.storage[stp_id] = {}
            self.overflow[stp_id] = {}
            self.dispatched[stp_id] = {}
            
            # Initialize all dates with zero dispatched
            for date_str in self.date_strings:
                self.dispatched[stp_id][date_str] = 0.0
                self.storage[stp_id][date_str] = 0.0
                self.overflow[stp_id][date_str] = 0.0
        
        # Set initial storage (before first day)
        self._initial_storage = {}
        for stp_id in self.stp_ids:
            self._initial_storage[stp_id] = (
                self.storage_max[stp_id] * initial_storage_fraction
            )
    
    def dispatch(self, stp_id: str, date_str: str, tons: float) -> None:
        """
        Record biosolids dispatched from an STP on a given date.
        
        Args:
            stp_id: STP identifier
            date_str: Date in YYYY-MM-DD format
            tons: Amount dispatched in tons
        """
        if stp_id in self.dispatched and date_str in self.dispatched[stp_id]:
            self.dispatched[stp_id][date_str] += tons
    
    def simulate_day(self, date_str: str) -> Dict[str, float]:
        """
        Simulate storage for a single day across all STPs.
        
        Args:
            date_str: Date to simulate in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping stp_id -> overflow amount for this day
        """
        day_overflow = {}
        date_idx = self.date_to_idx.get(date_str, -1)
        
        for stp_id in self.stp_ids:
            # Get previous day's storage
            if date_idx == 0:
                prev_storage = self._initial_storage[stp_id]
            else:
                prev_date = self.date_strings[date_idx - 1]
                prev_storage = self.storage[stp_id][prev_date]
            
            # Calculate today's storage
            incoming = self.daily_output[stp_id]
            outgoing = self.dispatched[stp_id][date_str]
            
            new_storage = prev_storage + incoming - outgoing
            
            # Cannot go negative
            new_storage = max(0.0, new_storage)
            
            # Check for overflow
            max_storage = self.storage_max[stp_id]
            if new_storage > max_storage:
                overflow = new_storage - max_storage
                new_storage = max_storage
            else:
                overflow = 0.0
            
            # Store results
            self.storage[stp_id][date_str] = new_storage
            self.overflow[stp_id][date_str] = overflow
            day_overflow[stp_id] = overflow
        
        return day_overflow
    
    def simulate_all(self) -> None:
        """Simulate storage for all days in sequence."""
        for date_str in self.date_strings:
            self.simulate_day(date_str)
    
    def get_total_overflow(self) -> Dict[str, float]:
        """Get total overflow per STP across all days."""
        totals = {}
        for stp_id in self.stp_ids:
            totals[stp_id] = sum(self.overflow[stp_id].values())
        return totals
    
    def get_available_storage(self, stp_id: str, date_str: str) -> float:
        """
        Get available storage capacity for an STP on a given date.
        
        This considers the storage at the start of the day (before incoming).
        """
        date_idx = self.date_to_idx.get(date_str, -1)
        
        if date_idx == 0:
            current = self._initial_storage[stp_id]
        else:
            prev_date = self.date_strings[date_idx - 1]
            current = self.storage[stp_id][prev_date]
        
        return self.storage_max[stp_id] - current
    
    def get_available_to_dispatch(self, stp_id: str, date_str: str) -> float:
        """
        Get amount available to dispatch from an STP on a given date.
        
        This is the storage at start of day plus today's incoming production,
        minus what has already been dispatched today.
        """
        date_idx = self.date_to_idx.get(date_str, -1)
        
        if date_idx == 0:
            prev_storage = self._initial_storage[stp_id]
        else:
            prev_date = self.date_strings[date_idx - 1]
            prev_storage = self.storage[stp_id][prev_date]
        
        incoming = self.daily_output[stp_id]
        already_dispatched = self.dispatched[stp_id][date_str]
        
        return prev_storage + incoming - already_dispatched
    
    def reset(self) -> None:
        """Reset all tracking to initial state."""
        for stp_id in self.stp_ids:
            for date_str in self.date_strings:
                self.dispatched[stp_id][date_str] = 0.0
                self.storage[stp_id][date_str] = 0.0
                self.overflow[stp_id][date_str] = 0.0
    
    def summary(self) -> str:
        """Return a summary string of storage tracking results."""
        lines = ["STP Storage Summary:", "=" * 40]
        
        for stp_id in self.stp_ids:
            total_overflow = sum(self.overflow[stp_id].values())
            total_dispatched = sum(self.dispatched[stp_id].values())
            max_storage = max(self.storage[stp_id].values()) if self.storage[stp_id] else 0
            
            lines.append(f"{stp_id}:")
            lines.append(f"  Max capacity: {self.storage_max[stp_id]} tons")
            lines.append(f"  Daily output: {self.daily_output[stp_id]} tons")
            lines.append(f"  Total dispatched: {total_dispatched:.1f} tons")
            lines.append(f"  Total overflow: {total_overflow:.1f} tons")
            lines.append(f"  Peak storage: {max_storage:.1f} tons")
        
        return "\n".join(lines)


def compute_travel_time_matrix(
    distance_matrix: np.ndarray,
    avg_speed_kmh: float = None
) -> np.ndarray:
    """
    Compute travel time matrix from distances.
    
    Args:
        distance_matrix: Distance matrix in km
        avg_speed_kmh: Average vehicle speed (from config if not provided)
        
    Returns:
        Travel time matrix in hours
    """
    if avg_speed_kmh is None:
        avg_speed_kmh = config.get('avg_vehicle_speed_kmh', 40.0)
    
    return distance_matrix / avg_speed_kmh


def compute_weather_factors(
    weather_df: pd.DataFrame,
    date_col: str = 'date'
) -> Dict[str, np.ndarray]:
    """
    Precompute weather impact factors for each day.
    
    Args:
        weather_df: Daily weather data
        
    Returns:
        Dictionary mapping dates to weather factor arrays
    """
    weather_factors = {}
    
    # Example: compute a delivery efficiency factor based on weather
    # Adjust this based on actual weather columns and problem requirements
    for _, row in weather_df.iterrows():
        date_key = str(row[date_col].date()) if hasattr(row[date_col], 'date') else str(row[date_col])
        
        # Example factor calculation - adjust based on actual requirements
        # This is a placeholder that should be customized
        efficiency = 1.0
        
        if 'rainfall' in row.index or 'rain' in row.index:
            rain_col = 'rainfall' if 'rainfall' in row.index else 'rain'
            rain = row[rain_col]
            if rain > 50:
                efficiency *= 0.5  # Heavy rain reduces efficiency
            elif rain > 20:
                efficiency *= 0.8  # Moderate rain
        
        weather_factors[date_key] = efficiency
    
    return weather_factors


def pivot_demand_data(
    demand_df: pd.DataFrame,
    farm_col: str = 'farm_id',
    date_col: str = 'date',
    demand_col: str = 'n_demand'
) -> pd.DataFrame:
    """
    Pivot demand data for easy lookup by farm and date.
    
    Args:
        demand_df: Daily demand data
        
    Returns:
        Pivoted DataFrame with farms as rows and dates as columns
    """
    return demand_df.pivot_table(
        index=farm_col,
        columns=date_col,
        values=demand_col,
        aggfunc='sum'
    ).fillna(0)


def precompute_all(data: DataContainer) -> PrecomputedData:
    """
    Perform all precomputations for the optimization.
    
    Args:
        data: Loaded data container
        
    Returns:
        PrecomputedData with all derived values
    """
    print("Starting precomputations...")
    
    # Compute distances
    print("  Computing distance matrix...")
    distance_matrix = compute_distance_matrix(
        data.stp_registry,
        data.farm_locations
    )
    
    # Compute travel times
    print("  Computing travel time matrix...")
    travel_time_matrix = compute_travel_time_matrix(distance_matrix)
    
    # Extract capacities
    print("  Extracting STP capacities...")
    capacity_col = config.get('capacity_column', 'daily_capacity')
    if capacity_col in data.stp_registry.columns:
        stp_capacities = data.stp_registry[capacity_col].values
    else:
        print(f"  Warning: {capacity_col} not found, using default capacity")
        stp_capacities = np.ones(len(data.stp_registry)) * 1000
    
    # Get IDs
    farm_id_col = config.get('farm_id_column', 'farm_id')
    stp_id_col = config.get('stp_id_column', 'stp_id')
    
    farm_ids = data.farm_locations[farm_id_col].tolist() if farm_id_col in data.farm_locations.columns else list(range(len(data.farm_locations)))
    stp_ids = data.stp_registry[stp_id_col].tolist() if stp_id_col in data.stp_registry.columns else list(range(len(data.stp_registry)))
    
    # Compute weather factors
    print("  Computing weather factors...")
    weather_factors = compute_weather_factors(data.daily_weather)
    
    # Pivot demand data
    print("  Pivoting demand data...")
    demand_by_farm_date = pivot_demand_data(data.daily_demand)
    
    precomputed = PrecomputedData(
        distance_matrix=distance_matrix,
        travel_time_matrix=travel_time_matrix,
        stp_capacities=stp_capacities,
        farm_ids=farm_ids,
        stp_ids=stp_ids,
        weather_factors=weather_factors,
        demand_by_farm_date=demand_by_farm_date
    )
    
    print("Precomputation complete!")
    print(f"  - Distance matrix shape: {distance_matrix.shape}")
    print(f"  - {len(stp_ids)} STPs, {len(farm_ids)} farms")
    print(f"  - {len(weather_factors)} days of weather data")
    
    return precomputed


if __name__ == "__main__":
    from data_loader import load_all_data
    
    try:
        data = load_all_data()
        precomputed = precompute_all(data)
        print(f"\nPrecomputed data ready for optimization!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
