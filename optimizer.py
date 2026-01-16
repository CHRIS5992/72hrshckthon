"""
Delivery optimization logic for maximizing carbon credit score.
Contains the core scheduling algorithms and constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq

from precompute import PrecomputedData
from config import config


@dataclass
class Delivery:
    """Represents a single delivery assignment."""
    date: str
    stp_id: str
    farm_id: str
    quantity: float
    travel_time: float = 0.0
    distance: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date,
            'stp_id': self.stp_id,
            'farm_id': self.farm_id,
            'quantity': self.quantity,
            'travel_time': self.travel_time,
            'distance': self.distance
        }


@dataclass
class Schedule:
    """Complete delivery schedule."""
    deliveries: List[Delivery] = field(default_factory=list)
    
    def add_delivery(self, delivery: Delivery):
        self.deliveries.append(delivery)
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.deliveries:
            return pd.DataFrame()
        return pd.DataFrame([d.to_dict() for d in self.deliveries])
    
    def __len__(self) -> int:
        return len(self.deliveries)


class DeliveryOptimizer:
    """
    Main optimizer class for generating delivery schedules.
    """
    
    def __init__(self, precomputed: PrecomputedData):
        self.precomputed = precomputed
        self.distance_matrix = precomputed.distance_matrix
        self.travel_time_matrix = precomputed.travel_time_matrix
        self.stp_capacities = precomputed.stp_capacities
        self.farm_ids = precomputed.farm_ids
        self.stp_ids = precomputed.stp_ids
        self.weather_factors = precomputed.weather_factors
        self.demand_by_farm_date = precomputed.demand_by_farm_date
        
        # Get constraints from config
        self.max_daily_hours = config.get('max_daily_hours', 8)
        self.min_delivery_qty = config.get('min_delivery_quantity', 0)
        self.max_delivery_qty = config.get('max_delivery_quantity', float('inf'))
    
    def get_demand(self, farm_id: str, date: str) -> float:
        """Get demand for a farm on a specific date."""
        try:
            return self.demand_by_farm_date.loc[farm_id, date]
        except KeyError:
            return 0.0
    
    def get_distance(self, stp_idx: int, farm_idx: int) -> float:
        """Get distance between STP and farm."""
        return self.distance_matrix[stp_idx, farm_idx]
    
    def get_travel_time(self, stp_idx: int, farm_idx: int) -> float:
        """Get travel time between STP and farm."""
        return self.travel_time_matrix[stp_idx, farm_idx]
    
    def get_weather_efficiency(self, date: str) -> float:
        """Get weather efficiency factor for a date."""
        return self.weather_factors.get(date, 1.0)
    
    def optimize_greedy(self, dates: List[str] = None) -> Schedule:
        """
        Greedy optimization: assign closest STP to each farm demand.
        
        Args:
            dates: List of dates to optimize. Uses all dates if None.
            
        Returns:
            Optimized Schedule
        """
        schedule = Schedule()
        
        if dates is None:
            dates = [str(d) for d in self.demand_by_farm_date.columns]
        
        for date in dates:
            # Track remaining capacity per STP for this day
            remaining_capacity = self.stp_capacities.copy()
            
            # Get demands for all farms on this date
            farm_demands = []
            for farm_idx, farm_id in enumerate(self.farm_ids):
                demand = self.get_demand(farm_id, date)
                if demand > 0:
                    farm_demands.append((farm_idx, farm_id, demand))
            
            # Sort farms by demand (largest first) for better allocation
            farm_demands.sort(key=lambda x: -x[2])
            
            for farm_idx, farm_id, demand in farm_demands:
                remaining_demand = demand
                
                # Find STPs sorted by distance to this farm
                stp_distances = [(self.get_distance(i, farm_idx), i) 
                                for i in range(len(self.stp_ids))]
                stp_distances.sort()
                
                for distance, stp_idx in stp_distances:
                    if remaining_demand <= 0:
                        break
                    
                    if remaining_capacity[stp_idx] <= 0:
                        continue
                    
                    # Calculate how much can be delivered
                    quantity = min(remaining_demand, remaining_capacity[stp_idx])
                    
                    # Apply constraints
                    if quantity < self.min_delivery_qty:
                        continue
                    if quantity > self.max_delivery_qty:
                        quantity = self.max_delivery_qty
                    
                    # Create delivery
                    delivery = Delivery(
                        date=date,
                        stp_id=self.stp_ids[stp_idx],
                        farm_id=farm_id,
                        quantity=quantity,
                        travel_time=self.get_travel_time(stp_idx, farm_idx),
                        distance=distance
                    )
                    
                    schedule.add_delivery(delivery)
                    remaining_demand -= quantity
                    remaining_capacity[stp_idx] -= quantity
        
        return schedule
    
    def optimize_balanced(self, dates: List[str] = None) -> Schedule:
        """
        Balanced optimization: distribute load evenly across STPs while 
        minimizing total distance.
        
        Args:
            dates: List of dates to optimize. Uses all dates if None.
            
        Returns:
            Optimized Schedule
        """
        schedule = Schedule()
        
        if dates is None:
            dates = [str(d) for d in self.demand_by_farm_date.columns]
        
        for date in dates:
            remaining_capacity = self.stp_capacities.copy()
            
            # Build priority queue of (cost, stp_idx, farm_idx, quantity)
            # Cost considers both distance and capacity utilization
            assignments = []
            
            for farm_idx, farm_id in enumerate(self.farm_ids):
                demand = self.get_demand(farm_id, date)
                if demand <= 0:
                    continue
                
                for stp_idx in range(len(self.stp_ids)):
                    distance = self.get_distance(stp_idx, farm_idx)
                    capacity = self.stp_capacities[stp_idx]
                    
                    # Cost function: balance distance with utilization
                    # Lower cost = better assignment
                    cost = distance / (capacity + 1)  # +1 to avoid division by zero
                    
                    heapq.heappush(assignments, (cost, stp_idx, farm_idx, farm_id, demand))
            
            # Track fulfilled demand per farm
            fulfilled = {farm_id: 0.0 for farm_id in self.farm_ids}
            
            while assignments:
                cost, stp_idx, farm_idx, farm_id, original_demand = heapq.heappop(assignments)
                
                remaining_demand = original_demand - fulfilled[farm_id]
                if remaining_demand <= 0:
                    continue
                
                if remaining_capacity[stp_idx] <= 0:
                    continue
                
                quantity = min(remaining_demand, remaining_capacity[stp_idx])
                
                if quantity >= self.min_delivery_qty:
                    quantity = min(quantity, self.max_delivery_qty)
                    
                    delivery = Delivery(
                        date=date,
                        stp_id=self.stp_ids[stp_idx],
                        farm_id=farm_id,
                        quantity=quantity,
                        travel_time=self.get_travel_time(stp_idx, farm_idx),
                        distance=self.get_distance(stp_idx, farm_idx)
                    )
                    
                    schedule.add_delivery(delivery)
                    fulfilled[farm_id] += quantity
                    remaining_capacity[stp_idx] -= quantity
        
        return schedule
    
    def optimize(self, strategy: str = 'greedy', **kwargs) -> Schedule:
        """
        Run optimization with specified strategy.
        
        Args:
            strategy: 'greedy' or 'balanced'
            **kwargs: Additional arguments for the strategy
            
        Returns:
            Optimized Schedule
        """
        strategies = {
            'greedy': self.optimize_greedy,
            'balanced': self.optimize_balanced,
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
        
        print(f"Running {strategy} optimization...")
        schedule = strategies[strategy](**kwargs)
        print(f"Generated {len(schedule)} deliveries")
        
        return schedule


class GreedyDeliveryOptimizer:
    """
    Greedy delivery optimizer that considers all constraints.
    
    For each day in 2025, for each STP:
    - Determine available biosolids (storage + daily input)
    - Identify candidate farms (active, not rain-locked, demand > 0)
    - Sort by distance (nearest first)
    - Deliver in trucks of max 10 tons
    - Do NOT exceed daily nitrogen demand + 10% buffer
    - Track STP storage and overflow
    
    Output: List of (date, stp_id, farm_id, tons_delivered)
    """
    
    def __init__(
        self,
        stp_registry: pd.DataFrame,
        farm_locations: pd.DataFrame,
        daily_weather: pd.DataFrame,
        daily_n_demand: pd.DataFrame,
        planting_schedule: pd.DataFrame,
        distances: Dict[str, Dict[str, float]],
        rain_lock: Dict[str, Dict[str, bool]],
        is_active: Dict[str, Dict[str, bool]],
        config_dict: Dict = None
    ):
        """
        Initialize the greedy optimizer.
        
        Args:
            stp_registry: STP data (stp_id, daily_output_tons, storage_max_tons, lat, lon)
            farm_locations: Farm locations (farm_id, zone, area_ha, lat, lon)
            daily_weather: Weather data (date, zone columns with rainfall)
            daily_n_demand: Nitrogen demand (date as rows, farm_ids as columns)
            planting_schedule: Planting dates (farm_id, crop, plant_date, harvest_date)
            distances: Precomputed distances[stp_id][farm_id] in km
            rain_lock: Precomputed rain_lock[farm_id][date] -> bool
            is_active: Precomputed is_active[farm_id][date] -> bool
            config_dict: Optional config overrides
        """
        self.stp_registry = stp_registry
        self.farm_locations = farm_locations
        self.daily_weather = daily_weather
        self.daily_n_demand = daily_n_demand
        self.planting_schedule = planting_schedule
        self.distances = distances
        self.rain_lock = rain_lock
        self.is_active = is_active
        
        # Config with defaults
        cfg = config_dict or {}
        self.truck_capacity = cfg.get('truck_capacity_tons', 10)
        self.n_per_ton_biosolid = cfg.get('nitrogen_content_kg_per_ton_biosolid', 25)
        self.buffer_percent = cfg.get('application_buffer_percent', 10)
        
        # Get STP info
        self.stp_ids = stp_registry['stp_id'].tolist()
        self.farm_ids = farm_locations['farm_id'].tolist()
        
        # Parse demand data
        self._parse_demand_data()
        
        # Get all dates from weather
        weather_copy = daily_weather.copy()
        weather_copy['date'] = pd.to_datetime(weather_copy['date'])
        self.all_dates = sorted(weather_copy['date'].unique())
        self.date_strings = [d.strftime('%Y-%m-%d') for d in self.all_dates]
        
        # Track nitrogen delivered to each farm per day
        self.n_delivered: Dict[str, Dict[str, float]] = {}
        for farm_id in self.farm_ids:
            self.n_delivered[farm_id] = {d: 0.0 for d in self.date_strings}
    
    def _parse_demand_data(self):
        """Parse nitrogen demand data into lookup format."""
        # demand_data[farm_id][date_str] -> kg N demand
        self.demand_data: Dict[str, Dict[str, float]] = {}
        
        demand_df = self.daily_n_demand.copy()
        demand_df['date'] = pd.to_datetime(demand_df['date'])
        
        # Get all farm columns (everything except 'date')
        farm_cols = [c for c in demand_df.columns if c != 'date']
        
        for farm_id in farm_cols:
            self.demand_data[farm_id] = {}
            for _, row in demand_df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                self.demand_data[farm_id][date_str] = row[farm_id]
    
    def get_demand(self, farm_id: str, date_str: str) -> float:
        """Get nitrogen demand for a farm on a date (kg N)."""
        if farm_id in self.demand_data:
            return self.demand_data[farm_id].get(date_str, 0.0)
        return 0.0
    
    def get_remaining_demand(self, farm_id: str, date_str: str) -> float:
        """Get remaining nitrogen that can be delivered (with buffer)."""
        base_demand = self.get_demand(farm_id, date_str)
        max_demand = base_demand * (1 + self.buffer_percent / 100)
        already_delivered = self.n_delivered[farm_id][date_str]
        return max(0, max_demand - already_delivered)
    
    def is_candidate(self, farm_id: str, date_str: str) -> bool:
        """Check if farm is a valid candidate for delivery."""
        # Must be active (crop growing)
        if not self.is_active.get(farm_id, {}).get(date_str, False):
            return False
        
        # Must not be rain-locked
        if self.rain_lock.get(farm_id, {}).get(date_str, False):
            return False
        
        # Must have remaining demand
        if self.get_remaining_demand(farm_id, date_str) <= 0:
            return False
        
        return True
    
    def optimize(self) -> List[Tuple[str, str, str, float]]:
        """
        Run the greedy optimization.
        
        Algorithm:
        1. For each day:
           a. Collect all valid (STP, Farm) candidates where:
              - STP has stock
              - Farm needs N
              - Farm is active & not rain-locked
           b. Sort ALL candidates by distance (global nearest-neighbor)
           c. Assign deliveries in order until constraints are met
           d. Update storage and demand
        
        Returns:
            List of deliveries as tuples: (date, stp_id, farm_id, tons_delivered)
        """
        from precompute import STPStorageTracker
        
        # Initialize storage tracker
        tracker = STPStorageTracker(self.stp_registry, self.daily_weather)
        
        deliveries: List[Tuple[str, str, str, float]] = []
        
        # Process each day
        for date_str in self.date_strings:
            
            # 1. Identify all potential routes for this day
            potential_routes = []
            
            # Helper to cache remaining demand to avoid repeated lookups
            farm_remaining_demand = {}
            for farm_id in self.farm_ids:
                if self.is_candidate(farm_id, date_str):
                    farm_remaining_demand[farm_id] = self.get_remaining_demand(farm_id, date_str)
            
            # Only look at STPs that have something to give
            active_stps = {}
            for stp_id in self.stp_ids:
                available = tracker.get_available_to_dispatch(stp_id, date_str)
                if available > 0:
                    active_stps[stp_id] = available
            
            # Generate all valid pair combinations
            for stp_id, available_tons in active_stps.items():
                for farm_id, demand_n in farm_remaining_demand.items():
                    if demand_n > 0:
                        dist = self.distances[stp_id][farm_id]
                        potential_routes.append((dist, stp_id, farm_id))
            
            # 2. Sort globally by distance (nearest deliveries first)
            potential_routes.sort(key=lambda x: x[0])
            
            # 3. Fulfill deliveries
            for _, stp_id, farm_id in potential_routes:
                # Check current status (as it might have changed in this loop)
                available = active_stps.get(stp_id, 0.0)
                remaining_demand_n = farm_remaining_demand.get(farm_id, 0.0)
                
                if available <= 0 or remaining_demand_n <= 0:
                    continue
                
                # Convert remaining N demand to biosolid tons
                max_tons_for_demand = remaining_demand_n / self.n_per_ton_biosolid
                
                # Determine tons to deliver
                tons_to_deliver = min(
                    available,
                    max_tons_for_demand,
                    self.truck_capacity
                )
                
                # Round to reasonable precision
                tons_to_deliver = round(tons_to_deliver, 3)
                
                if tons_to_deliver <= 0.001:  # Skip negligible amounts
                    continue
                
                # Record delivery
                deliveries.append((date_str, stp_id, farm_id, tons_to_deliver))
                
                # Update our local trackers for this day
                available -= tons_to_deliver
                active_stps[stp_id] = available
                
                n_delivered = tons_to_deliver * self.n_per_ton_biosolid
                remaining_demand_n -= n_delivered
                farm_remaining_demand[farm_id] = remaining_demand_n
                
                # Update the persistent trackers
                tracker.dispatch(stp_id, date_str, tons_to_deliver)
                self.n_delivered[farm_id][date_str] += n_delivered
            
            # 4. Simulate storage at end of day (handling overflow/carry-over)
            tracker.simulate_day(date_str)
        
        # Store tracker for later scoring
        self.storage_tracker = tracker
        
        return deliveries
    
    def get_deliveries_dataframe(self, deliveries: List[Tuple]) -> pd.DataFrame:
        """Convert deliveries list to DataFrame."""
        return pd.DataFrame(deliveries, columns=['date', 'stp_id', 'farm_id', 'tons_delivered'])
    
    def summary(self, deliveries: List[Tuple]) -> str:
        """Generate summary of optimization results."""
        df = self.get_deliveries_dataframe(deliveries)
        
        total_tons = df['tons_delivered'].sum()
        total_deliveries = len(df)
        unique_days = df['date'].nunique()
        
        lines = [
            "Greedy Optimizer Summary",
            "=" * 40,
            f"Total deliveries: {total_deliveries}",
            f"Total biosolids delivered: {total_tons:.1f} tons",
            f"Active days: {unique_days}",
            "",
            "By STP:",
        ]
        
        for stp_id in self.stp_ids:
            stp_tons = df[df['stp_id'] == stp_id]['tons_delivered'].sum()
            stp_count = len(df[df['stp_id'] == stp_id])
            lines.append(f"  {stp_id}: {stp_tons:.1f} tons ({stp_count} deliveries)")
        
        if hasattr(self, 'storage_tracker'):
            lines.append("")
            lines.append("Overflow:")
            for stp_id in self.stp_ids:
                overflow = sum(self.storage_tracker.overflow[stp_id].values())
                lines.append(f"  {stp_id}: {overflow:.1f} tons")
        
        return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import load_all_data
    from precompute import precompute_all
    
    try:
        data = load_all_data()
        precomputed = precompute_all(data)
        
        optimizer = DeliveryOptimizer(precomputed)
        schedule = optimizer.optimize(strategy='greedy')
        
        print(f"\nSchedule preview:")
        print(schedule.to_dataframe().head(10))
    except FileNotFoundError as e:
        print(f"Error: {e}")
