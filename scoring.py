"""
Scoring function for evaluating delivery schedules.
Calculates the carbon credit score based on the optimization objective.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from optimizer import Schedule, Delivery
from precompute import PrecomputedData
from config import config


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of the score components."""
    total_score: float
    demand_satisfaction_score: float
    distance_penalty: float
    efficiency_bonus: float
    utilization_score: float
    penalty_score: float
    
    def __repr__(self) -> str:
        return (
            f"ScoreBreakdown:\n"
            f"  Total Score: {self.total_score:.2f}\n"
            f"  ├─ Demand Satisfaction: {self.demand_satisfaction_score:.2f}\n"
            f"  ├─ Distance Penalty: -{self.distance_penalty:.2f}\n"
            f"  ├─ Efficiency Bonus: +{self.efficiency_bonus:.2f}\n"
            f"  ├─ Utilization Score: {self.utilization_score:.2f}\n"
            f"  └─ Penalties: -{self.penalty_score:.2f}"
        )


class Scorer:
    """
    Calculates carbon credit score for a delivery schedule.
    
    The scoring function should match the Kaggle competition objective.
    Adjust the weights and calculations based on the actual scoring rules.
    """
    
    def __init__(self, precomputed: PrecomputedData):
        self.precomputed = precomputed
        
        # Load scoring weights from config
        self.demand_weight = config.get('score_demand_weight', 1.0)
        self.distance_weight = config.get('score_distance_weight', 0.1)
        self.efficiency_weight = config.get('score_efficiency_weight', 0.5)
        self.utilization_weight = config.get('score_utilization_weight', 0.2)
        
        # Carbon credit conversion factors
        self.carbon_per_kg_delivered = config.get('carbon_per_kg_delivered', 0.5)
        self.carbon_per_km_traveled = config.get('carbon_per_km_traveled', 0.2)
    
    def calculate_demand_satisfaction(
        self, 
        schedule: Schedule
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate how well demand is satisfied.
        
        Returns:
            Tuple of (satisfaction_score, fulfillment_by_farm_date)
        """
        df = schedule.to_dataframe()
        if df.empty:
            return 0.0, {}
        
        # Sum deliveries by farm and date
        delivered = df.groupby(['farm_id', 'date'])['quantity'].sum()
        
        # Compare to demand
        demand_df = self.precomputed.demand_by_farm_date.stack()
        demand_df.index.names = ['farm_id', 'date']
        
        # Calculate fulfillment ratio
        fulfillment = {}
        total_demand = 0
        total_delivered = 0
        
        for (farm_id, date), demand in demand_df.items():
            if demand > 0:
                total_demand += demand
                try:
                    qty = delivered.get((farm_id, str(date)), 0)
                    total_delivered += min(qty, demand)  # Cap at demand
                    fulfillment[(farm_id, str(date))] = min(qty / demand, 1.0)
                except KeyError:
                    fulfillment[(farm_id, str(date))] = 0.0
        
        satisfaction_ratio = total_delivered / total_demand if total_demand > 0 else 0
        score = satisfaction_ratio * 100 * self.demand_weight
        
        return score, fulfillment
    
    def calculate_distance_penalty(self, schedule: Schedule) -> float:
        """
        Calculate penalty for total distance traveled.
        
        Returns:
            Distance penalty score (to be subtracted)
        """
        df = schedule.to_dataframe()
        if df.empty:
            return 0.0
        
        total_distance = df['distance'].sum()
        
        # Normalize by number of deliveries for fair comparison
        avg_distance = total_distance / len(df) if len(df) > 0 else 0
        
        penalty = avg_distance * self.distance_weight
        return penalty
    
    def calculate_efficiency_bonus(self, schedule: Schedule) -> float:
        """
        Calculate bonus for efficient deliveries (high quantity per km).
        
        Returns:
            Efficiency bonus score
        """
        df = schedule.to_dataframe()
        if df.empty:
            return 0.0
        
        # Efficiency = quantity delivered per km traveled
        df['efficiency'] = df['quantity'] / (df['distance'] + 0.1)  # +0.1 to avoid div by zero
        
        avg_efficiency = df['efficiency'].mean()
        bonus = avg_efficiency * self.efficiency_weight
        
        return bonus
    
    def calculate_utilization_score(self, schedule: Schedule) -> float:
        """
        Calculate how well STP capacity is utilized.
        
        Returns:
            Utilization score
        """
        df = schedule.to_dataframe()
        if df.empty:
            return 0.0
        
        # Sum deliveries by STP and date
        stp_usage = df.groupby(['stp_id', 'date'])['quantity'].sum()
        
        # Calculate utilization ratio for each STP
        utilization_ratios = []
        for stp_idx, stp_id in enumerate(self.precomputed.stp_ids):
            capacity = self.precomputed.stp_capacities[stp_idx]
            if capacity > 0:
                daily_usage = stp_usage.loc[stp_id] if stp_id in stp_usage.index.get_level_values(0) else pd.Series([0])
                avg_utilization = daily_usage.mean() / capacity
                utilization_ratios.append(min(avg_utilization, 1.0))
        
        avg_utilization = np.mean(utilization_ratios) if utilization_ratios else 0
        score = avg_utilization * 100 * self.utilization_weight
        
        return score
    
    def calculate_penalties(self, schedule: Schedule) -> Tuple[float, Dict[str, float]]:
        """
        Calculate any penalties for constraint violations.
        
        Returns:
            Tuple of (total_penalty, penalty_breakdown)
        """
        df = schedule.to_dataframe()
        penalties = {}
        
        if df.empty:
            return 0.0, penalties
        
        # Penalty for over-delivery (beyond demand)
        # This would need actual demand data comparison
        over_delivery_penalty = 0.0
        penalties['over_delivery'] = over_delivery_penalty
        
        # Penalty for very long travel times
        long_travel_threshold = config.get('long_travel_hours', 4)
        long_trips = df[df['travel_time'] > long_travel_threshold]
        long_trip_penalty = len(long_trips) * 5
        penalties['long_trips'] = long_trip_penalty
        
        # Penalty for small inefficient deliveries
        min_qty = config.get('min_efficient_quantity', 50)
        small_deliveries = df[df['quantity'] < min_qty]
        small_delivery_penalty = len(small_deliveries) * 2
        penalties['small_deliveries'] = small_delivery_penalty
        
        total_penalty = sum(penalties.values())
        return total_penalty, penalties
    
    def calculate_carbon_credits(self, schedule: Schedule) -> float:
        """
        Calculate the actual carbon credits earned.
        
        This is a simplified model - adjust based on actual competition rules.
        
        Returns:
            Net carbon credits
        """
        df = schedule.to_dataframe()
        if df.empty:
            return 0.0
        
        # Carbon saved from recycled nutrients
        total_quantity = df['quantity'].sum()
        carbon_saved = total_quantity * self.carbon_per_kg_delivered
        
        # Carbon emitted from transport
        total_distance = df['distance'].sum()
        carbon_emitted = total_distance * self.carbon_per_km_traveled
        
        # Net carbon credit
        net_carbon = carbon_saved - carbon_emitted
        
        return net_carbon
    
    def score(self, schedule: Schedule, verbose: bool = True) -> ScoreBreakdown:
        """
        Calculate the complete score for a schedule.
        
        Args:
            schedule: The delivery schedule to score
            verbose: Print score breakdown if True
            
        Returns:
            ScoreBreakdown with all components
        """
        demand_score, _ = self.calculate_demand_satisfaction(schedule)
        distance_penalty = self.calculate_distance_penalty(schedule)
        efficiency_bonus = self.calculate_efficiency_bonus(schedule)
        utilization_score = self.calculate_utilization_score(schedule)
        penalty_score, _ = self.calculate_penalties(schedule)
        
        # Calculate total score
        total_score = (
            demand_score 
            - distance_penalty 
            + efficiency_bonus 
            + utilization_score 
            - penalty_score
        )
        
        breakdown = ScoreBreakdown(
            total_score=total_score,
            demand_satisfaction_score=demand_score,
            distance_penalty=distance_penalty,
            efficiency_bonus=efficiency_bonus,
            utilization_score=utilization_score,
            penalty_score=penalty_score
        )
        
        if verbose:
            print(breakdown)
            carbon = self.calculate_carbon_credits(schedule)
            print(f"  Net Carbon Credits: {carbon:.2f}")
        
        return breakdown


def compute_carbon_credit_score(
    deliveries: List[Tuple[str, str, str, float]],
    distances: Dict[str, Dict[str, float]],
    daily_n_demand: pd.DataFrame,
    storage_tracker=None,
    config_dict: Dict = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total carbon credits for a delivery schedule.
    
    Uses values from config.json:
      - +5 kg CO2 eq per kg effective nitrogen uptake
      - +0.2 kg CO2 eq per ton biosolids applied (converted from per kg)
      - -0.9 kg CO2 eq per km transported
      - -10 kg CO2 eq per kg excess nitrogen
      - -1000 kg CO2 eq per ton overflow
    
    Args:
        deliveries: List of tuples (date, stp_id, farm_id, tons_delivered)
        distances: Precomputed distances[stp_id][farm_id] in km
        daily_n_demand: Nitrogen demand DataFrame (date as rows, farm_ids as columns)
        storage_tracker: Optional STPStorageTracker with overflow data
        config_dict: Optional config overrides (loads from config.json if None)
        
    Returns:
        Tuple of (total_score, score_breakdown)
        score_breakdown contains:
            - 'nitrogen_uptake_credit': Credits from effective N application
            - 'biosolid_credit': Credits from soil organic carbon gain
            - 'transport_penalty': Penalty from transport emissions
            - 'excess_n_penalty': Penalty from excess nitrogen leaching
            - 'overflow_penalty': Penalty from STP overflow
    
    Example:
        >>> deliveries = optimizer.optimize()
        >>> score, breakdown = compute_carbon_credit_score(
        ...     deliveries, distances, daily_n_demand, optimizer.storage_tracker
        ... )
        >>> print(f"Total score: {score:.2f} kg CO2 eq")
    """
    # Load config values
    cfg = config_dict if config_dict is not None else config.all
    
    # Get scoring constants from config
    # Agronomic constants
    agronomic = cfg.get('agronomic_constants', {})
    n_per_ton = agronomic.get('nitrogen_content_kg_per_ton_biosolid', 25)
    n_credit_per_kg = agronomic.get('synthetic_n_offset_credit_kg_co2_per_kg_n', 5.0)
    soc_credit_per_kg = agronomic.get('soil_organic_carbon_gain_kg_co2_per_kg_biosolid', 0.2)
    leaching_penalty_per_kg = agronomic.get('leaching_penalty_kg_co2_per_kg_excess_n', 10.0)
    buffer_percent = agronomic.get('application_buffer_percent', 10)
    
    # Logistics constants
    logistics = cfg.get('logistics_constants', {})
    transport_penalty_per_km = logistics.get('diesel_emission_factor_kg_co2_per_km', 0.9)
    
    # Environmental thresholds
    environmental = cfg.get('environmental_thresholds', {})
    overflow_penalty_per_ton = environmental.get('stp_overflow_penalty_kg_co2_per_ton', 1000.0)
    
    # Initialize score components
    score_breakdown = {
        'nitrogen_uptake_credit': 0.0,
        'biosolid_credit': 0.0,
        'transport_penalty': 0.0,
        'excess_n_penalty': 0.0,
        'overflow_penalty': 0.0
    }
    
    # Parse demand data for excess N calculation
    demand_df = daily_n_demand.copy()
    if 'date' in demand_df.columns:
        demand_df['date'] = pd.to_datetime(demand_df['date'])
        demand_df = demand_df.set_index('date')
    
    # Track N delivered per farm per date
    n_delivered: Dict[str, Dict[str, float]] = {}
    
    # Process each delivery
    total_distance = 0.0
    total_biosolid_tons = 0.0
    total_n_delivered = 0.0
    
    for date_str, stp_id, farm_id, tons_delivered in deliveries:
        # 1. Calculate nitrogen delivered
        n_kg = tons_delivered * n_per_ton
        total_n_delivered += n_kg
        total_biosolid_tons += tons_delivered
        
        # Track N per farm/date for excess calculation
        if farm_id not in n_delivered:
            n_delivered[farm_id] = {}
        if date_str not in n_delivered[farm_id]:
            n_delivered[farm_id][date_str] = 0.0
        n_delivered[farm_id][date_str] += n_kg
        
        # 2. Calculate distance for transport penalty
        dist = distances.get(stp_id, {}).get(farm_id, 0.0)
        total_distance += dist
    
    # Calculate effective nitrogen uptake (capped at demand + buffer)
    effective_n_uptake = 0.0
    total_excess_n = 0.0
    
    for farm_id, date_n_dict in n_delivered.items():
        for date_str, n_kg in date_n_dict.items():
            # Get demand for this farm/date
            try:
                date_parsed = pd.to_datetime(date_str)
                if farm_id in demand_df.columns:
                    demand = demand_df.loc[date_parsed, farm_id] if date_parsed in demand_df.index else 0.0
                else:
                    demand = 0.0
            except (KeyError, TypeError):
                demand = 0.0
            
            # Max allowed is demand + buffer
            max_allowed = demand * (1 + buffer_percent / 100)
            
            # Effective uptake is min of delivered and max allowed
            effective = min(n_kg, max_allowed)
            effective_n_uptake += effective
            
            # Excess is anything above max allowed
            if n_kg > max_allowed:
                excess = n_kg - max_allowed
                total_excess_n += excess
    
    # Calculate credits and penalties
    # +5 kg CO2 eq per kg effective nitrogen uptake
    score_breakdown['nitrogen_uptake_credit'] = effective_n_uptake * n_credit_per_kg
    
    # +0.2 kg CO2 eq per kg biosolids applied (config says per kg, so convert tons to kg)
    total_biosolid_kg = total_biosolid_tons * 1000  # Convert tons to kg
    score_breakdown['biosolid_credit'] = total_biosolid_kg * soc_credit_per_kg
    
    # -0.9 kg CO2 eq per km transported
    score_breakdown['transport_penalty'] = total_distance * transport_penalty_per_km
    
    # -10 kg CO2 eq per kg excess nitrogen
    score_breakdown['excess_n_penalty'] = total_excess_n * leaching_penalty_per_kg
    
    # -1000 kg CO2 eq per ton overflow
    if storage_tracker is not None:
        total_overflow = sum(
            sum(overflow_dict.values())
            for overflow_dict in storage_tracker.overflow.values()
        )
        score_breakdown['overflow_penalty'] = total_overflow * overflow_penalty_per_ton
    
    # Calculate total score
    total_score = (
        score_breakdown['nitrogen_uptake_credit'] +
        score_breakdown['biosolid_credit'] -
        score_breakdown['transport_penalty'] -
        score_breakdown['excess_n_penalty'] -
        score_breakdown['overflow_penalty']
    )
    
    return total_score, score_breakdown


def print_score_breakdown(total_score: float, breakdown: Dict[str, float]) -> None:
    """
    Print a formatted breakdown of the carbon credit score.
    
    Args:
        total_score: The total carbon credit score
        breakdown: Dictionary with score components
    """
    print("\n" + "=" * 50)
    print("CARBON CREDIT SCORE BREAKDOWN")
    print("=" * 50)
    print(f"\n  (+) Nitrogen Uptake Credit:  {breakdown['nitrogen_uptake_credit']:>12,.2f} kg CO2 eq")
    print(f"  (+) Biosolid Application:    {breakdown['biosolid_credit']:>12,.2f} kg CO2 eq")
    print(f"  (-) Transport Emissions:     {breakdown['transport_penalty']:>12,.2f} kg CO2 eq")
    print(f"  (-) Excess N Leaching:       {breakdown['excess_n_penalty']:>12,.2f} kg CO2 eq")
    print(f"  (-) STP Overflow:            {breakdown['overflow_penalty']:>12,.2f} kg CO2 eq")
    print("-" * 50)
    print(f"  TOTAL SCORE:                 {total_score:>12,.2f} kg CO2 eq")
    print("=" * 50 + "\n")


def validate_solution(schedule: Schedule, precomputed: PrecomputedData) -> bool:
    """
    Validate that a solution meets all hard constraints.
    
    Args:
        schedule: Schedule to validate
        precomputed: Precomputed data for constraint checking
        
    Returns:
        True if valid, raises exception if invalid
    """
    df = schedule.to_dataframe()
    
    if df.empty:
        print("Warning: Empty schedule")
        return True
    
    errors = []
    
    # Check STP IDs are valid
    invalid_stps = set(df['stp_id']) - set(precomputed.stp_ids)
    if invalid_stps:
        errors.append(f"Invalid STP IDs: {invalid_stps}")
    
    # Check farm IDs are valid
    invalid_farms = set(df['farm_id']) - set(precomputed.farm_ids)
    if invalid_farms:
        errors.append(f"Invalid farm IDs: {invalid_farms}")
    
    # Check quantities are positive
    if (df['quantity'] <= 0).any():
        errors.append("Found non-positive delivery quantities")
    
    # Check capacity constraints by date and STP
    for (date, stp_id), group in df.groupby(['date', 'stp_id']):
        total_delivered = group['quantity'].sum()
        stp_idx = precomputed.stp_ids.index(stp_id)
        capacity = precomputed.stp_capacities[stp_idx]
        
        if total_delivered > capacity * 1.001:  # Small tolerance for float errors
            errors.append(
                f"STP {stp_id} exceeded capacity on {date}: "
                f"{total_delivered:.2f} > {capacity:.2f}"
            )
    
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Solution validation passed!")
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
        
        scorer = Scorer(precomputed)
        breakdown = scorer.score(schedule)
        
        validate_solution(schedule, precomputed)
    except FileNotFoundError as e:
        print(f"Error: {e}")
