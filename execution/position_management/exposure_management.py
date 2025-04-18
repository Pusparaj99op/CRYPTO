import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExposureConfig:
    max_total_exposure: float = 1.0  # Max exposure as fraction of total capital
    max_single_market_exposure: float = 0.25  # Max exposure to a single market
    max_correlated_exposure: float = 0.40  # Max exposure to correlated assets
    correlation_threshold: float = 0.7  # Threshold to consider assets correlated
    diversification_target: int = 4  # Min number of different markets
    auto_hedge_threshold: float = 0.35  # Threshold to auto-hedge exposure
    base_risk_weight: Dict[str, float] = None  # Base risk weights per market

class ExposureManager:
    """
    Manages trading exposure across markets, applying risk limits
    and ensuring proper diversification across a crypto portfolio.
    """
    
    def __init__(
        self,
        config: ExposureConfig = None,
        correlation_matrix: Dict[Tuple[str, str], float] = None,
        market_categories: Dict[str, str] = None,
        volatility_lookup: Dict[str, float] = None,
        on_exposure_exceeded: Optional[Callable] = None
    ):
        self.config = config or ExposureConfig()
        self.correlation_matrix = correlation_matrix or {}
        self.market_categories = market_categories or {}
        self.volatility_lookup = volatility_lookup or {}
        self.on_exposure_exceeded = on_exposure_exceeded
        
        self.current_exposures = {}
        self.category_exposures = {}
        self.total_exposure = 0.0
        self.exposure_history = []
        
    def update_correlation_matrix(self, new_matrix: Dict[Tuple[str, str], float]) -> None:
        """Update the correlation matrix between markets."""
        self.correlation_matrix = new_matrix
        
    def update_market_volatility(self, market: str, volatility: float) -> None:
        """Update volatility values for a specific market."""
        self.volatility_lookup[market] = volatility
        
    def calculate_exposure(
        self, 
        positions: Dict[str, Dict[str, Any]], 
        account_value: float
    ) -> Dict[str, float]:
        """
        Calculate current exposure levels across all markets.
        
        Args:
            positions: Dictionary mapping market symbols to position details
            account_value: Total account value in base currency
            
        Returns:
            Dictionary of exposures by market as fraction of account value
        """
        exposures = {}
        self.category_exposures = {}
        total_exposure = 0.0
        
        for market, position in positions.items():
            # Calculate notional exposure (position size * leverage)
            position_size = position.get('size', 0)
            leverage = position.get('leverage', 1.0)
            notional_exposure = position_size * leverage
            
            # Calculate as percentage of account value
            relative_exposure = notional_exposure / account_value if account_value > 0 else 0
            exposures[market] = relative_exposure
            total_exposure += abs(relative_exposure)
            
            # Update category exposure
            category = self.market_categories.get(market, 'uncategorized')
            self.category_exposures[category] = self.category_exposures.get(category, 0) + abs(relative_exposure)
            
        self.current_exposures = exposures
        self.total_exposure = total_exposure
        self.exposure_history.append((total_exposure, dict(exposures)))
        
        return exposures
    
    def check_exposure_limits(self) -> Dict[str, Any]:
        """
        Check if any exposure limits have been exceeded.
        
        Returns:
            Dictionary with violation details
        """
        violations = {
            'exceeded_limits': [],
            'is_compliant': True,
            'highest_exposure': {'market': None, 'value': 0.0},
            'diversification_status': 'ADEQUATE' if len(self.current_exposures) >= self.config.diversification_target else 'INSUFFICIENT'
        }
        
        # Check total exposure
        if self.total_exposure > self.config.max_total_exposure:
            violations['exceeded_limits'].append({
                'type': 'TOTAL_EXPOSURE',
                'current': self.total_exposure,
                'limit': self.config.max_total_exposure,
                'excess': self.total_exposure - self.config.max_total_exposure
            })
            violations['is_compliant'] = False
            
        # Check single market exposure
        for market, exposure in self.current_exposures.items():
            abs_exposure = abs(exposure)
            if abs_exposure > self.config.max_single_market_exposure:
                violations['exceeded_limits'].append({
                    'type': 'SINGLE_MARKET',
                    'market': market,
                    'current': abs_exposure,
                    'limit': self.config.max_single_market_exposure,
                    'excess': abs_exposure - self.config.max_single_market_exposure
                })
                violations['is_compliant'] = False
                
            # Track highest exposure
            if abs_exposure > violations['highest_exposure']['value']:
                violations['highest_exposure'] = {'market': market, 'value': abs_exposure}
                
        # Check correlated exposure
        correlated_groups = self._identify_correlated_groups()
        for group, markets in correlated_groups.items():
            group_exposure = sum(abs(self.current_exposures.get(market, 0)) for market in markets)
            if group_exposure > self.config.max_correlated_exposure:
                violations['exceeded_limits'].append({
                    'type': 'CORRELATED_EXPOSURE',
                    'group': group,
                    'markets': markets,
                    'current': group_exposure,
                    'limit': self.config.max_correlated_exposure,
                    'excess': group_exposure - self.config.max_correlated_exposure
                })
                violations['is_compliant'] = False
                
        # Check category exposure
        for category, exposure in self.category_exposures.items():
            # Using same limit as correlated assets for now
            if exposure > self.config.max_correlated_exposure:
                violations['exceeded_limits'].append({
                    'type': 'CATEGORY_EXPOSURE',
                    'category': category,
                    'current': exposure,
                    'limit': self.config.max_correlated_exposure,
                    'excess': exposure - self.config.max_correlated_exposure
                })
                violations['is_compliant'] = False
                
        # Call handler if limits exceeded
        if not violations['is_compliant'] and self.on_exposure_exceeded:
            self.on_exposure_exceeded(violations)
            
        return violations
    
    def _identify_correlated_groups(self) -> Dict[str, List[str]]:
        """Identify groups of correlated markets based on correlation matrix."""
        markets = list(self.current_exposures.keys())
        groups = {}
        group_id = 0
        
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                key = (market1, market2) if market1 < market2 else (market2, market1)
                correlation = self.correlation_matrix.get(key, 0)
                
                if abs(correlation) >= self.config.correlation_threshold:
                    # Check if either market is already in a group
                    found_group = None
                    for gid, group_markets in groups.items():
                        if market1 in group_markets or market2 in group_markets:
                            found_group = gid
                            break
                            
                    if found_group is not None:
                        # Add to existing group
                        if market1 not in groups[found_group]:
                            groups[found_group].append(market1)
                        if market2 not in groups[found_group]:
                            groups[found_group].append(market2)
                    else:
                        # Create new group
                        groups[f"group_{group_id}"] = [market1, market2]
                        group_id += 1
                        
        return groups
    
    def get_max_position_size(
        self, 
        market: str, 
        account_value: float,
        leverage: float = 1.0
    ) -> float:
        """
        Calculate maximum position size allowable for a market given current exposure.
        
        Args:
            market: Market symbol
            account_value: Total account value
            leverage: Leverage multiplier
            
        Returns:
            Maximum position size in base currency
        """
        # Apply market-specific risk weight if available
        risk_weight = self.config.base_risk_weight.get(market, 1.0) if self.config.base_risk_weight else 1.0
        
        # Factor in volatility
        volatility_factor = 1.0
        if market in self.volatility_lookup:
            # Higher volatility = lower position size
            vol = self.volatility_lookup[market]
            volatility_factor = 1.0 / (1.0 + vol)
            
        # Get limits based on remaining exposure allowed
        remaining_total = self.config.max_total_exposure - self.total_exposure
        remaining_single = self.config.max_single_market_exposure - self.current_exposures.get(market, 0)
        
        # Take the more restrictive limit
        max_exposure = min(remaining_total, remaining_single)
        max_exposure = max(0, max_exposure)  # Ensure non-negative
        
        # Apply risk weight and volatility adjustment
        adjusted_exposure = max_exposure * risk_weight * volatility_factor
        
        # Convert to absolute position size
        max_position_size = (adjusted_exposure * account_value) / leverage
        
        return max_position_size
    
    def suggest_exposure_rebalance(self) -> List[Dict[str, Any]]:
        """
        Suggest rebalancing actions to optimize exposure distribution.
        
        Returns:
            List of suggested actions to rebalance exposure
        """
        suggestions = []
        
        # If total exposure is too high, suggest reduction
        if self.total_exposure > self.config.max_total_exposure:
            excess = self.total_exposure - self.config.max_total_exposure
            # Sort positions by size (largest first) to reduce
            markets_by_size = sorted(
                self.current_exposures.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for market, exposure in markets_by_size:
                suggestions.append({
                    'action': 'REDUCE_POSITION',
                    'market': market,
                    'current_exposure': exposure,
                    'target_reduction': min(abs(exposure), excess),
                    'reason': 'Total exposure exceeds maximum allowed'
                })
                
                # Stop once we've suggested enough reductions
                excess -= abs(exposure)
                if excess <= 0:
                    break
        
        # Individual market exposure checks
        for market, exposure in self.current_exposures.items():
            if abs(exposure) > self.config.max_single_market_exposure:
                excess = abs(exposure) - self.config.max_single_market_exposure
                suggestions.append({
                    'action': 'REDUCE_POSITION',
                    'market': market,
                    'current_exposure': exposure,
                    'target_reduction': excess,
                    'reason': f'Single market exposure for {market} exceeds limit'
                })
        
        # Check for insufficient diversification
        if len(self.current_exposures) < self.config.diversification_target:
            suggestions.append({
                'action': 'INCREASE_DIVERSIFICATION',
                'current_markets': len(self.current_exposures),
                'target_markets': self.config.diversification_target,
                'reason': 'Portfolio has insufficient diversification'
            })
            
        # Check for correlated exposure
        correlated_groups = self._identify_correlated_groups()
        for group, markets in correlated_groups.items():
            group_exposure = sum(abs(self.current_exposures.get(market, 0)) for market in markets)
            if group_exposure > self.config.max_correlated_exposure:
                suggestions.append({
                    'action': 'REDUCE_CORRELATED_EXPOSURE',
                    'markets': markets,
                    'current_exposure': group_exposure,
                    'target_reduction': group_exposure - self.config.max_correlated_exposure,
                    'reason': 'Exposure to correlated assets exceeds limit'
                })
                
                # If exposure is very high, suggest hedging
                if group_exposure > self.config.auto_hedge_threshold:
                    suggestions.append({
                        'action': 'HEDGE_EXPOSURE',
                        'markets': markets,
                        'current_exposure': group_exposure,
                        'reason': 'High exposure to correlated assets'
                    })
        
        return suggestions
    
    def get_exposure_metrics(self) -> Dict[str, Any]:
        """
        Get current exposure metrics for reporting.
        
        Returns:
            Dictionary with exposure metrics
        """
        # Calculate effective number of positions (concentration adjusted)
        exposures = list(self.current_exposures.values())
        herfindahl_index = sum(x**2 for x in exposures) if exposures else 0
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        # Calculate exposure direction (net long/short)
        directional_exposure = sum(self.current_exposures.values())
        direction = "LONG" if directional_exposure > 0 else "SHORT" if directional_exposure < 0 else "NEUTRAL"
        
        # Calculate exposure by category
        category_breakdown = dict(self.category_exposures)
        
        return {
            "total_exposure": self.total_exposure,
            "directional_exposure": directional_exposure,
            "direction": direction,
            "num_markets": len(self.current_exposures),
            "effective_positions": effective_positions, 
            "highest_single_exposure": max(abs(e) for e in exposures) if exposures else 0,
            "category_breakdown": category_breakdown,
            "exposure_vs_limit": self.total_exposure / self.config.max_total_exposure if self.config.max_total_exposure > 0 else float('inf'),
            "compliance_status": "COMPLIANT" if self.total_exposure <= self.config.max_total_exposure else "NON_COMPLIANT"
        } 