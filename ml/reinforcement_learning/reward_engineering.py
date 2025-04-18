"""
Reward engineering for reinforcement learning in cryptocurrency trading.

This module provides reward functions and classes that can be used 
to shape rewards in trading environments.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Callable
import pandas as pd


class RewardFunction:
    """Base class for all reward functions."""
    
    def __init__(self, scaling: float = 1.0, window_size: int = 20):
        """
        Initialize the reward function.
        
        Args:
            scaling: Scaling factor to apply to the reward
            window_size: Number of steps to consider for certain metrics
        """
        self.scaling = scaling
        self.window_size = window_size
        
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate the reward value.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters for specific reward functions
            
        Returns:
            Calculated reward value
        """
        return 0.0


class ReturnReward(RewardFunction):
    """Simple return-based reward."""
    
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate reward based on portfolio return.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Return-based reward
        """
        if previous_value <= 0:
            return 0.0
            
        return ((current_value - previous_value) / previous_value) * self.scaling


class SharpeRatio(RewardFunction):
    """Sharpe ratio reward function."""
    
    def __init__(self, scaling: float = 1.0, window_size: int = 20, risk_free_rate: float = 0.0):
        """
        Initialize Sharpe ratio reward function.
        
        Args:
            scaling: Scaling factor to apply to the reward
            window_size: Number of steps to consider
            risk_free_rate: Risk-free rate (annualized)
        """
        super().__init__(scaling, window_size)
        self.risk_free_rate = risk_free_rate  # Annualized risk-free rate
        self.daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1  # Daily risk-free rate
        
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate reward based on Sharpe ratio.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Sharpe ratio-based reward
        """
        if history is None or len(history) < 2:
            # Fall back to return if not enough history
            return ((current_value - previous_value) / previous_value) * self.scaling
            
        # Calculate returns
        history = history[-self.window_size:] if len(history) > self.window_size else history
        history.append(current_value)
        returns = [(history[i] - history[i-1]) / history[i-1] for i in range(1, len(history))]
        
        # Calculate Sharpe ratio
        if len(returns) < 2:
            return returns[0] * self.scaling
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return mean_return * self.scaling
            
        # Daily Sharpe ratio
        daily_sharpe = (mean_return - self.daily_risk_free) / std_return
        
        # Annualized Sharpe ratio
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        
        # For incremental learning, scale down by window size
        return (annualized_sharpe / np.sqrt(self.window_size)) * self.scaling


class SortinoRatio(RewardFunction):
    """Sortino ratio reward function."""
    
    def __init__(self, scaling: float = 1.0, window_size: int = 20, risk_free_rate: float = 0.0):
        """
        Initialize Sortino ratio reward function.
        
        Args:
            scaling: Scaling factor to apply to the reward
            window_size: Number of steps to consider
            risk_free_rate: Risk-free rate (annualized)
        """
        super().__init__(scaling, window_size)
        self.risk_free_rate = risk_free_rate  # Annualized risk-free rate
        self.daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1  # Daily risk-free rate
        
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate reward based on Sortino ratio.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Sortino ratio-based reward
        """
        if history is None or len(history) < 2:
            # Fall back to return if not enough history
            return ((current_value - previous_value) / previous_value) * self.scaling
            
        # Calculate returns
        history = history[-self.window_size:] if len(history) > self.window_size else history
        history.append(current_value)
        returns = [(history[i] - history[i-1]) / history[i-1] for i in range(1, len(history))]
        
        # Calculate Sortino ratio
        if len(returns) < 2:
            return returns[0] * self.scaling
            
        mean_return = np.mean(returns)
        
        # Downside deviation (only consider negative returns)
        downside_returns = [r for r in returns if r < self.daily_risk_free]
        
        # If no downside returns, return a positive scaled value
        if len(downside_returns) == 0:
            return mean_return * self.scaling * 2  # Double reward for no downside
            
        downside_deviation = np.sqrt(np.mean([(r - self.daily_risk_free)**2 for r in downside_returns]))
        
        if downside_deviation == 0:
            return mean_return * self.scaling
            
        # Daily Sortino ratio
        daily_sortino = (mean_return - self.daily_risk_free) / downside_deviation
        
        # Annualized Sortino ratio
        annualized_sortino = daily_sortino * np.sqrt(252)
        
        # For incremental learning, scale down by window size
        return (annualized_sortino / np.sqrt(self.window_size)) * self.scaling


class CalmarRatio(RewardFunction):
    """Calmar ratio reward function."""
    
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate reward based on Calmar ratio.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Calmar ratio-based reward
        """
        if history is None or len(history) < 2:
            # Fall back to return if not enough history
            return ((current_value - previous_value) / previous_value) * self.scaling
            
        # Calculate returns
        history = history[-self.window_size:] if len(history) > self.window_size else history
        history.append(current_value)
        returns = [(history[i] - history[i-1]) / history[i-1] for i in range(1, len(history))]
        
        # Calculate Calmar ratio
        if len(returns) < 2:
            return returns[0] * self.scaling
            
        # Calculate max drawdown
        peak = 1.0
        drawdowns = []
        for r in returns:
            peak = max(peak, peak * (1 + r))
            drawdowns.append(1 - (peak * (1 + r)) / peak)
            
        max_drawdown = max(drawdowns) if drawdowns else 0.001
        
        # Prevent division by zero
        if max_drawdown == 0:
            max_drawdown = 0.001
            
        # Calculate annual return
        annual_return = np.prod([1 + r for r in returns]) ** (252 / len(returns)) - 1
        
        # Calculate Calmar ratio
        calmar = annual_return / max_drawdown
        
        # For incremental learning, scale down by window size
        return (calmar / self.window_size) * self.scaling


class MaxDrawdownPenalty(RewardFunction):
    """Maximum drawdown penalty reward function."""
    
    def __init__(self, scaling: float = -1.0, window_size: int = 20):
        """
        Initialize maximum drawdown penalty reward function.
        
        Args:
            scaling: Scaling factor to apply to the reward (negative for penalty)
            window_size: Number of steps to consider
        """
        super().__init__(scaling, window_size)
        
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate penalty based on maximum drawdown.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Maximum drawdown penalty reward
        """
        if history is None or len(history) < 2:
            # No penalty if no history
            return 0.0
            
        # Calculate historical peaks and drawdowns
        history = history[-self.window_size:] if len(history) > self.window_size else history
        history.append(current_value)
        
        peak = history[0]
        drawdown = 0.0
        
        for value in history:
            if value > peak:
                peak = value
            current_drawdown = (peak - value) / peak
            drawdown = max(drawdown, current_drawdown)
            
        # Return negative value as penalty
        return drawdown * self.scaling


class RiskAdjustedReward(RewardFunction):
    """Risk-adjusted reward function."""
    
    def __init__(self, scaling: float = 1.0, window_size: int = 20, risk_aversion: float = 1.0):
        """
        Initialize risk-adjusted reward function.
        
        Args:
            scaling: Scaling factor to apply to the reward
            window_size: Number of steps to consider
            risk_aversion: Risk aversion coefficient
        """
        super().__init__(scaling, window_size)
        self.risk_aversion = risk_aversion
        
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate risk-adjusted reward.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Risk-adjusted reward
        """
        if history is None or len(history) < 2:
            # Fall back to return if not enough history
            return ((current_value - previous_value) / previous_value) * self.scaling
            
        # Calculate returns
        history = history[-self.window_size:] if len(history) > self.window_size else history
        history.append(current_value)
        returns = [(history[i] - history[i-1]) / history[i-1] for i in range(1, len(history))]
        
        if len(returns) < 2:
            return returns[0] * self.scaling
            
        # Calculate mean and variance of returns
        mean_return = np.mean(returns)
        var_return = np.var(returns)
        
        # Risk-adjusted return (like utility function in CAPM)
        risk_adjusted = mean_return - 0.5 * self.risk_aversion * var_return
        
        return risk_adjusted * self.scaling


class ConsistencyReward(RewardFunction):
    """Reward function that prioritizes consistent returns."""
    
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate reward based on consistency of returns.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Consistency-based reward
        """
        current_return = (current_value - previous_value) / previous_value
        
        if history is None or len(history) < 2:
            return current_return * self.scaling
            
        # Calculate historical returns
        history = history[-self.window_size:] if len(history) > self.window_size else history
        returns = [(history[i] - history[i-1]) / history[i-1] for i in range(1, len(history))]
        
        # Add current return
        returns.append(current_return)
        
        if len(returns) < 2:
            return current_return * self.scaling
            
        # Base reward is the current return
        reward = current_return
        
        # Check if current return is in the same direction as the average
        avg_return = np.mean(returns[:-1])  # Exclude current return
        
        # Add consistency bonus if current return has the same sign as the average
        if (current_return >= 0 and avg_return >= 0) or (current_return < 0 and avg_return < 0):
            reward += 0.2 * abs(current_return)
        
        # Add consistency bonus for low standard deviation
        std_returns = np.std(returns)
        consistency_bonus = 0.1 * (1.0 / (1.0 + 10 * std_returns))
        reward += consistency_bonus
        
        return reward * self.scaling


class WinningStreakReward(RewardFunction):
    """Reward function that incentivizes winning streaks."""
    
    def __init__(self, scaling: float = 1.0, window_size: int = 20, streak_bonus: float = 0.1):
        """
        Initialize winning streak reward function.
        
        Args:
            scaling: Scaling factor to apply to the reward
            window_size: Number of steps to consider
            streak_bonus: Additional multiplier for consecutive positive returns
        """
        super().__init__(scaling, window_size)
        self.streak_bonus = streak_bonus
        self.current_streak = 0
        
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate reward with winning streak bonus.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Winning streak reward
        """
        current_return = (current_value - previous_value) / previous_value
        
        # Update streak
        if current_return > 0:
            self.current_streak += 1
        else:
            self.current_streak = 0
            
        # Base reward is the current return
        reward = current_return
        
        # Add streak bonus
        streak_multiplier = 1.0 + (self.streak_bonus * min(self.current_streak, 10))
        reward *= streak_multiplier
        
        return reward * self.scaling


class CustomReward(RewardFunction):
    """Custom reward function using a user-provided calculation function."""
    
    def __init__(self, reward_func: Callable, scaling: float = 1.0, window_size: int = 20):
        """
        Initialize custom reward function.
        
        Args:
            reward_func: User-provided function to calculate reward
            scaling: Scaling factor to apply to the reward
            window_size: Number of steps to consider
        """
        super().__init__(scaling, window_size)
        self.reward_func = reward_func
        
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate reward using user-provided function.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters
            
        Returns:
            Custom calculated reward
        """
        reward = self.reward_func(current_value, previous_value, history, **kwargs)
        return reward * self.scaling


class RewardCombiner:
    """Combine multiple reward functions with specified weights."""
    
    def __init__(self, reward_functions: List[RewardFunction], weights: List[float] = None):
        """
        Initialize reward combiner.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: Weights for each reward function
        """
        self.reward_functions = reward_functions
        
        if weights is None:
            # Equal weights if not specified
            self.weights = [1.0 / len(reward_functions)] * len(reward_functions)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
    def calculate(self, 
                 current_value: float, 
                 previous_value: float, 
                 history: List[float] = None, 
                 **kwargs) -> float:
        """
        Calculate combined reward.
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            history: Historical portfolio values
            **kwargs: Additional parameters passed to all reward functions
            
        Returns:
            Combined reward value
        """
        rewards = []
        
        for func, weight in zip(self.reward_functions, self.weights):
            reward = func.calculate(current_value, previous_value, history, **kwargs)
            rewards.append(reward * weight)
            
        return sum(rewards)


# Example usage function
def create_reward_function(reward_type: str, **kwargs) -> RewardFunction:
    """
    Factory function to create reward functions.
    
    Args:
        reward_type: Type of reward function
        **kwargs: Additional parameters for the reward function
        
    Returns:
        Instantiated reward function
    """
    if reward_type == 'return':
        return ReturnReward(**kwargs)
    elif reward_type == 'sharpe':
        return SharpeRatio(**kwargs)
    elif reward_type == 'sortino':
        return SortinoRatio(**kwargs)
    elif reward_type == 'calmar':
        return CalmarRatio(**kwargs)
    elif reward_type == 'max_drawdown_penalty':
        return MaxDrawdownPenalty(**kwargs)
    elif reward_type == 'risk_adjusted':
        return RiskAdjustedReward(**kwargs)
    elif reward_type == 'consistency':
        return ConsistencyReward(**kwargs)
    elif reward_type == 'winning_streak':
        return WinningStreakReward(**kwargs)
    elif reward_type == 'combined':
        # For combined, expect a list of dictionaries with 'type' and other params
        if 'functions' not in kwargs:
            raise ValueError("Must provide 'functions' parameter for combined reward")
            
        funcs = []
        weights = kwargs.get('weights', None)
        
        for func_config in kwargs['functions']:
            func_type = func_config.pop('type')
            funcs.append(create_reward_function(func_type, **func_config))
            
        return RewardCombiner(funcs, weights)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}") 