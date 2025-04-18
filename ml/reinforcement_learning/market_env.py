"""
Market environment simulation for cryptocurrency trading reinforcement learning.

This module provides environment classes that simulate cryptocurrency markets
for reinforcement learning agents to interact with.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import List, Dict, Tuple, Union, Optional, Any
import matplotlib.pyplot as plt
import os


class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency trading environment that follows gym interface.
    
    This environment simulates cryptocurrency trading for a single asset.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        reward_function: str = 'sharpe',
        reward_scaling: float = 1.0,
        features: Optional[List[str]] = None,
        feature_normalization: bool = True,
        max_position: float = 1.0
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Number of past observations to include in state
            initial_balance: Starting account balance
            commission: Trading commission as a fraction
            reward_function: Reward function type ('returns', 'sharpe', 'sortino', 'calmar')
            reward_scaling: Scaling factor for rewards
            features: List of column names to use as features
            feature_normalization: Whether to normalize features
            max_position: Maximum position size (1.0 = 100% of portfolio)
        """
        super(CryptoTradingEnv, self).__init__()
        
        # Store parameters
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_function = reward_function
        self.reward_scaling = reward_scaling
        self.feature_normalization = feature_normalization
        self.max_position = max_position
        
        # Set default features if not provided
        if features is None:
            self.features = ['open', 'high', 'low', 'close', 'volume']
        else:
            self.features = features
            
        # Check if all required features are in DataFrame
        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not in DataFrame")
                
        # Calculate number of features and observations
        self.num_features = len(self.features)
        self.num_observations = len(df)
        
        # Define action and observation spaces
        # Action space: continuous value between -1 (max short) and 1 (max long)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: market data features + account state
        # Market data: window_size * num_features
        # Account state: [balance, position, portfolio_value, profit_loss]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size * self.num_features + 4,), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Reset account state
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_history = []
        self.balance_history = [self.initial_balance]
        self.nav_history = [self.initial_balance]  # Net asset value
        self.rewards_history = []
        
        # Reset market position
        self.current_step = self.window_size
        
        # Get initial observation
        return self._get_observation()
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Trading action (position size between -1 and 1)
            
        Returns:
            next_observation, reward, done, info
        """
        # Execute the trade
        reward, info = self._execute_trade(action)
        
        # Store histories
        self.position_history.append(self.position)
        self.balance_history.append(self.balance)
        self.nav_history.append(info['portfolio_value'])
        self.rewards_history.append(reward)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.num_observations
        
        # Get new observation
        observation = self._get_observation()
        
        return observation, reward, done, info
        
    def _execute_trade(self, action):
        """
        Execute a trade based on the action.
        
        Args:
            action: Position size between -1 (max short) and 1 (max long)
            
        Returns:
            reward, info
        """
        # Clip action to ensure it's within bounds
        action = np.clip(action[0], -self.max_position, self.max_position)
        
        # Get current price data
        current_price = self.df.iloc[self.current_step]['close']
        
        # Calculate position change
        previous_position = self.position
        target_position = action
        position_change = target_position - previous_position
        
        # Calculate cost to execute trade (commission * traded value)
        if abs(position_change) > 0.0001:  # Minimum change threshold to avoid tiny trades
            trade_value = abs(position_change) * self.nav_history[-1]
            commission_cost = trade_value * self.commission
            
            # Update balance and position
            self.balance -= commission_cost
            self.position = target_position
        
        # Calculate portfolio value
        position_value = self.position * self.nav_history[-1] * (
            self.df.iloc[self.current_step]['close'] / self.df.iloc[self.current_step-1]['close']
        )
        portfolio_value = self.balance + position_value
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_value)
        
        info = {
            'step': self.current_step,
            'position': self.position,
            'action': action,
            'position_change': position_change,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'price': current_price,
            'reward': reward
        }
        
        return reward, info
        
    def _calculate_reward(self, portfolio_value):
        """
        Calculate reward based on specified reward function.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            reward value
        """
        # Simple returns
        if len(self.nav_history) > 0:
            returns = (portfolio_value - self.nav_history[-1]) / self.nav_history[-1]
        else:
            returns = 0
            
        # Default to returns reward
        reward = returns
        
        # Apply different reward functions
        if self.reward_function == 'returns':
            reward = returns
        elif self.reward_function == 'sharpe' and len(self.nav_history) > 1:
            # Simple Sharpe ratio approximation
            returns_history = [(self.nav_history[i] - self.nav_history[i-1]) / self.nav_history[i-1]
                              for i in range(1, len(self.nav_history))]
            returns_history.append(returns)
            
            if len(returns_history) > 1 and np.std(returns_history) > 0:
                sharpe = np.mean(returns_history) / np.std(returns_history)
                reward = sharpe
            else:
                reward = returns
        elif self.reward_function == 'sortino' and len(self.nav_history) > 1:
            # Simple Sortino ratio approximation
            returns_history = [(self.nav_history[i] - self.nav_history[i-1]) / self.nav_history[i-1]
                              for i in range(1, len(self.nav_history))]
            returns_history.append(returns)
            
            # Calculate downside deviation (only negative returns)
            neg_returns = [r for r in returns_history if r < 0]
            if len(neg_returns) > 0 and np.std(neg_returns) > 0:
                sortino = np.mean(returns_history) / np.std(neg_returns)
                reward = sortino
            else:
                reward = returns
        
        # Scale reward
        reward *= self.reward_scaling
        
        return reward
        
    def _get_observation(self):
        """
        Get current state observation.
        
        Returns:
            numpy array of state features
        """
        # Get market data for window
        market_data = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # Extract features
        feature_data = market_data[self.features].values
        
        # Normalize features if enabled
        if self.feature_normalization:
            # Normalize each feature to recent history
            for i in range(self.num_features):
                mean = np.mean(feature_data[:, i])
                std = np.std(feature_data[:, i])
                if std > 0:
                    feature_data[:, i] = (feature_data[:, i] - mean) / std
                    
        # Account state
        portfolio_value = self.balance
        if len(self.nav_history) > 0:
            portfolio_value = self.nav_history[-1]
            
        profit_loss = portfolio_value - self.initial_balance
            
        account_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            portfolio_value / self.initial_balance,  # Normalized portfolio value
            profit_loss / self.initial_balance  # Normalized P&L
        ])
        
        # Combine market data and account state
        observation = np.concatenate([feature_data.flatten(), account_state])
        
        return observation
        
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            return
            
        # Plot performance if we have data
        if len(self.nav_history) > 10:
            plt.figure(figsize=(12, 8))
            
            # Plot portfolio value
            plt.subplot(3, 1, 1)
            plt.plot(self.nav_history)
            plt.title('Portfolio Value')
            plt.grid(True)
            
            # Plot position
            plt.subplot(3, 1, 2)
            plt.plot(self.position_history)
            plt.title('Position')
            plt.grid(True)
            
            # Plot reward
            plt.subplot(3, 1, 3)
            plt.plot(self.rewards_history)
            plt.title('Reward')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Step: {self.current_step}, Position: {self.position:.2f}, Balance: {self.balance:.2f}")
            
    def close(self):
        """Close environment."""
        plt.close()
        
        
class MultiAssetTradingEnv(gym.Env):
    """
    Multi-asset cryptocurrency trading environment.
    
    This environment simulates trading across multiple cryptocurrency assets.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        dfs: Dict[str, pd.DataFrame],
        window_size: int = 20,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        reward_function: str = 'sharpe',
        features: Optional[List[str]] = None,
        max_position_per_asset: float = 0.25
    ):
        """
        Initialize multi-asset trading environment.
        
        Args:
            dfs: Dictionary of DataFrames with OHLCV data for each asset
            window_size: Number of past observations to include in state
            initial_balance: Starting account balance
            commission: Trading commission as a fraction
            reward_function: Reward function type ('returns', 'sharpe', 'sortino')
            features: List of column names to use as features
            max_position_per_asset: Maximum position size per asset
        """
        super(MultiAssetTradingEnv, self).__init__()
        
        # Store parameters
        self.dfs = dfs
        self.assets = list(dfs.keys())
        self.num_assets = len(self.assets)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_function = reward_function
        self.max_position_per_asset = max_position_per_asset
        
        # Set default features if not provided
        if features is None:
            self.features = ['open', 'high', 'low', 'close', 'volume']
        else:
            self.features = features
            
        # Check if all required features are in all DataFrames
        for asset, df in self.dfs.items():
            for feature in self.features:
                if feature not in df.columns:
                    raise ValueError(f"Feature '{feature}' not in DataFrame for {asset}")
                    
        # Calculate number of features and minimum observations
        self.num_features = len(self.features)
        self.num_observations = min(len(df) for df in self.dfs.values())
        
        # Define action and observation spaces
        # Action space: continuous values between -1 (max short) and 1 (max long) for each asset
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )
        
        # Observation space: market data features for each asset + account state
        # Market data: window_size * num_features * num_assets
        # Account state: [balance, positions (1 per asset), portfolio_value, profit_loss]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size * self.num_features * self.num_assets + 3 + self.num_assets,), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Reset account state
        self.balance = self.initial_balance
        self.positions = {asset: 0.0 for asset in self.assets}
        self.positions_history = []
        self.balance_history = [self.initial_balance]
        self.nav_history = [self.initial_balance]  # Net asset value
        self.asset_values = {asset: 0.0 for asset in self.assets}
        self.rewards_history = []
        
        # Reset market position
        self.current_step = self.window_size
        
        # Get initial observation
        return self._get_observation()
        
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: Trading actions (position sizes between -1 and 1) for each asset
            
        Returns:
            next_observation, reward, done, info
        """
        # Clip actions to ensure they're within bounds
        actions = np.clip(actions, -self.max_position_per_asset, self.max_position_per_asset)
        
        # Execute the trades for each asset
        reward, info = self._execute_trades(actions)
        
        # Store histories
        self.positions_history.append({k: v for k, v in self.positions.items()})
        self.balance_history.append(self.balance)
        self.nav_history.append(info['portfolio_value'])
        self.rewards_history.append(reward)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.num_observations
        
        # Get new observation
        observation = self._get_observation()
        
        return observation, reward, done, info
        
    def _execute_trades(self, actions):
        """
        Execute trades for all assets based on actions.
        
        Args:
            actions: Position sizes between -1 and 1 for each asset
            
        Returns:
            reward, info
        """
        total_commission = 0.0
        trades_info = {}
        
        # Process each asset
        for i, asset in enumerate(self.assets):
            # Get current price data
            current_price = self.dfs[asset].iloc[self.current_step]['close']
            previous_price = self.dfs[asset].iloc[self.current_step - 1]['close']
            
            # Calculate position change
            previous_position = self.positions[asset]
            target_position = actions[i]
            position_change = target_position - previous_position
            
            # Calculate value change due to price movement
            price_change_ratio = current_price / previous_price
            self.asset_values[asset] = self.asset_values[asset] * price_change_ratio
            
            # Calculate cost to execute trade
            if abs(position_change) > 0.0001:  # Minimum change threshold
                # Calculate trade value based on current portfolio value
                portfolio_value = self.balance + sum(self.asset_values.values())
                trade_value = abs(position_change) * portfolio_value
                commission_cost = trade_value * self.commission
                
                # Update balance and position
                self.balance -= commission_cost
                total_commission += commission_cost
                
                # Update asset value based on new position
                position_value_change = position_change * portfolio_value
                self.balance -= position_value_change
                self.asset_values[asset] += position_value_change
                
                # Update position
                self.positions[asset] = target_position
                
            trades_info[asset] = {
                'position': self.positions[asset],
                'position_change': position_change,
                'price': current_price,
                'asset_value': self.asset_values[asset]
            }
            
        # Calculate total portfolio value
        portfolio_value = self.balance + sum(self.asset_values.values())
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_value)
        
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'positions': self.positions,
            'asset_values': self.asset_values,
            'commission_paid': total_commission,
            'reward': reward,
            'trades': trades_info
        }
        
        return reward, info
        
    def _calculate_reward(self, portfolio_value):
        """
        Calculate reward based on specified reward function.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            reward value
        """
        # Simple returns
        if len(self.nav_history) > 0:
            returns = (portfolio_value - self.nav_history[-1]) / self.nav_history[-1]
        else:
            returns = 0
            
        # Default to returns reward
        reward = returns
        
        # Apply different reward functions
        if self.reward_function == 'returns':
            reward = returns
        elif self.reward_function == 'sharpe' and len(self.nav_history) > 1:
            # Simple Sharpe ratio approximation
            returns_history = [(self.nav_history[i] - self.nav_history[i-1]) / self.nav_history[i-1]
                              for i in range(1, len(self.nav_history))]
            returns_history.append(returns)
            
            if len(returns_history) > 1 and np.std(returns_history) > 0:
                sharpe = np.mean(returns_history) / np.std(returns_history)
                reward = sharpe
            else:
                reward = returns
        elif self.reward_function == 'sortino' and len(self.nav_history) > 1:
            # Simple Sortino ratio approximation
            returns_history = [(self.nav_history[i] - self.nav_history[i-1]) / self.nav_history[i-1]
                              for i in range(1, len(self.nav_history))]
            returns_history.append(returns)
            
            # Calculate downside deviation (only negative returns)
            neg_returns = [r for r in returns_history if r < 0]
            if len(neg_returns) > 0 and np.std(neg_returns) > 0:
                sortino = np.mean(returns_history) / np.std(neg_returns)
                reward = sortino
            else:
                reward = returns
                
        # Add diversity bonus to encourage diversification
        position_sum = sum(abs(p) for p in self.positions.values())
        if position_sum > 0:
            # Herfindahl-Hirschman Index (HHI) for concentration
            # Lower HHI means more diversification
            normalized_positions = [abs(p) / position_sum for p in self.positions.values()]
            hhi = sum(np.square(normalized_positions))
            
            # Diversity bonus (1 - HHI) is higher for more diverse portfolios
            diversity_bonus = (1 - hhi) * 0.01
            reward += diversity_bonus
        
        return reward
        
    def _get_observation(self):
        """
        Get current state observation for all assets.
        
        Returns:
            numpy array of state features
        """
        all_features = []
        
        # Get market data for each asset
        for asset in self.assets:
            # Get market data for window
            market_data = self.dfs[asset].iloc[self.current_step - self.window_size:self.current_step]
            
            # Extract features
            feature_data = market_data[self.features].values
            
            # Normalize features
            for i in range(self.num_features):
                mean = np.mean(feature_data[:, i])
                std = np.std(feature_data[:, i])
                if std > 0:
                    feature_data[:, i] = (feature_data[:, i] - mean) / std
                    
            all_features.append(feature_data.flatten())
            
        # Combine all asset features
        market_features = np.concatenate(all_features)
        
        # Account state
        portfolio_value = self.balance + sum(self.asset_values.values())
        profit_loss = portfolio_value - self.initial_balance
        
        # Get position values as a list in the same order as assets
        positions_list = [self.positions[asset] for asset in self.assets]
        
        account_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            portfolio_value / self.initial_balance,  # Normalized portfolio value
            profit_loss / self.initial_balance,  # Normalized P&L
            *positions_list  # Current positions
        ])
        
        # Combine market data and account state
        observation = np.concatenate([market_features, account_state])
        
        return observation
        
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            return
            
        # Plot performance if we have data
        if len(self.nav_history) > 10:
            plt.figure(figsize=(15, 10))
            
            # Plot portfolio value
            plt.subplot(3, 1, 1)
            plt.plot(self.nav_history)
            plt.title('Portfolio Value')
            plt.grid(True)
            
            # Plot positions for each asset
            plt.subplot(3, 1, 2)
            for asset in self.assets:
                asset_positions = [ph[asset] for ph in self.positions_history]
                plt.plot(asset_positions, label=asset)
            plt.title('Asset Positions')
            plt.legend()
            plt.grid(True)
            
            # Plot reward
            plt.subplot(3, 1, 3)
            plt.plot(self.rewards_history)
            plt.title('Reward')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Portfolio: {self.nav_history[-1]:.2f}")
            for asset in self.assets:
                print(f"  {asset}: Position: {self.positions[asset]:.2f}, Value: {self.asset_values[asset]:.2f}")
                
    def close(self):
        """Close environment."""
        plt.close() 