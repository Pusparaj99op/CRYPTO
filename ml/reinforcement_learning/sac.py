"""
Soft Actor-Critic (SAC) implementation for cryptocurrency trading.

This module provides a SAC agent specifically designed for trading in cryptocurrency markets.
SAC is an off-policy actor-critic deep RL algorithm that maximizes expected reward plus entropy.
It's particularly effective for continuous action spaces and robust to hyperparameter settings.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional
import os
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 100000):
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done_memory = np.zeros(max_size, dtype=np.bool_)
        
        self.max_size = max_size
        self.pointer = 0
        self.size = 0
        
    def store(self, state, action, reward, next_state, done):
        index = self.pointer
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done
        
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size: int):
        if self.size < batch_size:
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
            
        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        next_states = self.next_state_memory[indices]
        dones = self.done_memory[indices]
        
        return states, actions, rewards, next_states, dones
        
class GaussianPolicy(Model):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], 
                 action_bounds: Tuple[float, float], log_std_min: float = -20, 
                 log_std_max: float = 2):
        super(GaussianPolicy, self).__init__()
        
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.hidden_layers = []
        for dim in hidden_dims:
            self.hidden_layers.append(Dense(dim, activation='relu'))
            self.hidden_layers.append(BatchNormalization())
            
        self.mean_layer = Dense(action_dim)
        self.log_std_layer = Dense(action_dim)
        
    def call(self, states):
        x = states
        for layer in self.hidden_layers:
            x = layer(x)
            
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
        
    def sample_action(self, states, deterministic=False):
        mean, log_std = self(states)
        
        if deterministic:
            action = mean
        else:
            std = tf.exp(log_std)
            normal_noise = tf.random.normal(shape=mean.shape)
            action = mean + normal_noise * std
            
        # Squash to [-1, 1] and then scale to action bounds
        action = tf.tanh(action)
        scaled_action = action * (self.action_bounds[1] - self.action_bounds[0])/2 + \
                      (self.action_bounds[1] + self.action_bounds[0])/2
                      
        return scaled_action, action
        
    def log_prob(self, states, actions):
        mean, log_std = self(states)
        std = tf.exp(log_std)
        
        # Inverse tanh transformation (arctanh)
        # Clip for numerical stability
        actions_tanh = tf.clip_by_value(actions, -0.999, 0.999)
        actions_unsquashed = tf.atanh(actions_tanh)
        
        # Gaussian log probability
        log_prob = -0.5 * (
            tf.square((actions_unsquashed - mean) / std) + 
            2 * log_std + 
            tf.math.log(2 * np.pi)
        )
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        
        # Apply change of variables formula (for tanh transformation)
        # d/dx tanh(x) = 1 - tanh^2(x)
        log_det_jacobian = tf.reduce_sum(
            tf.math.log(1 - tf.square(actions_tanh) + 1e-6), 
            axis=1, 
            keepdims=True
        )
        
        log_prob = log_prob - log_det_jacobian
        
        return log_prob

class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1000000,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        target_entropy: Optional[float] = None,
        auto_entropy: bool = True,
        reward_scale: float = 1.0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_bounds = action_bounds
        self.auto_entropy = auto_entropy
        self.reward_scale = reward_scale
        
        # Initialize target entropy (negative of action dimension by default)
        if target_entropy is None:
            self.target_entropy = -np.prod(action_dim)
        else:
            self.target_entropy = target_entropy
            
        # Initialize alpha (entropy coefficient)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        
        # Build policy (actor)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims, action_bounds)
        
        # Build Q-networks (critics)
        self.q1 = self._build_critic(hidden_dims)
        self.q2 = self._build_critic(hidden_dims)
        self.target_q1 = self._build_critic(hidden_dims)
        self.target_q2 = self._build_critic(hidden_dims)
        
        # Copy weights to target networks
        self._update_target_networks(tau=1.0)
        
        # Create optimizers
        self.policy_optimizer = Adam(learning_rate=actor_lr)
        self.q1_optimizer = Adam(learning_rate=critic_lr)
        self.q2_optimizer = Adam(learning_rate=critic_lr)
        self.alpha_optimizer = Adam(learning_rate=alpha_lr)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        
        # Training metrics
        self.policy_losses = []
        self.q1_losses = []
        self.q2_losses = []
        self.alpha_losses = []
        self.q_values = []
        
    def _build_critic(self, hidden_dims: List[int]) -> Model:
        inputs = [
            Input(shape=(self.state_dim,)),
            Input(shape=(self.action_dim,))
        ]
        
        x = Concatenate()(inputs)
        
        for dim in hidden_dims:
            x = Dense(dim, activation='relu')(x)
            x = BatchNormalization()(x)
            
        q_value = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=q_value)
        return model
    
    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
            
        # Update target Q networks with soft update
        q1_weights = self.q1.get_weights()
        target_q1_weights = self.target_q1.get_weights()
        
        new_target_q1_weights = []
        for q1_w, target_q1_w in zip(q1_weights, target_q1_weights):
            new_target_q1_weights.append((1 - tau) * target_q1_w + tau * q1_w)
            
        self.target_q1.set_weights(new_target_q1_weights)
        
        q2_weights = self.q2.get_weights()
        target_q2_weights = self.target_q2.get_weights()
        
        new_target_q2_weights = []
        for q2_w, target_q2_w in zip(q2_weights, target_q2_weights):
            new_target_q2_weights.append((1 - tau) * target_q2_w + tau * q2_w)
            
        self.target_q2.set_weights(new_target_q2_weights)
    
    def get_action(self, state, deterministic=False):
        state = np.array([state], dtype=np.float32)
        scaled_action, _ = self.policy.sample_action(state, deterministic)
        return scaled_action.numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        # Scale reward
        reward *= self.reward_scale
        
        # Store experience in replay buffer
        self.replay_buffer.store(state, action, reward, next_state, done)
    
    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return
            
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Un-scale actions (normalize to [-1, 1])
        norm_actions = (actions - (self.action_bounds[1] + self.action_bounds[0])/2) / \
                      ((self.action_bounds[1] - self.action_bounds[0])/2)
        
        with tf.GradientTape() as q1_tape, tf.GradientTape() as q2_tape:
            # Current Q-values
            q1_values = self.q1([states, norm_actions])
            q2_values = self.q2([states, norm_actions])
            
            # Next actions from current policy
            next_scaled_actions, next_actions = self.policy.sample_action(next_states)
            
            # Next Q-values from target networks
            next_q1 = self.target_q1([next_states, next_actions])
            next_q2 = self.target_q2([next_states, next_actions])
            next_q = tf.minimum(next_q1, next_q2)
            
            # Log probs of next actions
            next_log_probs = self.policy.log_prob(next_states, next_actions)
            
            # Alpha (entropy coefficient)
            alpha = tf.exp(self.log_alpha)
            
            # Compute target Q-values
            next_value = next_q - alpha * next_log_probs
            target_q = rewards[:, None] + self.gamma * (1 - dones[:, None]) * next_value
            
            # Compute Q-network losses
            q1_loss = tf.reduce_mean(tf.square(q1_values - target_q))
            q2_loss = tf.reduce_mean(tf.square(q2_values - target_q))
            
        # Update Q-networks
        q1_grads = q1_tape.gradient(q1_loss, self.q1.trainable_variables)
        self.q1_optimizer.apply_gradients(zip(q1_grads, self.q1.trainable_variables))
        
        q2_grads = q2_tape.gradient(q2_loss, self.q2.trainable_variables)
        self.q2_optimizer.apply_gradients(zip(q2_grads, self.q2.trainable_variables))
        
        # Update policy network
        with tf.GradientTape() as policy_tape:
            # Get actions and log probs from current policy
            _, actions_new = self.policy.sample_action(states)
            log_probs = self.policy.log_prob(states, actions_new)
            
            # Get Q-values for new actions
            q1_new = self.q1([states, actions_new])
            q2_new = self.q2([states, actions_new])
            q_new = tf.minimum(q1_new, q2_new)
            
            # Compute policy loss
            alpha = tf.exp(self.log_alpha)
            policy_loss = tf.reduce_mean(alpha * log_probs - q_new)
            
        policy_grads = policy_tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
        
        # Update alpha (if automatic entropy tuning)
        if self.auto_entropy:
            with tf.GradientTape() as alpha_tape:
                alpha = tf.exp(self.log_alpha)
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * (log_probs + self.target_entropy)
                )
                
            alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        # Update target networks
        self._update_target_networks()
        
        # Store metrics
        self.policy_losses.append(policy_loss.numpy())
        self.q1_losses.append(q1_loss.numpy())
        self.q2_losses.append(q2_loss.numpy())
        self.q_values.append(tf.reduce_mean(q_new).numpy())
        
        return {
            'policy_loss': policy_loss.numpy(),
            'q1_loss': q1_loss.numpy(),
            'q2_loss': q2_loss.numpy(),
            'alpha': alpha.numpy(),
            'q_value': tf.reduce_mean(q_new).numpy()
        }
    
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.policy.save_weights(os.path.join(save_dir, 'policy_weights.h5'))
        self.q1.save_weights(os.path.join(save_dir, 'q1_weights.h5'))
        self.q2.save_weights(os.path.join(save_dir, 'q2_weights.h5'))
        self.target_q1.save_weights(os.path.join(save_dir, 'target_q1_weights.h5'))
        self.target_q2.save_weights(os.path.join(save_dir, 'target_q2_weights.h5'))
        
        # Save alpha
        np.save(os.path.join(save_dir, 'log_alpha.npy'), self.log_alpha.numpy())
        
    def load(self, save_dir: str):
        self.policy.load_weights(os.path.join(save_dir, 'policy_weights.h5'))
        self.q1.load_weights(os.path.join(save_dir, 'q1_weights.h5'))
        self.q2.load_weights(os.path.join(save_dir, 'q2_weights.h5'))
        self.target_q1.load_weights(os.path.join(save_dir, 'target_q1_weights.h5'))
        self.target_q2.load_weights(os.path.join(save_dir, 'target_q2_weights.h5'))
        
        # Load alpha
        log_alpha_value = np.load(os.path.join(save_dir, 'log_alpha.npy'))
        self.log_alpha.assign(log_alpha_value)
    
    def plot_training_metrics(self, save_path: str = None):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.policy_losses)
        plt.title('Policy Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.q1_losses, label='Q1')
        plt.plot(self.q2_losses, label='Q2')
        plt.title('Q Network Losses')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.q_values)
        plt.title('Average Q Values')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.tight_layout()
        plt.show()

class SACTrader:
    """Trading agent that uses the SAC algorithm for continuous action trading decisions."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,  # Continuous action space for position sizing
        initial_capital: float = 10000.0,
        transaction_fee: float = 0.001,
        position_limits: Tuple[float, float] = (-1.0, 1.0),  # Short to long position limits
        risk_tolerance: float = 0.02,
        **sac_params
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.position_limits = position_limits
        self.risk_tolerance = risk_tolerance
        
        # Initialize agent
        self.agent = SACAgent(state_dim, action_dim, action_bounds=position_limits, **sac_params)
        
        # Trading variables
        self.reset()
        
        # Performance tracking
        self.portfolio_values = []
        self.returns = []
        self.positions = []
        self.trades = []
        self.rewards_history = []
        
    def reset(self):
        """Reset the trader to initial state."""
        self.cash = self.initial_capital
        self.position = 0.0  # Current position size (-1.0 to 1.0)
        self.asset_held = 0.0
        self.portfolio_value = self.initial_capital
        self.last_trade_price = 0.0
        self.trade_count = 0
        
        # Clear history
        self.portfolio_values = [self.initial_capital]
        self.returns = [0.0]
        self.positions = [0.0]
        self.trades = []
        self.rewards_history = []
        
    def _calculate_state(self, market_data):
        """
        Calculate state representation from market data.
        
        Override this method to implement custom state representations.
        """
        return market_data
        
    def _calculate_reward(self, old_portfolio_value, new_portfolio_value, action, done=False):
        """
        Calculate reward based on change in portfolio value and risk.
        
        Args:
            old_portfolio_value: Portfolio value before action
            new_portfolio_value: Portfolio value after action
            action: Action taken
            done: Whether episode is done
            
        Returns:
            float: Calculated reward
        """
        # Calculate return (percentage change in portfolio value)
        portfolio_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Base reward on portfolio return
        reward = portfolio_return
        
        # Add penalty for large position changes (to discourage excessive trading)
        position_change = abs(action - self.positions[-2] if len(self.positions) > 1 else action)
        trading_penalty = position_change * self.transaction_fee
        
        # Penalize for excessive risk (large positions)
        risk_penalty = max(0, abs(action) - self.risk_tolerance) * 0.1
        
        return reward - trading_penalty - risk_penalty
        
    def act(self, state, deterministic=False):
        """Get trading action from SAC agent."""
        return self.agent.get_action(state, deterministic)
        
    def step(self, action, market_data, current_price):
        """
        Execute trading action in the environment.
        
        Args:
            action: Trading action (position size between -1 and 1)
            market_data: Current market data (for state calculation)
            current_price: Current asset price
            
        Returns:
            Tuple: (new_state, reward, done, info)
        """
        # Record portfolio value before action
        old_portfolio_value = self.portfolio_value
        
        # Execute trade
        self._execute_trade(action, current_price)
        
        # Calculate new state
        new_state = self._calculate_state(market_data)
        
        # Update portfolio value
        self.portfolio_value = self.cash + self.asset_held * current_price
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate return
        portfolio_return = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        self.returns.append(portfolio_return)
        
        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value, self.portfolio_value, action)
        self.rewards_history.append(reward)
        
        # Information dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'return': portfolio_return,
            'position': self.position,
            'cash': self.cash,
            'asset_held': self.asset_held
        }
        
        # Default done to False (typically controlled by external environment)
        done = False
        
        return new_state, reward, done, info
        
    def _execute_trade(self, target_position, current_price):
        """
        Execute a trade to achieve the target position.
        
        Args:
            target_position: Target position size (-1 to 1)
            current_price: Current asset price
        """
        if current_price <= 0:
            return  # Avoid division by zero or negative prices
            
        # Calculate position change
        current_position = self.position
        position_change = target_position - current_position
        
        if abs(position_change) < 0.01:
            return  # Ignore very small position changes
            
        # Calculate the asset amount to trade
        portfolio_value = self.portfolio_value
        asset_amount_change = (position_change * portfolio_value) / current_price
        
        # Calculate transaction cost
        transaction_cost = abs(asset_amount_change * current_price * self.transaction_fee)
        
        # Execute the trade
        self.cash -= asset_amount_change * current_price + transaction_cost
        self.asset_held += asset_amount_change
        self.position = target_position
        self.positions.append(target_position)
        
        # Record trade
        trade_info = {
            'timestamp': len(self.portfolio_values),
            'price': current_price,
            'position_change': position_change,
            'asset_change': asset_amount_change,
            'transaction_cost': transaction_cost
        }
        self.trades.append(trade_info)
        self.trade_count += 1
        self.last_trade_price = current_price
        
    def train(self):
        """Train the SAC agent on stored experiences."""
        return self.agent.train()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in agent's replay buffer."""
        # Convert single value action to array if needed
        if isinstance(action, (int, float)):
            action = np.array([action])
            
        self.agent.remember(state, action, reward, next_state, done)
        
    def save(self, path):
        """Save the SAC agent."""
        self.agent.save(path)
        
    def load(self, path):
        """Load the SAC agent."""
        self.agent.load(path)
        
    def calculate_performance_metrics(self):
        """Calculate trading performance metrics."""
        portfolio_values = np.array(self.portfolio_values)
        returns = np.array(self.returns)
        
        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized return (assuming daily returns and 252 trading days)
        n_days = len(returns) - 1  # Subtract initial 0 return
        annualized_return = ((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = drawdown.max()
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate
        trade_returns = [trade['price'] / self.trades[i-1]['price'] - 1 
                         if i > 0 and trade['position_change'] != 0 
                         else 0 for i, trade in enumerate(self.trades)]
        win_rate = sum(1 for r in trade_returns if r > 0) / sum(1 for r in trade_returns if r != 0)
        
        # Trading frequency
        trading_frequency = self.trade_count / n_days if n_days > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'trading_frequency': trading_frequency,
            'final_portfolio_value': portfolio_values[-1]
        }
        
    def plot_performance(self, save_path=None):
        """Plot trading performance metrics."""
        metrics = self.calculate_performance_metrics()
        
        plt.figure(figsize=(15, 12))
        
        # Portfolio value
        plt.subplot(3, 1, 1)
        plt.plot(self.portfolio_values)
        plt.title('Portfolio Value')
        plt.grid(True)
        
        # Position over time
        plt.subplot(3, 1, 2)
        plt.plot(self.positions)
        plt.title('Position Size (-1 to 1)')
        plt.grid(True)
        
        # Rewards
        plt.subplot(3, 1, 3)
        plt.plot(self.rewards_history)
        plt.title('Rewards')
        plt.grid(True)
        
        # Add performance metrics as text
        plt.figtext(0.01, 0.01, f"Total Return: {metrics['total_return']:.2%}\n"
                             f"Annualized Return: {metrics['annualized_return']:.2%}\n"
                             f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                             f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                             f"Win Rate: {metrics['win_rate']:.2%}\n"
                             f"Trading Frequency: {metrics['trading_frequency']:.2f} trades/day",
                   fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show() 