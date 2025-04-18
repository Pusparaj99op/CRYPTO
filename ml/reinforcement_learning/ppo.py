"""
Proximal Policy Optimization (PPO) algorithm implementation for cryptocurrency trading.

This module provides a PPO agent specifically designed for trading in cryptocurrency markets.
PPO is an on-policy algorithm that uses a clipped surrogate objective function to ensure
stable policy updates while maintaining good sample efficiency.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional
import os

class PPOBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int = 2048):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        self.pointer = 0
        self.size = 0
        self.buffer_size = buffer_size
        
    def store(self, state, action, reward, value, log_prob, done, next_state):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.values[self.pointer] = value
        self.log_probs[self.pointer] = log_prob
        self.dones[self.pointer] = done
        self.next_states[self.pointer] = next_state
        
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def compute_advantages_and_returns(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """Compute Generalized Advantage Estimation (GAE) and returns."""
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
                
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
        
        self.returns = self.advantages + self.values
        
    def get_minibatch(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            tf.convert_to_tensor(self.states[indices]),
            tf.convert_to_tensor(self.actions[indices]),
            tf.convert_to_tensor(self.returns[indices]),
            tf.convert_to_tensor(self.advantages[indices]),
            tf.convert_to_tensor(self.log_probs[indices])
        )
        
    def clear(self):
        self.pointer = 0
        self.size = 0

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden_units: List[int] = [128, 64],
        critic_hidden_units: List[int] = [128, 64],
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        batch_size: int = 64,
        buffer_size: int = 2048,
        epochs: int = 10,
        action_bounds: Optional[Tuple[float, float]] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.epochs = epochs
        self.action_bounds = action_bounds if action_bounds else (-1.0, 1.0)
        
        # Build actor (policy) and critic (value) networks
        self.actor = self._build_actor(actor_hidden_units)
        self.critic = self._build_critic(critic_hidden_units)
        
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)
        
        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size)
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []
        
    def _build_actor(self, hidden_units: List[int]) -> Model:
        inputs = Input(shape=(self.state_dim,))
        x = inputs
        
        for units in hidden_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            
        # Mean and log standard deviation for continuous actions
        mean = Dense(self.action_dim, activation='tanh')(x)
        log_std = Dense(self.action_dim, activation='linear', 
                        kernel_initializer=tf.keras.initializers.constant(-0.5))(x)
        
        model = Model(inputs=inputs, outputs=[mean, log_std], name='actor')
        return model
        
    def _build_critic(self, hidden_units: List[int]) -> Model:
        inputs = Input(shape=(self.state_dim,))
        x = inputs
        
        for units in hidden_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            
        value = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=value, name='critic')
        return model
    
    def get_action(self, state, deterministic=False):
        state = np.array([state], dtype=np.float32)
        mean, log_std = self.actor(state)
        
        if deterministic:
            # For evaluation, use the mean action
            action = mean.numpy()[0]
        else:
            # For training, sample from the distribution
            std = tf.exp(log_std)
            distribution = tf.random.normal(shape=mean.shape)
            action = mean + distribution * std
            action = action.numpy()[0]
            
        # Clip action to bounds
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        
        # Get log probability of action
        log_prob = self._log_prob(mean, log_std, action)
        
        # Get value estimate
        value = self.critic(state).numpy()[0, 0]
        
        return action, log_prob, value
    
    def _log_prob(self, mean, log_std, action):
        std = tf.exp(log_std)
        logprob = -0.5 * tf.reduce_sum(
            tf.square((action - mean) / std) + 2 * log_std + np.log(2 * np.pi), 
            axis=-1, keepdims=True
        )
        return logprob
    
    def train(self, last_value):
        # Compute advantages and returns
        self.buffer.compute_advantages_and_returns(last_value, self.gamma, self.lam)
        
        # Normalize advantages
        advantages = self.buffer.advantages[:self.buffer.size]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        self.buffer.advantages[:self.buffer.size] = advantages
        
        # Training for multiple epochs
        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []
        
        for _ in range(self.epochs):
            for _ in range(self.buffer.size // self.batch_size):
                states, actions, returns, advantages, old_log_probs = self.buffer.get_minibatch(self.batch_size)
                
                with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                    # Get new log probabilities
                    mean, log_std = self.actor(states)
                    new_log_probs = self._log_prob(mean, log_std, actions)
                    
                    # Calculate ratio and clipped ratio
                    ratio = tf.exp(new_log_probs - old_log_probs)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    
                    # Entropy for exploration
                    std = tf.exp(log_std)
                    entropy = tf.reduce_mean(log_std + 0.5 * np.log(2 * np.pi * np.e))
                    
                    # Policy loss
                    surrogate1 = ratio * advantages
                    surrogate2 = clipped_ratio * advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                    
                    # Value loss
                    values = self.critic(states)
                    critic_loss = tf.reduce_mean(tf.square(returns - values))
                    
                    # Total loss
                    total_loss = actor_loss - self.entropy_coef * entropy + self.vf_coef * critic_loss
                
                # Compute gradients and apply updates
                actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                
                # Store metrics
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                entropy_losses.append(entropy.numpy())
                total_losses.append(total_loss.numpy())
        
        # Store average losses
        self.actor_losses.append(np.mean(actor_losses))
        self.critic_losses.append(np.mean(critic_losses))
        self.entropy_losses.append(np.mean(entropy_losses))
        self.total_losses.append(np.mean(total_losses))
        
        # Clear buffer after training
        self.buffer.clear()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses)
        }
    
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.actor.save_weights(os.path.join(save_dir, 'ppo_actor.h5'))
        self.critic.save_weights(os.path.join(save_dir, 'ppo_critic.h5'))
        
    def load(self, save_dir: str):
        self.actor.load_weights(os.path.join(save_dir, 'ppo_actor.h5'))
        self.critic.load_weights(os.path.join(save_dir, 'ppo_critic.h5'))
        
    def plot_training_metrics(self, save_path: str = None):
        """Plot training metrics over time."""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(self.actor_losses)
        ax1.set_ylabel('Actor Loss')
        ax1.grid(True)
        
        ax2.plot(self.critic_losses)
        ax2.set_ylabel('Critic Loss')
        ax2.grid(True)
        
        ax3.plot(self.entropy_losses)
        ax3.set_ylabel('Entropy')
        ax3.grid(True)
        
        ax4.plot(self.total_losses)
        ax4.set_ylabel('Total Loss')
        ax4.set_xlabel('Training Steps')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

class PPOTrader:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        initial_capital: float = 10000,
        transaction_fee: float = 0.001,
        **ppo_params
    ):
        self.agent = PPOAgent(state_dim, action_dim, **ppo_params)
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        
        # Portfolio state
        self.reset()
        
        # Performance metrics
        self.portfolio_values = []
        self.returns = []
        self.sharpe_ratio = None
        self.max_drawdown = None
        
    def reset(self):
        """Reset the trading environment."""
        self.portfolio_value = self.initial_capital
        self.cash_balance = self.initial_capital
        self.holdings = np.zeros(self.agent.action_dim)
        self.previous_portfolio_value = self.initial_capital
        
        # Trading stats
        self.trades = 0
        self.profitable_trades = 0
        
        # Performance tracking
        self.portfolio_values = [self.initial_capital]
        self.returns = []
        
        return self._get_state(None)
    
    def _get_state(self, market_data):
        """Convert market data and portfolio state into agent state."""
        if market_data is None:
            # Initial state with zeros for market data
            return np.zeros(self.agent.state_dim)
        
        # Prepare market features
        market_features = self._prepare_market_features(market_data)
        
        # Add portfolio state
        portfolio_state = np.concatenate([
            [self.cash_balance / self.initial_capital],  # Normalized cash
            self.holdings  # Current asset holdings
        ])
        
        # Combine market and portfolio features
        return np.concatenate([market_features, portfolio_state])
    
    def _prepare_market_features(self, market_data):
        """Extract relevant features from market data."""
        # Implement feature extraction based on market data
        # Example: prices, volumes, technical indicators, etc.
        # This needs to be implemented based on the specific format of market_data
        return market_data
    
    def act(self, state, deterministic=False):
        """Get trading action from the agent."""
        action, log_prob, value = self.agent.get_action(state, deterministic)
        return action, log_prob, value
    
    def step(self, action, market_data, current_prices):
        """Execute trading action and calculate reward."""
        # Current portfolio value before action
        pre_action_value = self.portfolio_value
        
        # Execute the trade
        self._execute_trade(action, current_prices)
        
        # Update portfolio value
        self.portfolio_value = self.cash_balance + np.sum(self.holdings * current_prices)
        
        # Calculate reward (e.g., change in portfolio value)
        reward = (self.portfolio_value - pre_action_value) / pre_action_value
        
        # Update trading statistics
        if self.portfolio_value > self.previous_portfolio_value:
            self.profitable_trades += 1
        
        self.previous_portfolio_value = self.portfolio_value
        
        # Update performance metrics
        self.portfolio_values.append(self.portfolio_value)
        daily_return = (self.portfolio_value / self.portfolio_values[-2]) - 1 if len(self.portfolio_values) > 1 else 0
        self.returns.append(daily_return)
        
        # Get new state
        new_state = self._get_state(market_data)
        
        # Check if episode is done (e.g., bankruptcy or end of data)
        done = self.portfolio_value <= 0
        
        return new_state, reward, done
    
    def _execute_trade(self, action, current_prices):
        """Execute the trading action."""
        # action is a vector of target portfolio weights for each asset
        action = np.clip(action, -1, 1)  # Ensure actions are in allowed range
        
        for i, target_weight in enumerate(action):
            current_value = self.holdings[i] * current_prices[i]
            target_value = target_weight * self.portfolio_value
            
            # Calculate the difference in value
            value_diff = target_value - current_value
            
            # Skip tiny trades
            if abs(value_diff) < 1e-5 * self.portfolio_value:
                continue
                
            # Buy or sell
            if value_diff > 0:  # Buy
                # Calculate shares to buy
                shares_to_buy = value_diff / current_prices[i]
                # Apply transaction fee
                cost = value_diff * (1 + self.transaction_fee)
                
                # Execute if enough cash
                if cost <= self.cash_balance:
                    self.holdings[i] += shares_to_buy
                    self.cash_balance -= cost
                    self.trades += 1
            else:  # Sell
                # Calculate shares to sell
                shares_to_sell = -value_diff / current_prices[i]
                
                # Execute if enough holdings
                if shares_to_sell <= self.holdings[i]:
                    self.holdings[i] -= shares_to_sell
                    # Apply transaction fee
                    self.cash_balance += -value_diff * (1 - self.transaction_fee)
                    self.trades += 1
    
    def train_on_batch(self, last_value):
        """Train the agent on collected experiences."""
        return self.agent.train(last_value)
    
    def store_experience(self, state, action, reward, value, log_prob, done, next_state):
        """Store experience in agent's replay buffer."""
        self.agent.buffer.store(state, action, reward, value, log_prob, done, next_state)
    
    def save(self, path):
        """Save the agent's models."""
        self.agent.save(path)
    
    def load(self, path):
        """Load the agent's models."""
        self.agent.load(path)
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics for the trading strategy."""
        returns = np.array(self.returns)
        
        # Annualized return (assuming daily returns)
        annual_return = np.mean(returns) * 252
        
        # Annualized volatility
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        self.sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1)
        self.max_drawdown = np.max(drawdown)
        
        # Win rate
        win_rate = self.profitable_trades / self.trades if self.trades > 0 else 0
        
        return {
            "annualized_return": annual_return,
            "annualized_volatility": annual_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": win_rate,
            "total_trades": self.trades,
            "profitable_trades": self.profitable_trades
        }
    
    def plot_performance(self, save_path=None):
        """Plot portfolio performance over time."""
        metrics = self.calculate_performance_metrics()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Portfolio value
        ax1.plot(self.portfolio_values)
        ax1.set_title(f'Portfolio Value Over Time\nSharpe: {metrics["sharpe_ratio"]:.2f}, Max Drawdown: {metrics["max_drawdown"]:.2%}')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Returns
        ax2.plot(self.returns)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.set_title(f'Daily Returns\nAnn. Return: {metrics["annualized_return"]:.2%}, Ann. Vol: {metrics["annualized_volatility"]:.2%}')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show() 