"""
Deep Deterministic Policy Gradient (DDPG) implementation for cryptocurrency trading.

This module provides a DDPG agent designed for continuous action spaces in
cryptocurrency trading environments, using actor-critic architecture with
target networks for stable learning.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Optional, Any
import random
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling batches of experiences.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store in the buffer
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class OUActionNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.
    Used for exploration in continuous action spaces.
    """
    
    def __init__(
        self, 
        mean: float = 0.0, 
        std_dev: float = 0.2, 
        theta: float = 0.15, 
        dt: float = 0.01, 
        x_initial: Optional[np.ndarray] = None
    ):
        """
        Initialize Ornstein-Uhlenbeck noise process.
        
        Args:
            mean: Mean of the noise process
            std_dev: Standard deviation of the noise process
            theta: Rate of mean reversion
            dt: Time step
            x_initial: Initial state of the noise process
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        
    def __call__(self) -> np.ndarray:
        """Generate noise sample."""
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
        
    def reset(self):
        """Reset the noise process to the initial state."""
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent for reinforcement learning 
    in continuous action spaces for cryptocurrency trading.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        actor_learning_rate: float = 0.0001,
        critic_learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.001,
        batch_size: int = 64,
        buffer_size: int = 100000,
        noise_std: float = 0.1,
        l2_reg: float = 0.01,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        actor_hidden_layers: List[int] = [400, 300],
        critic_hidden_layers: List[int] = [400, 300]
    ):
        """
        Initialize DDPG agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            actor_learning_rate: Learning rate for actor network
            critic_learning_rate: Learning rate for critic network
            gamma: Discount factor for future rewards
            tau: Soft update coefficient for target networks
            batch_size: Mini-batch size for training
            buffer_size: Size of replay buffer
            noise_std: Standard deviation of exploration noise
            l2_reg: L2 regularization factor
            action_bounds: Tuple of (min_action, max_action)
            actor_hidden_layers: List of hidden layer sizes for actor network
            critic_hidden_layers: List of hidden layer sizes for critic network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.noise_std = noise_std
        self.l2_reg = l2_reg
        self.action_low, self.action_high = action_bounds
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Build networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Make target networks identical to main networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Initialize optimizers
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
        
        # Initialize noise process
        self.noise = OUActionNoise(mean=np.zeros(action_size), std_dev=float(noise_std) * np.ones(action_size))
        
        # Training metrics
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.reward_history = []
        self.q_value_history = []
        
    def _build_actor(self) -> Model:
        """
        Build actor network that maps states to actions.
        
        Returns:
            Keras model of actor network
        """
        # Input layer (state)
        inputs = Input(shape=(self.state_size,))
        
        # Hidden layers
        x = inputs
        for i, units in enumerate(self.actor_hidden_layers):
            x = Dense(
                units, 
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                name=f'actor_hidden_{i}'
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
        # Output layer (action)
        outputs = Dense(
            self.action_size, 
            activation='tanh',
            kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003),
            name='actor_output'
        )(x)
        
        # Scale to action bounds
        outputs = outputs  # Already scaled to [-1, 1] by tanh
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        return model
        
    def _build_critic(self) -> Model:
        """
        Build critic network that maps state-action pairs to Q-values.
        
        Returns:
            Keras model of critic network
        """
        # Input layers
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(self.action_size,))
        
        # Process state input
        state_x = state_input
        for i, units in enumerate(self.critic_hidden_layers[:-1]):
            state_x = Dense(
                units, 
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                name=f'critic_state_hidden_{i}'
            )(state_x)
            state_x = BatchNormalization()(state_x)
            
        # Merge state and action pathways
        merged = Concatenate()([state_x, action_input])
        
        # Process merged pathway
        x = merged
        for i, units in enumerate(self.critic_hidden_layers[-1:]):
            x = Dense(
                units, 
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                name=f'critic_merged_hidden_{i}'
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
        # Output layer (Q-value)
        outputs = Dense(
            1, 
            activation='linear',
            kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003),
            name='critic_output'
        )(x)
        
        # Create model
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        
        return model
        
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Choose action based on current state.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state = np.reshape(state, [1, self.state_size])
        action = self.actor.predict(state, verbose=0)[0]
        
        if add_noise:
            noise = self.noise()
            action += noise
            
        # Clip action to bounds
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
        
    def train(self, experiences: Optional[Tuple] = None) -> Dict[str, float]:
        """
        Train DDPG agent using experiences from replay buffer.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
                        If None, sample from replay buffer
            
        Returns:
            Dictionary of training metrics
        """
        if experiences is None:
            # Sample experiences from replay buffer
            if len(self.buffer) < self.batch_size:
                # Not enough experiences to train
                return {
                    "actor_loss": 0.0,
                    "critic_loss": 0.0,
                    "mean_q_value": 0.0
                }
                
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = experiences
            
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Get target actions from target actor
            target_actions = self.target_actor(next_states, training=True)
            
            # Get target Q values from target critic
            target_q_values = self.target_critic([next_states, target_actions], training=True)
            target_q_values = tf.squeeze(target_q_values)
            
            # Compute TD targets
            td_targets = rewards + self.gamma * target_q_values * (1 - dones)
            
            # Get current Q values from critic
            current_q_values = self.critic([states, actions], training=True)
            current_q_values = tf.squeeze(current_q_values)
            
            # Compute critic loss
            critic_loss = tf.reduce_mean(tf.square(td_targets - current_q_values))
            
        # Update critic
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            # Get actions from actor
            actor_actions = self.actor(states, training=True)
            
            # Get Q values from critic
            actor_q_values = self.critic([states, actor_actions], training=True)
            
            # Compute actor loss (negative mean Q value)
            actor_loss = -tf.reduce_mean(actor_q_values)
            
        # Update actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Soft update of target networks
        self._update_target_networks()
        
        # Record metrics
        self.actor_loss_history.append(actor_loss.numpy())
        self.critic_loss_history.append(critic_loss.numpy())
        self.q_value_history.append(tf.reduce_mean(current_q_values).numpy())
        
        return {
            "actor_loss": actor_loss.numpy(),
            "critic_loss": critic_loss.numpy(),
            "mean_q_value": tf.reduce_mean(current_q_values).numpy()
        }
        
    def _update_target_networks(self):
        """
        Soft update of target networks.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        # Update target actor
        for local_var, target_var in zip(self.actor.variables, self.target_actor.variables):
            target_var.assign(self.tau * local_var + (1 - self.tau) * target_var)
            
        # Update target critic
        for local_var, target_var in zip(self.critic.variables, self.target_critic.variables):
            target_var.assign(self.tau * local_var + (1 - self.tau) * target_var)
            
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)
        
    def save(self, actor_path: str, critic_path: str) -> None:
        """Save models to files."""
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        
    def load(self, actor_path: str, critic_path: str) -> None:
        """Load models from files."""
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)
        self.target_actor = tf.keras.models.load_model(actor_path)
        self.target_critic = tf.keras.models.load_model(critic_path)
        
    def plot_metrics(self) -> None:
        """Plot training metrics."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot actor loss
        ax1.plot(self.actor_loss_history)
        ax1.set_title('Actor Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot critic loss
        ax2.plot(self.critic_loss_history)
        ax2.set_title('Critic Loss')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # Plot Q values
        ax3.plot(self.q_value_history)
        ax3.set_title('Mean Q Value')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Q Value')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot rewards if available
        if len(self.reward_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.reward_history)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.show()


class DDPGTrader:
    """
    DDPG-based cryptocurrency trading system that implements continuous action
    spaces for determining position sizes.
    """
    
    def __init__(
        self,
        state_size: int,
        num_assets: int = 1,
        learning_rate: float = 0.0001,
        risk_adjustment: float = 0.1,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        max_position_size: float = 1.0,
        reward_scaling: float = 0.1,
        batch_size: int = 64,
        buffer_size: int = 100000,
        noise_std: float = 0.1
    ):
        """
        Initialize the DDPG Trader.
        
        Args:
            state_size: Dimension of state space
            num_assets: Number of tradable assets
            learning_rate: Learning rate for DDPG model
            risk_adjustment: Risk adjustment factor for reward calculation
            initial_balance: Starting account balance
            transaction_fee: Transaction fee as a percentage
            max_position_size: Maximum position size as a fraction of portfolio
            reward_scaling: Scaling factor for rewards
            batch_size: Mini-batch size for training
            buffer_size: Size of replay buffer
            noise_std: Standard deviation of exploration noise
        """
        self.state_size = state_size
        self.num_assets = num_assets
        self.learning_rate = learning_rate
        self.risk_adjustment = risk_adjustment
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position_size = max_position_size
        self.reward_scaling = reward_scaling
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.noise_std = noise_std
        
        # Initialize DDPG agent
        self.agent = DDPGAgent(
            state_size=state_size,
            action_size=num_assets,  # Position sizes for each asset
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate * 2,
            batch_size=batch_size,
            buffer_size=buffer_size,
            noise_std=noise_std
        )
        
        # Trading state variables
        self.reset()
        
        # Performance metrics
        self.performance_history = {
            'portfolio_value': [],
            'cash_balance': [],
            'asset_holdings': [],
            'actions': [],
            'returns': [],
            'sharpe_ratio': None
        }
        
    def reset(self) -> None:
        """Reset trading state."""
        self.balance = self.initial_balance
        self.asset_holdings = np.zeros(self.num_assets)
        self.current_prices = np.ones(self.num_assets)
        self.previous_portfolio_value = self.initial_balance
        self.current_step = 0
        self.total_reward = 0
        self.daily_returns = []
        self.transaction_history = []
        
        # Reset exploration noise
        self.agent.noise.reset()
        
    def prepare_state(self, market_data: np.ndarray) -> np.ndarray:
        """
        Prepare state input for the DDPG agent from market data.
        
        Args:
            market_data: Market data including price, volume, etc.
            
        Returns:
            Processed state representation
        """
        # Get portfolio information
        portfolio_value = self.get_portfolio_value()
        portfolio_weights = self.get_portfolio_weights()
        
        # Combine market data with portfolio information
        # This creates a state that includes both external market conditions
        # and internal portfolio state
        state = np.concatenate([
            market_data.flatten(),
            portfolio_weights,
            [self.balance / portfolio_value]  # Cash ratio
        ])
        
        return state
        
    def act(self, state: np.ndarray, prices: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Execute trading action based on current state.
        
        Args:
            state: Current market state
            prices: Current asset prices
            training: Whether to use exploration noise
            
        Returns:
            Action taken
        """
        self.current_prices = prices
        self.current_step += 1
        
        # Get action from DDPG agent
        action = self.agent.act(state, add_noise=training)
        
        # For continuous actions, the output is directly the target position size ratio
        # Scale from [-1, 1] to [-max_position, max_position]
        target_positions = action * self.max_position_size
            
        # Execute trades to achieve target positions
        portfolio_value = self.get_portfolio_value()
        
        for i in range(self.num_assets):
            current_position_value = self.asset_holdings[i] * self.current_prices[i]
            current_position_ratio = current_position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate target position value
            target_position_ratio = target_positions[i]
            target_position_value = portfolio_value * target_position_ratio
            
            # Calculate trade amount
            trade_value = target_position_value - current_position_value
            
            # Execute trade
            if trade_value > 0:  # Buy
                # Calculate fee
                trade_amount = trade_value / self.current_prices[i]
                fee = trade_value * self.transaction_fee
                
                if self.balance >= (trade_value + fee):
                    self.asset_holdings[i] += trade_amount
                    self.balance -= (trade_value + fee)
                    self.transaction_history.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'asset': i,
                        'amount': trade_amount,
                        'price': self.current_prices[i],
                        'value': trade_value,
                        'fee': fee
                    })
            
            elif trade_value < 0:  # Sell
                # Calculate fee
                trade_amount = -trade_value / self.current_prices[i]
                
                if self.asset_holdings[i] >= trade_amount:
                    self.asset_holdings[i] -= trade_amount
                    sell_value = trade_amount * self.current_prices[i]
                    fee = sell_value * self.transaction_fee
                    self.balance += (sell_value - fee)
                    self.transaction_history.append({
                        'step': self.current_step,
                        'type': 'sell',
                        'asset': i,
                        'amount': trade_amount,
                        'price': self.current_prices[i],
                        'value': sell_value,
                        'fee': fee
                    })
        
        # Update performance history
        new_portfolio_value = self.get_portfolio_value()
        daily_return = (new_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        self.daily_returns.append(daily_return)
        
        self.performance_history['portfolio_value'].append(new_portfolio_value)
        self.performance_history['cash_balance'].append(self.balance)
        self.performance_history['asset_holdings'].append(self.asset_holdings.copy())
        self.performance_history['actions'].append(action.copy() if isinstance(action, np.ndarray) else action)
        self.performance_history['returns'].append(daily_return)
        
        self.previous_portfolio_value = new_portfolio_value
        
        return action
        
    def calculate_reward(self) -> float:
        """
        Calculate reward for the current step.
        
        Returns:
            Reward value
        """
        # Calculate portfolio return
        portfolio_value = self.get_portfolio_value()
        portfolio_return = (portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        
        # Apply reward scaling
        scaled_return = portfolio_return * self.reward_scaling
        
        # Calculate Sharpe ratio component if we have enough returns
        if len(self.daily_returns) >= 20:
            recent_returns = self.daily_returns[-20:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns) + 1e-8  # Avoid division by zero
            sharpe_component = mean_return / std_return
            
            # Update Sharpe ratio in performance history
            self.performance_history['sharpe_ratio'] = sharpe_component
            
            # Risk-adjusted reward
            risk_adjusted_reward = scaled_return + self.risk_adjustment * sharpe_component
        else:
            risk_adjusted_reward = scaled_return
            
        # Calculate transaction cost penalty (from last timestep)
        transaction_cost = 0
        for transaction in self.transaction_history:
            if transaction['step'] == self.current_step:
                transaction_cost += transaction['fee']
                
        # Final reward
        reward = risk_adjusted_reward - transaction_cost * 0.1
        
        self.total_reward += reward
        return reward
        
    def train_step(self, state, action, reward, next_state, done) -> Dict[str, float]:
        """
        Train the DDPG agent on a single experience.
        
        Args:
            state: Current state
            action: Executed action
            reward: Received reward
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Add experience to replay buffer
        self.agent.remember(state, action, reward, next_state, done)
        
        # Train agent
        metrics = self.agent.train()
        
        # Record reward history
        if done:
            self.agent.reward_history.append(self.total_reward)
            
        return metrics
        
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.balance + np.sum(self.asset_holdings * self.current_prices)
    
    def get_portfolio_weights(self) -> np.ndarray:
        """Get current portfolio weights."""
        portfolio_value = self.get_portfolio_value()
        if portfolio_value == 0:
            return np.zeros(self.num_assets)
        
        return (self.asset_holdings * self.current_prices) / portfolio_value
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate returns
        returns = np.array(self.performance_history['returns'])
        
        # Calculate metrics
        metrics = {
            'total_return': (self.get_portfolio_value() / self.initial_balance) - 1,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'profit_factor': 0
        }
        
        if len(returns) > 1:
            # Calculate Sharpe ratio (annualized)
            annual_factor = 252  # Trading days in a year
            avg_return = np.mean(returns) * annual_factor
            std_return = np.std(returns) * np.sqrt(annual_factor)
            
            if std_return > 0:
                metrics['sharpe_ratio'] = avg_return / std_return
                
            # Calculate Sortino ratio (using only negative returns for denominator)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(annual_factor)
                if downside_deviation > 0:
                    metrics['sortino_ratio'] = avg_return / downside_deviation
            
            # Calculate maximum drawdown
            portfolio_values = np.array(self.performance_history['portfolio_value'])
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            metrics['max_drawdown'] = np.min(drawdown)
            
            # Calculate annualized volatility
            metrics['volatility'] = std_return
            
            # Calculate profit factor
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) > 0 and np.sum(np.abs(negative_returns)) > 0:
                metrics['profit_factor'] = np.sum(positive_returns) / np.sum(np.abs(negative_returns))
            
        return metrics
    
    def plot_performance(self) -> None:
        """Plot trading performance metrics."""
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot portfolio value
        ax1.plot(self.performance_history['portfolio_value'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Trading Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Plot asset allocation
        steps = range(len(self.performance_history['cash_balance']))
        
        # Prepare data for stacked plot
        allocations = [self.performance_history['cash_balance']]
        labels = ['Cash']
        
        # Create allocation array for each asset
        for i in range(self.num_assets):
            asset_values = []
            for j, holdings in enumerate(self.performance_history['asset_holdings']):
                asset_values.append(holdings[i] * self.current_prices[i])
            allocations.append(asset_values)
            labels.append(f'Asset {i+1}')
        
        ax2.stackplot(steps, *allocations, labels=labels, alpha=0.7)
        ax2.set_title('Asset Allocation Over Time')
        ax2.set_xlabel('Trading Step')
        ax2.set_ylabel('Value ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Plot returns
        ax3.plot(self.performance_history['returns'])
        ax3.set_title('Daily Returns')
        ax3.set_xlabel('Trading Step')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot additional metrics
        # Create figure for action plots
        plt.figure(figsize=(12, 6))
        
        # Plot actions
        actions = np.array(self.performance_history['actions'])
        
        # For continuous actions
        for i in range(self.num_assets):
            plt.plot(actions[:, i], label=f'Asset {i+1} Position')
                
        plt.title('Agent Actions Over Time')
        plt.xlabel('Trading Step')
        plt.ylabel('Action Value')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
        
        # Plot agent metrics
        self.agent.plot_metrics()
        
        # Display metrics
        metrics = self.get_portfolio_metrics()
        
        print("Portfolio Performance Metrics:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Annualized Volatility: {metrics['volatility']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
    def save(self, filepath_prefix: str) -> None:
        """Save trader model to files."""
        actor_path = f"{filepath_prefix}_actor.h5"
        critic_path = f"{filepath_prefix}_critic.h5"
        self.agent.save(actor_path, critic_path)
        
    def load(self, filepath_prefix: str) -> None:
        """Load trader model from files."""
        actor_path = f"{filepath_prefix}_actor.h5"
        critic_path = f"{filepath_prefix}_critic.h5"
        self.agent.load(actor_path, critic_path) 