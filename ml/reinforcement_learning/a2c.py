"""
Advantage Actor-Critic (A2C) implementation for cryptocurrency trading.

This module provides an A2C-based reinforcement learning agent designed for
both discrete and continuous action spaces in cryptocurrency trading environments.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, Lambda, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from typing import List, Tuple, Dict, Union, Optional, Any
import matplotlib.pyplot as plt

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent for reinforcement learning in trading environments.
    
    Implements a synchronized A2C with shared networks between actor and critic.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        continuous_actions: bool = True,
        shared_network: bool = True,
        actor_learning_rate: float = 0.0003,
        critic_learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_sizes: List[int] = [128, 64]
    ):
        """
        Initialize the A2C agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            continuous_actions: Whether actions are continuous (default) or discrete
            shared_network: Whether to use a shared network for actor and critic
            actor_learning_rate: Learning rate for actor network
            critic_learning_rate: Learning rate for critic network
            gamma: Discount factor for future rewards
            entropy_coef: Entropy coefficient for exploration bonus
            value_coef: Value function coefficient in loss
            max_grad_norm: Maximum gradient norm for gradient clipping
            hidden_sizes: List of hidden layer sizes for networks
        """
        self.state_size = state_size
        self.action_size = action_size
        self.continuous_actions = continuous_actions
        self.shared_network = shared_network
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.hidden_sizes = hidden_sizes
        
        # Initialize optimizer
        self.optimizer = Adam(learning_rate=self.actor_lr)
        
        # Build model
        self.model = self._build_model()
        
        # Training metrics
        self.loss_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.reward_history = []
        
    def _build_model(self) -> Model:
        """
        Build neural network model for A2C.
        
        Returns:
            Keras model with multiple outputs
        """
        # Input layer
        state_input = Input(shape=(self.state_size,))
        
        # Shared network
        if self.shared_network:
            x = state_input
            for size in self.hidden_sizes:
                x = Dense(size, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.1)(x)
                
            shared_features = x
            
            # Actor output
            if self.continuous_actions:
                mu = Dense(self.action_size, activation='tanh')(shared_features)  # Mean (scaled to [-1, 1])
                log_std = Dense(self.action_size, activation='linear')(shared_features)
                log_std = Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)  # Clip log standard deviation
                # Critic output (state value)
                value = Dense(1, activation='linear')(shared_features)
                
                model = Model(inputs=state_input, outputs=[mu, log_std, value])
            else:
                # Discrete actions: output action logits
                action_probs = Dense(self.action_size, activation='softmax')(shared_features)
                # Critic output (state value)
                value = Dense(1, activation='linear')(shared_features)
                
                model = Model(inputs=state_input, outputs=[action_probs, value])
        else:
            # Separate networks for actor and critic
            # Actor network
            x_actor = state_input
            for size in self.hidden_sizes:
                x_actor = Dense(size, activation='relu')(x_actor)
                x_actor = BatchNormalization()(x_actor)
                x_actor = Dropout(0.1)(x_actor)
                
            if self.continuous_actions:
                mu = Dense(self.action_size, activation='tanh')(x_actor)
                log_std = Dense(self.action_size, activation='linear')(x_actor)
                log_std = Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
            else:
                action_probs = Dense(self.action_size, activation='softmax')(x_actor)
                
            # Critic network
            x_critic = state_input
            for size in self.hidden_sizes:
                x_critic = Dense(size, activation='relu')(x_critic)
                x_critic = BatchNormalization()(x_critic)
                x_critic = Dropout(0.1)(x_critic)
                
            value = Dense(1, activation='linear')(x_critic)
            
            if self.continuous_actions:
                model = Model(inputs=state_input, outputs=[mu, log_std, value])
            else:
                model = Model(inputs=state_input, outputs=[action_probs, value])
                
        # Compile model with custom loss
        model.compile(optimizer=self.optimizer)
        
        return model
        
    def get_policy_and_value(self, state: np.ndarray) -> Tuple:
        """
        Get policy distribution and value estimate for a state.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (policy_distribution, value)
        """
        state = np.reshape(state, [1, self.state_size])
        
        if self.continuous_actions:
            mu, log_std, value = self.model.predict(state, verbose=0)
            mu = mu[0]
            std = np.exp(log_std[0])
            value = value[0, 0]
            
            # Create multivariate normal distribution
            return tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=std), value
        else:
            action_probs, value = self.model.predict(state, verbose=0)
            return tfp.distributions.Categorical(probs=action_probs[0]), value[0, 0]
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Choose action based on current state.
        
        Args:
            state: Current state
            deterministic: Whether to act deterministically (use mean)
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        policy, value = self.get_policy_and_value(state)
        
        if deterministic or not self.continuous_actions:
            if self.continuous_actions:
                action = policy.mean().numpy()
            else:
                action = np.argmax(policy.probs_parameter().numpy())
        else:
            action = policy.sample().numpy()
            
        log_prob = policy.log_prob(action).numpy()
        
        # For continuous actions, clip to action space boundaries
        if self.continuous_actions:
            action = np.clip(action, -1.0, 1.0)
            
        return action, log_prob, value
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for state.
        
        Args:
            state: Current state
            
        Returns:
            Estimated state value
        """
        state = np.reshape(state, [1, self.state_size])
        
        if self.continuous_actions:
            _, _, value = self.model.predict(state, verbose=0)
        else:
            _, value = self.model.predict(state, verbose=0)
            
        return value[0, 0]
    
    def train_step(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        next_states: np.ndarray, 
        dones: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform a single training step on a batch of experiences.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of terminal flags
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        if self.continuous_actions:
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        else:
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Calculate discounted returns and advantages
        next_values = np.zeros_like(rewards)
        for i in range(len(next_states)):
            next_values[i] = self.get_value(next_states[i])
            
        next_values = tf.convert_to_tensor(next_values, dtype=tf.float32)
        
        # Calculate returns and advantages
        returns = rewards + self.gamma * next_values * (1 - dones)
        
        with tf.GradientTape() as tape:
            # Forward pass
            if self.continuous_actions:
                mu, log_std, values = self.model(states, training=True)
                values = tf.squeeze(values)
                
                # Create normal distribution
                std = tf.exp(log_std)
                policy = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=std)
                
                # Calculate log probabilities and entropy
                log_probs = policy.log_prob(actions)
                entropy = tf.reduce_mean(policy.entropy())
            else:
                action_probs, values = self.model(states, training=True)
                values = tf.squeeze(values)
                
                # Create categorical distribution
                policy = tfp.distributions.Categorical(probs=action_probs)
                
                # Calculate log probabilities and entropy
                log_probs = tf.reduce_sum(policy.log_prob(actions))
                entropy = tf.reduce_mean(policy.entropy())
            
            # Calculate advantages
            advantages = returns - values
            
            # Calculate actor (policy) loss
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
            
            # Calculate critic (value) loss
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Add entropy bonus for exploration
            entropy_loss = -self.entropy_coef * entropy
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss + entropy_loss
            
        # Get gradients and apply
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Record metrics
        self.actor_loss_history.append(actor_loss.numpy())
        self.critic_loss_history.append(critic_loss.numpy())
        self.entropy_history.append(entropy.numpy())
        self.loss_history.append(total_loss.numpy())
        
        # Return metrics
        return {
            "total_loss": total_loss.numpy(),
            "actor_loss": actor_loss.numpy(),
            "critic_loss": critic_loss.numpy(),
            "entropy": entropy.numpy()
        }
        
    def train(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        next_states: np.ndarray, 
        dones: np.ndarray
    ) -> Dict[str, float]:
        """
        Train A2C agent on a batch of experience.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of terminal flags
            
        Returns:
            Dictionary of training metrics
        """
        metrics = self.train_step(states, actions, rewards, next_states, dones)
        
        # Record total reward for the episode
        if len(rewards) > 0:
            episode_reward = np.sum(rewards)
            self.reward_history.append(episode_reward)
            
        return metrics
        
    def save(self, filepath: str) -> None:
        """Save model to file."""
        self.model.save(filepath)
        
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = tf.keras.models.load_model(filepath)
        
    def plot_metrics(self) -> None:
        """Plot training metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot total loss
        ax1.plot(self.loss_history)
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot actor loss
        ax2.plot(self.actor_loss_history)
        ax2.set_title('Actor Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # Plot critic loss
        ax3.plot(self.critic_loss_history)
        ax3.set_title('Critic Loss')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Plot entropy
        ax4.plot(self.entropy_history)
        ax4.set_title('Policy Entropy')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Entropy')
        ax4.grid(True)
        
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


class A2CTrader:
    """
    A2C-based cryptocurrency trading system that implements both discrete and 
    continuous action spaces for trading decisions.
    """
    
    def __init__(
        self,
        state_size: int,
        num_assets: int = 1,
        continuous_actions: bool = True,
        learning_rate: float = 0.0003,
        risk_adjustment: float = 0.1,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        max_position_size: float = 1.0,
        reward_scaling: float = 0.1,
        n_discrete_actions: int = 3  # Sell, Hold, Buy
    ):
        """
        Initialize the A2C Trader.
        
        Args:
            state_size: Dimension of state space
            num_assets: Number of tradable assets
            continuous_actions: Whether to use continuous or discrete actions
            learning_rate: Learning rate for A2C model
            risk_adjustment: Risk adjustment factor for reward calculation
            initial_balance: Starting account balance
            transaction_fee: Transaction fee as a percentage
            max_position_size: Maximum position size as a fraction of portfolio
            reward_scaling: Scaling factor for rewards
            n_discrete_actions: Number of discrete actions per asset (if using discrete actions)
        """
        self.state_size = state_size
        self.num_assets = num_assets
        self.continuous_actions = continuous_actions
        self.learning_rate = learning_rate
        self.risk_adjustment = risk_adjustment
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position_size = max_position_size
        self.reward_scaling = reward_scaling
        
        # For continuous actions, action space is position size for each asset
        # For discrete actions, action space is [sell, hold, buy] for each asset
        if continuous_actions:
            action_size = num_assets  # Position sizes
        else:
            self.n_discrete_actions = n_discrete_actions
            action_size = n_discrete_actions * num_assets  # Discrete actions per asset
        
        # Initialize A2C agent
        self.agent = A2CAgent(
            state_size=state_size,
            action_size=action_size,
            continuous_actions=continuous_actions,
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate * 2
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
        
    def prepare_state(self, market_data: np.ndarray) -> np.ndarray:
        """
        Prepare state input for the A2C agent from market data.
        
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
        
    def act(self, state: np.ndarray, prices: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Execute trading action based on current state.
        
        Args:
            state: Current market state
            prices: Current asset prices
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action taken
        """
        self.current_prices = prices
        self.current_step += 1
        
        # Get action from A2C agent
        action, _, _ = self.agent.act(state, deterministic)
        
        if self.continuous_actions:
            # For continuous actions, the output is directly the target position size
            # Scale from [-1, 1] to [-max_position, max_position]
            target_positions = action * self.max_position_size
        else:
            # For discrete actions, convert to position changes
            target_positions = np.zeros(self.num_assets)
            
            for i in range(self.num_assets):
                action_idx = np.argmax(action[i * self.n_discrete_actions:(i + 1) * self.n_discrete_actions])
                
                # Interpret discrete action
                if action_idx == 0:  # Sell
                    target_positions[i] = -self.max_position_size  # Sell all
                elif action_idx == 1:  # Hold
                    target_positions[i] = 0  # No change
                elif action_idx == 2:  # Buy
                    target_positions[i] = self.max_position_size  # Buy maximum
                    
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
        
    def train(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        next_states: np.ndarray, 
        dones: np.ndarray
    ) -> Dict[str, float]:
        """
        Train the A2C agent.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of terminal flags
            
        Returns:
            Dictionary of training metrics
        """
        return self.agent.train(states, actions, rewards, next_states, dones)
    
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
        
        if self.continuous_actions:
            # For continuous actions
            for i in range(self.num_assets):
                plt.plot(actions[:, i], label=f'Asset {i+1} Position')
        else:
            # For discrete actions, we need to extract the chosen action
            discrete_actions = np.zeros((len(actions), self.num_assets))
            
            for i in range(len(actions)):
                for j in range(self.num_assets):
                    action_slice = actions[i, j*self.n_discrete_actions:(j+1)*self.n_discrete_actions]
                    if isinstance(action_slice, np.ndarray):
                        discrete_actions[i, j] = np.argmax(action_slice) - 1  # -1, 0, 1 for sell, hold, buy
                    else:
                        # Handle the case where we stored the actual action
                        discrete_actions[i, j] = action_slice
            
            for i in range(self.num_assets):
                plt.plot(discrete_actions[:, i], label=f'Asset {i+1} Action')
                
        plt.title('Agent Actions Over Time')
        plt.xlabel('Trading Step')
        plt.ylabel('Action Value')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
        
        # Display metrics
        metrics = self.get_portfolio_metrics()
        
        print("Portfolio Performance Metrics:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Annualized Volatility: {metrics['volatility']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
    def save(self, filepath: str) -> None:
        """Save trader model to file."""
        self.agent.save(filepath)
        
    def load(self, filepath: str) -> None:
        """Load trader model from file."""
        self.agent.load(filepath) 