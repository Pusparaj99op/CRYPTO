"""
Deep Q-Network (DQN) implementation for cryptocurrency trading.

This module provides DQN-based reinforcement learning agents specialized for
making discrete trading decisions in cryptocurrency markets.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from typing import List, Tuple, Dict, Union, Optional, Any
import matplotlib.pyplot as plt

class DQNAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning in trading environments.
    
    Implements core DQN algorithm with experience replay and target network.
    """
    
    def __init__(
        self, 
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        exploration_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_model_update: int = 5,
        dueling_network: bool = False,
        double_dqn: bool = True
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for the optimizer
            discount_factor: Discount factor (gamma) for future rewards
            exploration_rate: Initial exploration rate
            exploration_decay: Decay rate for exploration
            exploration_min: Minimum exploration rate
            batch_size: Batch size for training
            memory_size: Size of experience replay memory
            target_model_update: Frequency to update target network
            dueling_network: Whether to use dueling network architecture
            double_dqn: Whether to use double DQN algorithm
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_model_update = target_model_update
        self.dueling_network = dueling_network
        self.double_dqn = double_dqn
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
        self.q_value_history = []
        
        # Step counter for target model update
        self.steps = 0
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        # Initialize target model with same weights
        self.update_target_model()
        
    def _build_model(self) -> Model:
        """
        Build neural network model for DQN.
        
        Returns:
            Keras model for Q-function approximation
        """
        if self.dueling_network:
            return self._build_dueling_model()
        
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def _build_dueling_model(self) -> Model:
        """
        Build dueling network architecture for DQN.
        
        Returns:
            Keras model with dueling architecture
        """
        input_layer = Input(shape=(self.state_size,))
        
        # Shared network
        x = Dense(64, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Value stream
        value_stream = Dense(32, activation='relu')(x)
        value = Dense(1, activation='linear')(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(32, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine value and advantage streams
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self) -> None:
        """Update target model with weights from main model."""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action based on current state using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses exploration)
            
        Returns:
            Selected action index
        """
        if training and np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]
        self.q_value_history.append(np.mean(q_values))
        return np.argmax(q_values)
    
    def replay(self) -> float:
        """
        Train model using experience replay.
        
        Returns:
            Loss value from training
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            target = reward
            
            if not done:
                if self.double_dqn:
                    # Double DQN: Use main model to select action, target model to estimate value
                    next_action = np.argmax(self.model.predict(np.array([next_state]), verbose=0)[0])
                    target += self.discount_factor * self.target_model.predict(
                        np.array([next_state]), verbose=0)[0][next_action]
                else:
                    # Standard DQN
                    target += self.discount_factor * np.amax(
                        self.target_model.predict(np.array([next_state]), verbose=0)[0])
            
            # Create targets for the batch
            targets[i] = self.model.predict(np.array([state]), verbose=0)[0]
            targets[i][action] = target
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Update exploration rate
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            
        # Update target model periodically
        self.steps += 1
        if self.steps % self.target_model_update == 0:
            self.update_target_model()
            
        return loss
        
    def save(self, filepath: str) -> None:
        """Save model to file."""
        self.model.save(filepath)
        
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_model()
        
    def plot_metrics(self) -> None:
        """Plot training metrics."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot loss
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot rewards
        ax2.plot(self.reward_history)
        ax2.set_title('Episode Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.grid(True)
        
        # Plot average Q-values
        ax3.plot(self.q_value_history)
        ax3.set_title('Average Q-Value')
        ax3.set_xlabel('Action Selection')
        ax3.set_ylabel('Mean Q-Value')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()


class DQNTrader:
    """
    DQN-based cryptocurrency trading system that combines the DQN agent
    with trading-specific logic for market actions.
    """
    
    def __init__(
        self, 
        state_size: int,
        action_size: int = 3,  # buy, sell, hold
        learning_rate: float = 0.0001,
        risk_tolerance: float = 0.02,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        window_size: int = 10,
        use_cnn: bool = False
    ):
        """
        Initialize the DQN Trader.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space (default: 3 for buy, sell, hold)
            learning_rate: Learning rate for the optimizer
            risk_tolerance: Maximum portfolio percentage to risk per trade
            initial_balance: Starting account balance
            transaction_fee: Transaction fee as a percentage
            window_size: Window size for price history features
            use_cnn: Whether to use CNN architecture for feature extraction
        """
        self.state_size = state_size
        self.action_size = action_size
        self.risk_tolerance = risk_tolerance
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.use_cnn = use_cnn
        
        # Initialize DQN agent
        if use_cnn:
            self.agent = self._build_cnn_agent(learning_rate)
        else:
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                dueling_network=True,
                double_dqn=True
            )
        
        # Trading state variables
        self.reset()
        
        # Performance metrics
        self.performance_history = {
            'portfolio_value': [],
            'cash_balance': [],
            'asset_holdings': [],
            'trades': []
        }
        
    def _build_cnn_agent(self, learning_rate: float) -> DQNAgent:
        """
        Build a DQN agent with CNN architecture for time series data.
        
        Args:
            learning_rate: Learning rate for the optimizer
            
        Returns:
            DQN agent with CNN model
        """
        class CNNDQNAgent(DQNAgent):
            def _build_model(self) -> Model:
                # Reshape input for CNN (assuming 1D time series data)
                input_shape = (self.window_size, self.state_size // self.window_size)
                
                model = Sequential()
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', 
                                input_shape=input_shape))
                model.add(BatchNormalization())
                model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
                model.add(BatchNormalization())
                model.add(Flatten())
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(self.action_size, activation='linear'))
                
                model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
                return model
                
        return CNNDQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=learning_rate,
            dueling_network=False,  # CNN architecture already defined
            double_dqn=True
        )
        
    def reset(self) -> None:
        """Reset trading state."""
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.current_price = 0
        self.total_trades = 0
        self.total_profit = 0
        self.current_step = 0
        
    def prepare_state(self, market_data: np.ndarray) -> np.ndarray:
        """
        Prepare state input for the DQN agent from market data.
        
        Args:
            market_data: Market data including price, volume, etc.
            
        Returns:
            Processed state representation
        """
        # Extract relevant features from market data
        # This could include price, volume, technical indicators, etc.
        if self.use_cnn:
            # Reshape for CNN input
            return market_data.reshape(1, self.window_size, -1)
        else:
            # For dense network, flatten the features
            return market_data.flatten()
        
    def act(self, state: np.ndarray, price: float, training: bool = True) -> Tuple[int, str, float]:
        """
        Execute trading action based on current state.
        
        Args:
            state: Current market state
            price: Current asset price
            training: Whether in training mode
            
        Returns:
            Tuple of (action_index, action_type, amount_traded)
        """
        self.current_price = price
        self.current_step += 1
        
        # Get action from DQN agent
        action = self.agent.act(state, training)
        
        # Map action to trading decision
        action_type = ["buy", "sell", "hold"][action]
        amount_traded = 0
        
        # Execute trading action
        if action_type == "buy" and self.balance > 0:
            # Calculate buy amount based on risk tolerance
            max_buy = self.balance / price
            buy_amount = min(max_buy, self.balance * self.risk_tolerance / price)
            
            # Apply transaction fee
            fee = buy_amount * price * self.transaction_fee
            amount_spent = buy_amount * price + fee
            
            if amount_spent <= self.balance:
                self.crypto_held += buy_amount
                self.balance -= amount_spent
                amount_traded = buy_amount
                self.total_trades += 1
                
                # Record trade
                self.performance_history['trades'].append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': price,
                    'amount': buy_amount,
                    'value': amount_spent,
                    'fee': fee
                })
        
        elif action_type == "sell" and self.crypto_held > 0:
            # Calculate sell amount based on risk tolerance
            sell_amount = min(self.crypto_held, self.crypto_held * self.risk_tolerance)
            
            # Apply transaction fee
            proceed = sell_amount * price
            fee = proceed * self.transaction_fee
            amount_received = proceed - fee
            
            self.crypto_held -= sell_amount
            self.balance += amount_received
            amount_traded = sell_amount
            self.total_trades += 1
            
            # Calculate profit/loss from this trade
            # (simplified - assumes FIFO for cost basis)
            trade_profit = proceed - (sell_amount / self.crypto_held) * self.initial_balance
            self.total_profit += trade_profit
            
            # Record trade
            self.performance_history['trades'].append({
                'step': self.current_step,
                'type': 'sell',
                'price': price,
                'amount': sell_amount,
                'value': proceed,
                'fee': fee,
                'profit': trade_profit
            })
            
        # Update performance history
        portfolio_value = self.balance + (self.crypto_held * price)
        self.performance_history['portfolio_value'].append(portfolio_value)
        self.performance_history['cash_balance'].append(self.balance)
        self.performance_history['asset_holdings'].append(self.crypto_held)
        
        return action, action_type, amount_traded
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in agent's memory."""
        self.agent.remember(state, action, reward, next_state, done)
        
    def train(self) -> float:
        """Train the agent and return loss."""
        return self.agent.replay()
        
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.balance + (self.crypto_held * self.current_price)
    
    def get_portfolio_returns(self) -> float:
        """Calculate portfolio returns."""
        current_value = self.get_portfolio_value()
        return (current_value - self.initial_balance) / self.initial_balance
    
    def plot_performance(self) -> None:
        """Plot trading performance metrics."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot portfolio value
        ax1.plot(self.performance_history['portfolio_value'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Trading Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Plot asset allocation
        steps = range(len(self.performance_history['cash_balance']))
        asset_values = [h * p for h, p in zip(
            self.performance_history['asset_holdings'],
            [self.current_price] * len(self.performance_history['asset_holdings'])
        )]
        
        ax2.stackplot(steps, 
            self.performance_history['cash_balance'],
            asset_values,
            labels=['Cash', 'Crypto'],
            alpha=0.7
        )
        ax2.set_title('Asset Allocation Over Time')
        ax2.set_xlabel('Trading Step')
        ax2.set_ylabel('Value ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Plot trade occurrences
        buy_steps = [t['step'] for t in self.performance_history['trades'] if t['type'] == 'buy']
        buy_values = [t['value'] for t in self.performance_history['trades'] if t['type'] == 'buy']
        
        sell_steps = [t['step'] for t in self.performance_history['trades'] if t['type'] == 'sell']
        sell_values = [t['value'] for t in self.performance_history['trades'] if t['type'] == 'sell']
        
        ax3.plot(self.performance_history['portfolio_value'], color='gray', alpha=0.5)
        ax3.scatter(buy_steps, [self.performance_history['portfolio_value'][s] for s in buy_steps], 
                   color='green', label='Buy', marker='^', s=100)
        ax3.scatter(sell_steps, [self.performance_history['portfolio_value'][s] for s in sell_steps], 
                   color='red', label='Sell', marker='v', s=100)
        ax3.set_title('Trade Occurrences')
        ax3.set_xlabel('Trading Step')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def save(self, filepath: str) -> None:
        """Save the trader's agent model to file."""
        self.agent.save(filepath)
        
    def load(self, filepath: str) -> None:
        """Load the trader's agent model from file."""
        self.agent.load(filepath) 