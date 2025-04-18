import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from enum import Enum
import time

class ExecutionStrategy(Enum):
    """Enum for different execution strategies"""
    TWAP = 0
    VWAP = 1
    POV = 2
    ADAPTIVE = 3
    IMPLEMENTATION_SHORTFALL = 4
    CUSTOM = 5

class MLExecution:
    """
    Machine Learning-based execution algorithm.
    
    Uses machine learning techniques to optimize order execution by
    dynamically selecting and parameterizing execution strategies
    based on market conditions and historical performance.
    """
    
    def __init__(
        self, 
        symbol: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        side: str,
        model: Optional[Any] = None,
        feature_extractor: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        available_strategies: Optional[List[ExecutionStrategy]] = None,
        initial_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
        exploration_rate: float = 0.1,
        learning_rate: float = 0.01,
        price_limit: Optional[float] = None,
        feature_lookback: int = 20
    ):
        """
        Initialize ML-based execution algorithm.
        
        Args:
            symbol: Trading pair symbol
            total_quantity: Total quantity to execute
            start_time: Algorithm start time
            end_time: Algorithm end time
            side: Order side ('buy' or 'sell')
            model: Optional pre-trained ML model
            feature_extractor: Optional function to extract features from market data
            reward_function: Optional function to calculate rewards for RL
            available_strategies: List of available execution strategies
            initial_strategy: Initial execution strategy to use
            exploration_rate: Exploration rate for strategy selection
            learning_rate: Learning rate for model updates
            price_limit: Optional limit price for the orders
            feature_lookback: Number of periods to look back for features
        """
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.side = side.lower()
        self.price_limit = price_limit
        self.model = model  # Can be any ML model (e.g., sklearn, tensorflow, etc.)
        self.feature_extractor = feature_extractor or self._default_feature_extractor
        self.reward_function = reward_function or self._default_reward_function
        self.exploration_rate = max(0.0, min(1.0, exploration_rate))
        self.learning_rate = learning_rate
        self.feature_lookback = max(5, feature_lookback)
        
        # Available execution strategies
        self.available_strategies = available_strategies or [
            ExecutionStrategy.TWAP, 
            ExecutionStrategy.VWAP,
            ExecutionStrategy.POV,
            ExecutionStrategy.ADAPTIVE
        ]
        
        # Current execution strategy
        self.current_strategy = initial_strategy
        self.strategy_params = self._get_default_strategy_params()
        
        # Market data history
        self.price_history = []
        self.volume_history = []
        self.time_history = []
        self.spread_history = []
        self.depth_history = []
        
        # Learning data
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Execution state
        self.is_running = False
        self.executed_quantity = 0.0
        self.remaining_quantity = total_quantity
        self.execution_history = []
        self.strategy_performance = {strategy.name: [] for strategy in ExecutionStrategy}
        
        # Initialize models if needed
        self._initialize_ml_components()
        
    def _initialize_ml_components(self):
        """Initialize machine learning components if not provided"""
        # If no model is provided, create a simple Q-learning table
        if self.model is None:
            # Simple Q-table for strategy selection
            # States: discretized market conditions (e.g., high/low volatility, trending/ranging)
            # Actions: available strategies
            self.model = {
                "q_table": {},  # State -> Action -> Value mapping
                "state_discretizer": self._discretize_state,
                "strategy_selector": self._select_strategy_from_q_table
            }
    
    def _get_default_strategy_params(self) -> Dict:
        """
        Get default parameters for the current strategy.
        
        Returns:
            Dictionary of strategy parameters
        """
        if self.current_strategy == ExecutionStrategy.TWAP:
            return {
                "num_slices": 10,
                "random_variance": 0.05
            }
        elif self.current_strategy == ExecutionStrategy.VWAP:
            return {
                "max_participation_rate": 0.15,
                "min_participation_rate": 0.02
            }
        elif self.current_strategy == ExecutionStrategy.POV:
            return {
                "target_participation_rate": 0.1,
                "adaptive_rate": True
            }
        elif self.current_strategy == ExecutionStrategy.ADAPTIVE:
            return {
                "aggression_level": 0.5,
                "volatility_threshold": 0.02
            }
        elif self.current_strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
            return {
                "urgency": 0.5,
                "risk_aversion": 1.0
            }
        else:  # CUSTOM
            return {}
    
    def _default_feature_extractor(self, market_data: Dict) -> np.ndarray:
        """
        Default feature extractor for market data.
        
        Args:
            market_data: Dictionary of market data
            
        Returns:
            Numpy array of extracted features
        """
        # Extract basic features from market data
        features = []
        
        # Price features
        if len(self.price_history) >= 2:
            # Recent returns
            returns = np.diff(self.price_history[-min(10, len(self.price_history)):]) / self.price_history[-min(10, len(self.price_history)):-1]
            
            # Volatility
            volatility = np.std(returns) if len(returns) > 0 else 0
            features.append(volatility)
            
            # Price momentum (slope of recent prices)
            if len(self.price_history) >= 5:
                y = np.array(self.price_history[-5:])
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
                momentum = slope / np.mean(y) if np.mean(y) > 0 else 0
                features.append(momentum)
            else:
                features.append(0)
                
            # Price level relative to day range
            if len(self.price_history) >= self.feature_lookback:
                price_range = max(self.price_history[-self.feature_lookback:]) - min(self.price_history[-self.feature_lookback:])
                if price_range > 0:
                    price_level = (self.price_history[-1] - min(self.price_history[-self.feature_lookback:])) / price_range
                    features.append(price_level)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
        else:
            # Not enough price history
            features.extend([0, 0, 0.5])
            
        # Volume features
        if len(self.volume_history) >= 2:
            # Volume momentum
            if len(self.volume_history) >= 5:
                y = np.array(self.volume_history[-5:])
                x = np.arange(len(y))
                volume_slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
                volume_momentum = volume_slope / np.mean(y) if np.mean(y) > 0 else 0
                features.append(volume_momentum)
            else:
                features.append(0)
                
            # Relative volume
            if len(self.volume_history) >= self.feature_lookback:
                avg_volume = np.mean(self.volume_history[-self.feature_lookback:])
                rel_volume = self.volume_history[-1] / avg_volume if avg_volume > 0 else 1
                features.append(rel_volume)
            else:
                features.append(1)
        else:
            # Not enough volume history
            features.extend([0, 1])
            
        # Spread and depth features if available
        if len(self.spread_history) > 0:
            features.append(self.spread_history[-1])
        else:
            features.append(0)
            
        if len(self.depth_history) > 0:
            features.append(self.depth_history[-1])
        else:
            features.append(0)
            
        # Time features
        total_time = (self.end_time - self.start_time).total_seconds()
        if 'current_time' in market_data:
            elapsed_time = (market_data['current_time'] - self.start_time).total_seconds()
            time_progress = elapsed_time / total_time if total_time > 0 else 0
            features.append(time_progress)
        else:
            features.append(0)
            
        # Execution progress
        execution_progress = self.executed_quantity / self.total_quantity if self.total_quantity > 0 else 0
        features.append(execution_progress)
        
        return np.array(features)
    
    def _discretize_state(self, features: np.ndarray) -> str:
        """
        Discretize state features into a string key for the Q-table.
        
        Args:
            features: Numpy array of continuous features
            
        Returns:
            String key representing the discretized state
        """
        if len(features) < 4:
            return "default"
            
        # Extract key features
        volatility = features[0]
        momentum = features[1]
        volume_momentum = features[3]
        time_progress = features[-2]
        
        # Discretize into categories
        vol_state = "high_vol" if volatility > 0.02 else "low_vol"
        
        if momentum > 0.005:
            trend_state = "up_trend"
        elif momentum < -0.005:
            trend_state = "down_trend"
        else:
            trend_state = "ranging"
            
        volume_state = "high_vol" if volume_momentum > 0.05 else "low_vol"
        
        time_state = "early" if time_progress < 0.3 else "mid" if time_progress < 0.7 else "late"
        
        # Combine into a state key
        state_key = f"{vol_state}_{trend_state}_{volume_state}_{time_state}"
        
        return state_key
    
    def _select_strategy_from_q_table(self, state_key: str) -> Tuple[ExecutionStrategy, Dict]:
        """
        Select strategy based on Q-table values.
        
        Args:
            state_key: Discretized state key
            
        Returns:
            Tuple of (selected_strategy, strategy_params)
        """
        # Initialize state in Q-table if not present
        if state_key not in self.model["q_table"]:
            self.model["q_table"][state_key] = {
                strategy.name: 0.0 for strategy in self.available_strategies
            }
            
        # Epsilon-greedy strategy selection
        if np.random.random() < self.exploration_rate:
            # Exploration: random strategy
            selected_strategy = np.random.choice(self.available_strategies)
        else:
            # Exploitation: best strategy according to Q-table
            q_values = self.model["q_table"][state_key]
            best_strategy_name = max(q_values, key=q_values.get)
            selected_strategy = ExecutionStrategy[best_strategy_name]
            
        # Get parameters for the selected strategy
        params = self._get_strategy_params(selected_strategy, state_key)
        
        return selected_strategy, params
    
    def _get_strategy_params(self, strategy: ExecutionStrategy, state_key: str) -> Dict:
        """
        Get optimized parameters for the given strategy and state.
        
        Args:
            strategy: Selected execution strategy
            state_key: Current state key
            
        Returns:
            Dictionary of strategy parameters
        """
        # In a more sophisticated implementation, this would use ML to
        # optimize strategy parameters based on the state.
        # For simplicity, we return default parameters with some adaptations
        
        base_params = self._get_default_strategy_params()
        
        # Simple adaptations based on state
        if state_key.startswith("high_vol"):
            # In high volatility, adjust parameters
            if strategy == ExecutionStrategy.TWAP:
                base_params["num_slices"] = max(15, base_params["num_slices"])
            elif strategy == ExecutionStrategy.POV:
                base_params["target_participation_rate"] = min(0.05, base_params["target_participation_rate"])
            elif strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
                base_params["risk_aversion"] = max(1.5, base_params["risk_aversion"])
                
        if "up_trend" in state_key and self.side == 'buy' or "down_trend" in state_key and self.side == 'sell':
            # Unfavorable trend, be more aggressive
            if strategy == ExecutionStrategy.ADAPTIVE:
                base_params["aggression_level"] = min(0.8, base_params["aggression_level"] * 1.5)
            elif strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
                base_params["urgency"] = min(0.9, base_params["urgency"] * 1.5)
                
        if "late" in state_key:
            # Late in execution window, increase urgency
            if strategy == ExecutionStrategy.TWAP:
                base_params["num_slices"] = max(3, base_params["num_slices"] // 2)
            elif strategy == ExecutionStrategy.POV:
                base_params["target_participation_rate"] = min(0.3, base_params["target_participation_rate"] * 2)
            elif strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
                base_params["urgency"] = min(0.9, base_params["urgency"] * 1.5)
                
        return base_params
        
    def _default_reward_function(self, state: np.ndarray, action: ExecutionStrategy, 
                               next_state: np.ndarray, execution_data: Dict) -> float:
        """
        Default reward function for reinforcement learning.
        
        Args:
            state: State features before action
            action: Execution strategy used
            next_state: State features after action
            execution_data: Data about the execution
            
        Returns:
            Reward value
        """
        # No reward if no execution took place
        if not execution_data.get("executed", False):
            return 0.0
            
        # Key execution metrics
        quantity = execution_data.get("quantity", 0)
        price = execution_data.get("price", 0)
        
        # Calculate price improvement/slippage
        if len(self.price_history) > 1:
            initial_price = self.price_history[0]
            if self.side == 'buy':
                # For buys, lower price is better
                price_improvement = (initial_price - price) / initial_price if initial_price > 0 else 0
            else:
                # For sells, higher price is better
                price_improvement = (price - initial_price) / initial_price if initial_price > 0 else 0
        else:
            price_improvement = 0
            
        # Calculate execution rate quality
        time_elapsed_pct = next_state[-2] - state[-2] if len(next_state) > 1 and len(state) > 1 else 0
        quantity_executed_pct = quantity / self.total_quantity if self.total_quantity > 0 else 0
        
        # Time-normalized execution rate
        # Higher is better if we're behind schedule, lower is better if ahead
        execution_schedule_quality = 0
        if time_elapsed_pct > 0:
            execution_progress = self.executed_quantity / self.total_quantity if self.total_quantity > 0 else 0
            expected_progress = next_state[-2]  # Time progress should roughly match execution progress
            progress_difference = abs(execution_progress - expected_progress)
            execution_schedule_quality = 1.0 - min(1.0, progress_difference)
            
        # Combine components with weights
        reward = (
            0.6 * price_improvement +  # Price improvement/slippage
            0.3 * execution_schedule_quality +  # How well we're following the schedule
            0.1 * (quantity_executed_pct / max(0.001, time_elapsed_pct))  # Efficiency of execution
        )
        
        return reward
    
    def start(self):
        """Start the ML-based execution"""
        if self.is_running:
            return
            
        self.is_running = True
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.execution_history = []
        self.price_history = []
        self.volume_history = []
        self.time_history = []
        self.spread_history = []
        self.depth_history = []
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Initialize current strategy
        self.current_strategy = ExecutionStrategy.ADAPTIVE  # Default starting strategy
    
    def stop(self):
        """Stop the ML-based execution"""
        self.is_running = False
        
        # Save learning data if needed
        self._update_model_from_execution()
    
    def update_market_data(self, market_data: Dict) -> None:
        """
        Update market data history.
        
        Args:
            market_data: Dictionary containing market data
        """
        if not self.is_running:
            return
            
        # Update histories
        if 'price' in market_data:
            self.price_history.append(market_data['price'])
            
        if 'volume' in market_data:
            self.volume_history.append(market_data['volume'])
            
        if 'time' in market_data:
            self.time_history.append(market_data['time'])
            
        if 'spread' in market_data:
            self.spread_history.append(market_data['spread'])
            
        if 'depth' in market_data:
            self.depth_history.append(market_data['depth'])
            
        # Keep histories within lookback window
        if len(self.price_history) > self.feature_lookback:
            self.price_history = self.price_history[-self.feature_lookback:]
            
        if len(self.volume_history) > self.feature_lookback:
            self.volume_history = self.volume_history[-self.feature_lookback:]
            
        if len(self.time_history) > self.feature_lookback:
            self.time_history = self.time_history[-self.feature_lookback:]
            
        if len(self.spread_history) > self.feature_lookback:
            self.spread_history = self.spread_history[-self.feature_lookback:]
            
        if len(self.depth_history) > self.feature_lookback:
            self.depth_history = self.depth_history[-self.feature_lookback:]
    
    def _update_model_from_execution(self):
        """Update ML model based on execution history"""
        # Skip if we don't have sufficient data
        if len(self.state_history) < 2 or len(self.action_history) < 1 or len(self.reward_history) < 1:
            return
            
        # Simple Q-learning update
        if hasattr(self.model, "q_table"):
            for i in range(len(self.action_history)):
                state_key = self.state_history[i]
                action = self.action_history[i]
                reward = self.reward_history[i]
                
                # Initialize state in Q-table if not present
                if state_key not in self.model["q_table"]:
                    self.model["q_table"][state_key] = {
                        strategy.name: 0.0 for strategy in self.available_strategies
                    }
                
                # Q-learning update rule
                old_value = self.model["q_table"][state_key][action.name]
                new_value = old_value + self.learning_rate * (reward - old_value)
                self.model["q_table"][state_key][action.name] = new_value
    
    def select_strategy(self, market_data: Dict) -> None:
        """
        Select execution strategy based on current market conditions.
        
        Args:
            market_data: Dictionary containing market data
        """
        # Extract features from market data
        features = self.feature_extractor(market_data)
        
        # Discretize state for Q-table lookup
        state_key = self.model["state_discretizer"](features)
        
        # Select strategy using the model
        strategy, params = self.model["strategy_selector"](state_key)
        
        # Update current strategy and parameters
        self.current_strategy = strategy
        self.strategy_params = params
        
        # Save state and action for learning
        self.state_history.append(state_key)
        self.action_history.append(strategy)
    
    def execute_slice(self, market_data: Dict) -> Dict:
        """
        Execute a slice of the order using the selected strategy.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Execution details as a dictionary
        """
        if not self.is_running or self.remaining_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Update market data
        self.update_market_data(market_data)
        
        # Select strategy if needed
        self.select_strategy(market_data)
        
        # Get current price
        current_price = market_data.get('price', 0)
        
        # Check price limit if specified
        if self.price_limit:
            if (self.side == 'buy' and current_price > self.price_limit) or \
               (self.side == 'sell' and current_price < self.price_limit):
                return {"executed": False, "quantity": 0, "price": 0}
        
        # Calculate target quantity based on selected strategy
        if self.current_strategy == ExecutionStrategy.TWAP:
            # Time-weighted execution
            total_time = (self.end_time - self.start_time).total_seconds()
            elapsed_time = (market_data['time'] - self.start_time).total_seconds()
            time_fraction = elapsed_time / total_time if total_time > 0 else 0
            expected_executed = self.total_quantity * time_fraction
            target_quantity = max(0, expected_executed - self.executed_quantity)
            
        elif self.current_strategy == ExecutionStrategy.VWAP:
            # Volume-weighted execution
            volume = market_data.get('volume', 0)
            participation_rate = self.strategy_params.get('max_participation_rate', 0.1)
            target_quantity = volume * participation_rate
            
        elif self.current_strategy == ExecutionStrategy.POV:
            # Percentage of volume
            volume = market_data.get('volume', 0)
            participation_rate = self.strategy_params.get('target_participation_rate', 0.1)
            target_quantity = volume * participation_rate
            
        elif self.current_strategy == ExecutionStrategy.ADAPTIVE:
            # Adaptive execution based on market conditions
            # Calculate time remaining and urgency factor
            total_time = (self.end_time - self.start_time).total_seconds()
            remaining_time = (self.end_time - market_data['time']).total_seconds()
            time_factor = max(0.1, min(10.0, total_time / max(1, remaining_time)))
            
            # Calculate volatility factor
            if len(self.price_history) >= 5:
                returns = np.diff(self.price_history[-5:]) / self.price_history[-5:-1]
                volatility = np.std(returns)
                vol_threshold = self.strategy_params.get('volatility_threshold', 0.02)
                vol_factor = 0.5 if volatility > vol_threshold else 1.5
            else:
                vol_factor = 1.0
                
            # Base rate proportional to remaining quantity
            base_rate = self.remaining_quantity / max(5, self.total_quantity / 10)
            target_quantity = base_rate * time_factor * vol_factor
            
        elif self.current_strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
            # Implementation shortfall with urgency parameter
            urgency = self.strategy_params.get('urgency', 0.5)
            risk_aversion = self.strategy_params.get('risk_aversion', 1.0)
            
            # Higher urgency means faster execution
            total_time = (self.end_time - self.start_time).total_seconds()
            elapsed_time = (market_data['time'] - self.start_time).total_seconds()
            time_fraction = elapsed_time / total_time if total_time > 0 else 0
            
            # For high urgency, front-load execution
            if urgency > 0.5:
                expected_executed = self.total_quantity * (2 * time_fraction * urgency)
            else:
                expected_executed = self.total_quantity * time_fraction
                
            target_quantity = max(0, expected_executed - self.executed_quantity)
            
        else:  # CUSTOM
            # Custom strategy (placeholder)
            target_quantity = self.remaining_quantity * 0.1
        
        # Ensure we don't exceed remaining quantity
        target_quantity = min(target_quantity, self.remaining_quantity)
        
        # Check if we should execute
        if target_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
        
        # Execute the slice
        self.executed_quantity += target_quantity
        self.remaining_quantity -= target_quantity
        
        execution_record = {
            "time": market_data['time'],
            "quantity": target_quantity,
            "price": current_price,
            "side": self.side,
            "executed": True,
            "strategy": self.current_strategy.name,
            "strategy_params": self.strategy_params
        }
        
        self.execution_history.append(execution_record)
        
        # Calculate reward and store it
        if len(self.state_history) > 0:
            current_features = self.feature_extractor(market_data)
            reward = self.reward_function(
                # Previous state
                self.feature_extractor({**market_data, 'current_time': market_data['time'] - timedelta(seconds=10)}),
                # Action
                self.current_strategy,
                # Current state
                current_features,
                # Execution data
                execution_record
            )
            self.reward_history.append(reward)
            
            # Update strategy performance
            self.strategy_performance[self.current_strategy.name].append(reward)
        
        return execution_record
    
    def get_execution_summary(self) -> Dict:
        """
        Get summary of execution performance.
        
        Returns:
            Dictionary with execution summary
        """
        if not self.execution_history:
            return {
                "status": "not_started" if not self.is_running else "running",
                "executed_quantity": 0,
                "remaining_quantity": self.total_quantity,
                "vwap_achieved": None,
                "completion_percentage": 0.0,
                "strategies_used": {}
            }
            
        # Calculate VWAP achieved
        vwap_numerator = sum(record["price"] * record["quantity"] for record in self.execution_history)
        vwap_denominator = sum(record["quantity"] for record in self.execution_history)
        vwap_achieved = vwap_numerator / vwap_denominator if vwap_denominator > 0 else None
        
        # Count strategy usage
        strategy_usage = {}
        for record in self.execution_history:
            strategy = record["strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
        # Calculate average reward per strategy
        strategy_rewards = {}
        for strategy, rewards in self.strategy_performance.items():
            if rewards:
                strategy_rewards[strategy] = sum(rewards) / len(rewards)
                
        completion_percentage = (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0
        
        # Calculate learning progress
        learning_progress = {}
        if hasattr(self.model, "q_table"):
            # Count the number of state-action pairs that have been updated
            total_pairs = 0
            updated_pairs = 0
            for state, actions in self.model["q_table"].items():
                for action, value in actions.items():
                    total_pairs += 1
                    if value != 0:
                        updated_pairs += 1
                        
            learning_progress["exploration_coverage"] = updated_pairs / total_pairs if total_pairs > 0 else 0
            learning_progress["state_count"] = len(self.model["q_table"])
        
        return {
            "status": "completed" if self.remaining_quantity <= 0 else "running",
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "vwap_achieved": vwap_achieved,
            "completion_percentage": completion_percentage,
            "strategies_used": strategy_usage,
            "strategy_performance": strategy_rewards,
            "current_strategy": self.current_strategy.name,
            "learning_progress": learning_progress
        } 