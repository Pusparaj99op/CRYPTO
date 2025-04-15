import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import warnings
import logging

logger = logging.getLogger(__name__)

class FlowToxicity:
    """
    A class for calculating order flow toxicity measures in financial markets.
    
    Order flow toxicity refers to the presence of adverse selection in markets,
    where informed traders exploit their information advantage over liquidity providers.
    
    This class implements various metrics to quantify flow toxicity including:
    - Volume Order Imbalance (VPIN)
    - Order Flow Imbalance (OFI)
    - Probability of Informed Trading (PIN)
    - Amihud's Lambda
    - Momentum indicators
    """
    
    def __init__(self, trade_data: Optional[pd.DataFrame] = None, 
                 order_data: Optional[pd.DataFrame] = None):
        """
        Initialize the FlowToxicity analyzer.
        
        Parameters:
        -----------
        trade_data : pd.DataFrame, optional
            DataFrame containing trade data with columns:
            - timestamp: time of trade
            - price: execution price
            - volume: execution volume
            - buy_sell: indicator for buy (1) or sell (-1)
            
        order_data : pd.DataFrame, optional
            DataFrame containing order book data with columns:
            - timestamp: time of order
            - type: 'limit', 'market', 'cancel'
            - side: 'buy' or 'sell'
            - price: order price
            - quantity: order quantity
        """
        self.trade_data = trade_data
        self.order_data = order_data
        self.metrics = {}
    
    def set_trade_data(self, trade_data: pd.DataFrame) -> None:
        """Set or update the trade data."""
        self.trade_data = trade_data
    
    def set_order_data(self, order_data: pd.DataFrame) -> None:
        """Set or update the order book data."""
        self.order_data = order_data
    
    def calculate_vpin(self, num_buckets: int = 50, bucket_size: Optional[float] = None) -> float:
        """
        Calculate Volume-synchronized Probability of Informed Trading (VPIN).
        
        VPIN is calculated by dividing the trading session into volume buckets,
        calculating order imbalance for each bucket, and taking the average.
        
        Parameters:
        -----------
        num_buckets : int
            Number of volume buckets to use
        bucket_size : float, optional
            Size of each volume bucket. If None, calculated as total volume / num_buckets
            
        Returns:
        --------
        float
            VPIN value between 0 and 1
        """
        if self.trade_data is None or len(self.trade_data) == 0:
            logger.warning("No trade data available for VPIN calculation")
            return np.nan
        
        # Check if required columns exist
        required_cols = ['volume']
        if not all(col in self.trade_data.columns for col in required_cols):
            logger.error(f"Missing required columns for VPIN: {required_cols}")
            return np.nan
        
        # If no explicit buy/sell indicator, try to infer from price and mid-price
        if 'buy_sell' not in self.trade_data.columns:
            if all(col in self.trade_data.columns for col in ['price', 'bid', 'ask']):
                # Calculate mid price and infer trade sign
                mid_price = (self.trade_data['bid'] + self.trade_data['ask']) / 2
                self.trade_data['buy_sell'] = np.where(
                    self.trade_data['price'] >= mid_price, 1, -1)
            else:
                logger.error("Cannot calculate VPIN without buy/sell information")
                return np.nan
        
        # Calculate total volume and bucket size
        total_volume = self.trade_data['volume'].sum()
        if bucket_size is None:
            bucket_size = total_volume / num_buckets
        
        # Create buckets based on cumulative volume
        self.trade_data['cum_vol'] = self.trade_data['volume'].cumsum()
        self.trade_data['bucket'] = (self.trade_data['cum_vol'] / bucket_size).astype(int)
        
        # Calculate buy and sell volume per bucket
        bucket_data = self.trade_data.groupby('bucket').apply(
            lambda x: pd.Series({
                'buy_volume': x.loc[x['buy_sell'] == 1, 'volume'].sum(),
                'sell_volume': x.loc[x['buy_sell'] == -1, 'volume'].sum(),
                'total_volume': x['volume'].sum()
            })
        )
        
        # Calculate order imbalance for each bucket
        bucket_data['imbalance'] = abs(bucket_data['buy_volume'] - bucket_data['sell_volume'])
        bucket_data['vpin_bucket'] = bucket_data['imbalance'] / bucket_data['total_volume']
        
        # Calculate VPIN as average of bucket VPINs
        vpin = bucket_data['vpin_bucket'].mean()
        
        self.metrics['vpin'] = vpin
        return vpin
    
    def calculate_order_flow_imbalance(self, window: int = 100) -> pd.Series:
        """
        Calculate Order Flow Imbalance (OFI) over a rolling window.
        
        OFI is the difference between buyer-initiated and seller-initiated volume
        over a specific time or event window.
        
        Parameters:
        -----------
        window : int
            Size of rolling window in number of trades
            
        Returns:
        --------
        pd.Series
            Series of OFI values
        """
        if self.trade_data is None or len(self.trade_data) == 0:
            logger.warning("No trade data available for OFI calculation")
            return pd.Series()
        
        # Check if required columns exist
        required_cols = ['volume']
        if not all(col in self.trade_data.columns for col in required_cols):
            logger.error(f"Missing required columns for OFI: {required_cols}")
            return pd.Series()
        
        # If no explicit buy/sell indicator, try to infer from price and mid-price
        if 'buy_sell' not in self.trade_data.columns:
            if all(col in self.trade_data.columns for col in ['price', 'bid', 'ask']):
                # Calculate mid price and infer trade sign
                mid_price = (self.trade_data['bid'] + self.trade_data['ask']) / 2
                self.trade_data['buy_sell'] = np.where(
                    self.trade_data['price'] >= mid_price, 1, -1)
            else:
                logger.error("Cannot calculate OFI without buy/sell information")
                return pd.Series()
        
        # Calculate signed volume
        self.trade_data['signed_volume'] = self.trade_data['volume'] * self.trade_data['buy_sell']
        
        # Calculate rolling sum of signed volume (OFI)
        ofi = self.trade_data['signed_volume'].rolling(window=window).sum()
        
        # Normalize by total volume in the window
        total_vol = self.trade_data['volume'].rolling(window=window).sum()
        normalized_ofi = ofi / total_vol
        
        self.metrics['ofi'] = normalized_ofi
        return normalized_ofi
    
    def calculate_pin_model(self, num_days: int = 60) -> Dict[str, float]:
        """
        Estimate the Probability of Informed Trading (PIN) model parameters.
        
        PIN is the probability that a given trade originates from an informed trader.
        It is based on a structural microstructure model.
        
        Parameters:
        -----------
        num_days : int
            Number of trading days to use for PIN estimation
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing PIN model parameters including PIN value
        """
        if self.trade_data is None or len(self.trade_data) == 0:
            logger.warning("No trade data available for PIN calculation")
            return {}
        
        # Check if required columns exist
        required_cols = ['timestamp', 'buy_sell', 'volume']
        if not all(col in self.trade_data.columns for col in required_cols):
            logger.error(f"Missing required columns for PIN: {required_cols}")
            return {}
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.trade_data['timestamp']):
            self.trade_data['timestamp'] = pd.to_datetime(self.trade_data['timestamp'])
        
        # Extract date from timestamp
        self.trade_data['date'] = self.trade_data['timestamp'].dt.date
        
        # Calculate buys and sells per day
        daily_stats = self.trade_data.groupby('date').apply(
            lambda x: pd.Series({
                'buys': x.loc[x['buy_sell'] == 1, 'volume'].sum(),
                'sells': x.loc[x['buy_sell'] == -1, 'volume'].sum()
            })
        )
        
        # Limit to requested number of days
        if len(daily_stats) > num_days:
            daily_stats = daily_stats.iloc[-num_days:]
        
        # Calculate means for buy and sell volumes
        mean_buys = daily_stats['buys'].mean()
        mean_sells = daily_stats['sells'].mean()
        
        # Calculate imbalance - this is an approximation of the full PIN model
        # which requires maximum likelihood estimation
        imbalance_days = abs(daily_stats['buys'] - daily_stats['sells']) > (mean_buys + mean_sells) / 2
        alpha = imbalance_days.mean()  # Probability of information events
        
        # Simple approximation of PIN parameters
        delta = 0.5  # Probability of bad news given information event (simplification)
        mu = abs(daily_stats['buys'] - daily_stats['sells']).mean() / 2  # Informed trade intensity
        epsilon_b = mean_buys - mu/2  # Uninformed buy rate
        epsilon_s = mean_sells - mu/2  # Uninformed sell rate
        
        # Calculate PIN
        pin = (alpha * mu) / (alpha * mu + epsilon_b + epsilon_s)
        
        pin_params = {
            'pin': pin,
            'alpha': alpha,
            'delta': delta,
            'mu': mu,
            'epsilon_b': epsilon_b,
            'epsilon_s': epsilon_s
        }
        
        self.metrics.update(pin_params)
        return pin_params
    
    def calculate_amihud_lambda(self, window: int = 20) -> pd.Series:
        """
        Calculate Amihud's Lambda (price impact coefficient) over a rolling window.
        
        Lambda measures how much prices move in response to order flow.
        
        Parameters:
        -----------
        window : int
            Size of rolling window in number of trades
            
        Returns:
        --------
        pd.Series
            Series of Lambda values
        """
        if self.trade_data is None or len(self.trade_data) < window:
            logger.warning("Insufficient trade data for Amihud's Lambda calculation")
            return pd.Series()
        
        # Check if required columns exist
        required_cols = ['price', 'volume']
        if not all(col in self.trade_data.columns for col in required_cols):
            logger.error(f"Missing required columns for Lambda: {required_cols}")
            return pd.Series()
        
        # Calculate returns
        self.trade_data['return'] = self.trade_data['price'].pct_change()
        
        # Calculate absolute returns
        self.trade_data['abs_return'] = self.trade_data['return'].abs()
        
        # Calculate Lambda as absolute return divided by volume (rolling window)
        lambda_values = pd.Series(index=self.trade_data.index)
        
        for i in range(window, len(self.trade_data)):
            window_data = self.trade_data.iloc[i-window:i]
            abs_ret_sum = window_data['abs_return'].sum()
            volume_sum = window_data['volume'].sum()
            
            if volume_sum > 0:
                lambda_values.iloc[i] = abs_ret_sum / volume_sum
        
        self.metrics['amihud_lambda'] = lambda_values
        return lambda_values
    
    def calculate_flow_toxicity_index(self) -> float:
        """
        Calculate a composite Flow Toxicity Index based on multiple metrics.
        
        Returns:
        --------
        float
            Flow Toxicity Index value between 0 and 1
        """
        # Ensure we have the necessary metrics
        if 'vpin' not in self.metrics:
            self.calculate_vpin()
        
        # Calculate OFI if not already done
        if 'ofi' not in self.metrics:
            self.calculate_order_flow_imbalance()
        
        # Get latest OFI value
        latest_ofi = self.metrics['ofi'].iloc[-1] if isinstance(self.metrics.get('ofi'), pd.Series) else np.nan
        
        # Calculate absolute OFI
        abs_ofi = abs(latest_ofi) if not np.isnan(latest_ofi) else np.nan
        
        # If we have PIN, use it, otherwise calculate it
        if 'pin' not in self.metrics:
            pin_params = self.calculate_pin_model()
            pin = pin_params.get('pin', np.nan)
        else:
            pin = self.metrics['pin']
        
        # Average the metrics we have
        available_metrics = []
        
        if not np.isnan(self.metrics.get('vpin', np.nan)):
            available_metrics.append(self.metrics['vpin'])
        
        if not np.isnan(abs_ofi):
            available_metrics.append(abs_ofi)
        
        if not np.isnan(pin):
            available_metrics.append(pin)
        
        if not available_metrics:
            logger.warning("No metrics available to calculate Flow Toxicity Index")
            return np.nan
        
        # Calculate the index as an average of available metrics
        toxicity_index = sum(available_metrics) / len(available_metrics)
        
        self.metrics['flow_toxicity_index'] = toxicity_index
        return toxicity_index
    
    def calculate_urgency_ratio(self, window: int = 100) -> pd.Series:
        """
        Calculate the ratio of market orders to limit orders as a measure of trading urgency.
        
        High urgency can indicate informed trading and flow toxicity.
        
        Parameters:
        -----------
        window : int
            Size of rolling window in number of orders
            
        Returns:
        --------
        pd.Series
            Series of urgency ratio values
        """
        if self.order_data is None or len(self.order_data) < window:
            logger.warning("Insufficient order data for urgency ratio calculation")
            return pd.Series()
        
        # Check if required columns exist
        required_cols = ['type']
        if not all(col in self.order_data.columns for col in required_cols):
            logger.error(f"Missing required columns for urgency ratio: {required_cols}")
            return pd.Series()
        
        # Count market and limit orders in rolling windows
        market_orders = (self.order_data['type'] == 'market').rolling(window=window).sum()
        limit_orders = (self.order_data['type'] == 'limit').rolling(window=window).sum()
        
        # Calculate ratio with safeguard against division by zero
        urgency_ratio = market_orders / limit_orders.replace(0, np.nan)
        
        self.metrics['urgency_ratio'] = urgency_ratio
        return urgency_ratio
    
    def calculate_order_cancellation_ratio(self, window: int = 100) -> pd.Series:
        """
        Calculate the ratio of canceled orders to new limit orders.
        
        High cancellation rates can indicate toxic flow or predatory trading strategies.
        
        Parameters:
        -----------
        window : int
            Size of rolling window in number of orders
            
        Returns:
        --------
        pd.Series
            Series of cancellation ratio values
        """
        if self.order_data is None or len(self.order_data) < window:
            logger.warning("Insufficient order data for cancellation ratio calculation")
            return pd.Series()
        
        # Check if required columns exist
        required_cols = ['type']
        if not all(col in self.order_data.columns for col in required_cols):
            logger.error(f"Missing required columns for cancellation ratio: {required_cols}")
            return pd.Series()
        
        # Count cancellations and limit orders in rolling windows
        cancellations = (self.order_data['type'] == 'cancel').rolling(window=window).sum()
        limit_orders = (self.order_data['type'] == 'limit').rolling(window=window).sum()
        
        # Calculate ratio with safeguard against division by zero
        cancellation_ratio = cancellations / limit_orders.replace(0, np.nan)
        
        self.metrics['cancellation_ratio'] = cancellation_ratio
        return cancellation_ratio
    
    def generate_toxicity_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive report on flow toxicity metrics.
        
        Returns:
        --------
        pd.DataFrame
            Report containing flow toxicity metrics with descriptions
        """
        # Calculate all metrics if not already done
        if 'vpin' not in self.metrics:
            self.calculate_vpin()
        
        if 'pin' not in self.metrics:
            self.calculate_pin_model()
        
        if 'flow_toxicity_index' not in self.metrics:
            self.calculate_flow_toxicity_index()
        
        # Create report dataframe
        report_data = []
        
        # Process scalar metrics
        scalar_metrics = ['vpin', 'pin', 'flow_toxicity_index']
        for metric in scalar_metrics:
            if metric in self.metrics and isinstance(self.metrics[metric], (int, float)) and not np.isnan(self.metrics[metric]):
                report_data.append({
                    'Metric': metric,
                    'Value': self.metrics[metric],
                    'Interpretation': self._interpret_metric(metric, self.metrics[metric]),
                    'Description': self._get_metric_description(metric)
                })
        
        # Process time series metrics by taking the latest value
        ts_metrics = ['ofi', 'amihud_lambda', 'urgency_ratio', 'cancellation_ratio']
        for metric in ts_metrics:
            if metric in self.metrics and isinstance(self.metrics[metric], pd.Series) and not self.metrics[metric].empty:
                latest_value = self.metrics[metric].iloc[-1]
                if not np.isnan(latest_value):
                    report_data.append({
                        'Metric': metric,
                        'Value': latest_value,
                        'Interpretation': self._interpret_metric(metric, latest_value),
                        'Description': self._get_metric_description(metric)
                    })
        
        # Create DataFrame
        report = pd.DataFrame(report_data)
        
        return report
    
    def _get_metric_description(self, metric: str) -> str:
        """Helper method to get description for each metric."""
        descriptions = {
            'vpin': 'Volume-synchronized Probability of Informed Trading',
            'ofi': 'Order Flow Imbalance - net directional pressure from orders',
            'pin': 'Probability of Informed Trading based on structural model',
            'amihud_lambda': 'Price impact coefficient - price move per unit of volume',
            'flow_toxicity_index': 'Composite index of order flow toxicity',
            'urgency_ratio': 'Ratio of market orders to limit orders',
            'cancellation_ratio': 'Ratio of canceled orders to new limit orders'
        }
        
        return descriptions.get(metric, 'No description available')
    
    def _interpret_metric(self, metric: str, value: float) -> str:
        """Helper method to interpret metric values."""
        
        # VPIN and PIN - probability values
        if metric in ['vpin', 'pin']:
            if value < 0.1:
                return "Low toxicity - few informed traders"
            elif value < 0.3:
                return "Moderate toxicity - some informed trading"
            else:
                return "High toxicity - significant informed trading"
        
        # Flow toxicity index
        elif metric == 'flow_toxicity_index':
            if value < 0.3:
                return "Low toxicity environment"
            elif value < 0.6:
                return "Moderate toxicity environment"
            else:
                return "High toxicity environment - caution advised"
        
        # OFI - can be positive or negative
        elif metric == 'ofi':
            abs_val = abs(value)
            direction = "buying" if value > 0 else "selling"
            if abs_val < 0.2:
                return f"Balanced flow - no strong {direction} pressure"
            elif abs_val < 0.5:
                return f"Moderate {direction} pressure"
            else:
                return f"Strong {direction} pressure - potential price movement"
        
        # Amihud's Lambda
        elif metric == 'amihud_lambda':
            # This is relative, so interpretation requires context
            return "Higher values indicate greater price impact per unit of volume"
        
        # Urgency ratio
        elif metric == 'urgency_ratio':
            if value < 0.5:
                return "Patient trading - mostly limit orders"
            elif value < 2:
                return "Balanced trading approach"
            else:
                return "Urgent trading - high proportion of market orders"
        
        # Cancellation ratio
        elif metric == 'cancellation_ratio':
            if value < 0.3:
                return "Low cancellation activity - stable order book"
            elif value < 1:
                return "Moderate cancellation activity"
            else:
                return "High cancellation activity - potential manipulative behavior"
        
        return "No interpretation available"

# Example usage
if __name__ == "__main__":
    # Create sample trade data
    np.random.seed(42)
    n_trades = 1000
    
    # Generate timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_trades, freq='1min')
    
    # Generate price path with some trend and volatility
    price = 100 + np.cumsum(np.random.normal(0, 0.1, n_trades))
    
    # Generate volume with log-normal distribution
    volume = np.random.lognormal(3, 1, n_trades)
    
    # Generate buy/sell indicators with some imbalance
    buy_sell = np.random.choice([1, -1], size=n_trades, p=[0.52, 0.48])
    
    # Create trade data DataFrame
    trade_data = pd.DataFrame({
        'timestamp': timestamps,
        'price': price,
        'volume': volume,
        'buy_sell': buy_sell
    })
    
    # Add bid and ask for spread calculations
    spread = np.random.uniform(0.05, 0.15, n_trades)
    trade_data['bid'] = trade_data['price'] - spread/2
    trade_data['ask'] = trade_data['price'] + spread/2
    
    # Create order data
    n_orders = 5000
    order_timestamps = pd.date_range(start='2023-01-01', periods=n_orders, freq='15s')
    
    # Generate order types with some distribution
    order_types = np.random.choice(['limit', 'market', 'cancel'], size=n_orders, p=[0.6, 0.2, 0.2])
    
    # Generate order sides
    order_sides = np.random.choice(['buy', 'sell'], size=n_orders, p=[0.52, 0.48])
    
    # Generate order prices around current price with some noise
    price_idx = np.minimum(np.searchsorted(timestamps, order_timestamps) - 1, len(price) - 1)
    price_idx = np.maximum(price_idx, 0)  # Ensure non-negative indices
    order_prices = price[price_idx] + np.random.normal(0, 0.2, n_orders)
    
    # Generate order quantities
    order_quantities = np.random.lognormal(2, 1, n_orders)
    
    # Create order data DataFrame
    order_data = pd.DataFrame({
        'timestamp': order_timestamps,
        'type': order_types,
        'side': order_sides,
        'price': order_prices,
        'quantity': order_quantities
    })
    
    # Initialize flow toxicity analyzer
    flow_analyzer = FlowToxicity(trade_data, order_data)
    
    # Calculate VPIN
    vpin = flow_analyzer.calculate_vpin()
    print(f"VPIN: {vpin:.4f}")
    
    # Calculate OFI
    ofi = flow_analyzer.calculate_order_flow_imbalance()
    print(f"Latest OFI: {ofi.iloc[-1]:.4f}")
    
    # Calculate PIN model
    pin_params = flow_analyzer.calculate_pin_model(num_days=20)
    print(f"PIN: {pin_params['pin']:.4f}")
    
    # Calculate flow toxicity index
    toxicity_index = flow_analyzer.calculate_flow_toxicity_index()
    print(f"Flow Toxicity Index: {toxicity_index:.4f}")
    
    # Generate toxicity report
    report = flow_analyzer.generate_toxicity_report()
    print("\nFlow Toxicity Report:")
    print(report) 