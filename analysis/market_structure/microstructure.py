import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class MarketMicrostructure:
    """
    A class for calculating and analyzing market microstructure metrics in financial markets.
    
    Market microstructure refers to the study of the process and outcomes of exchanging assets
    under a specific set of rules. This class implements various metrics to quantify market
    microstructure characteristics.
    
    Key metrics include:
    - Bid-ask spreads (quoted, effective, realized)
    - Market depth and liquidity measures
    - Order book imbalance
    - Price impact measures
    - Volatility and price efficiency measures
    - Intraday patterns
    - Market quality indicators
    """
    
    def __init__(self, quote_data: Optional[pd.DataFrame] = None, 
                 trade_data: Optional[pd.DataFrame] = None,
                 order_book_data: Optional[pd.DataFrame] = None):
        """
        Initialize the MarketMicrostructure analyzer.
        
        Parameters:
        -----------
        quote_data : pd.DataFrame, optional
            DataFrame containing quote data with columns:
            - timestamp: time of quote
            - bid: best bid price
            - ask: best ask price
            - bid_size: size/volume at best bid
            - ask_size: size/volume at best ask
            
        trade_data : pd.DataFrame, optional
            DataFrame containing trade data with columns:
            - timestamp: time of trade
            - price: execution price
            - volume: execution volume
            - side: buy or sell (optional)
            
        order_book_data : pd.DataFrame, optional
            DataFrame containing order book snapshots with columns:
            - timestamp: time of snapshot
            - level: price level
            - bid_price: price at bid level
            - bid_size: size at bid level
            - ask_price: price at ask level
            - ask_size: size at ask level
        """
        self.quote_data = quote_data
        self.trade_data = trade_data
        self.order_book_data = order_book_data
        self.metrics = {}
        
    def set_quote_data(self, quote_data: pd.DataFrame) -> None:
        """Set or update the quote data."""
        self.quote_data = quote_data
        
    def set_trade_data(self, trade_data: pd.DataFrame) -> None:
        """Set or update the trade data."""
        self.trade_data = trade_data
        
    def set_order_book_data(self, order_book_data: pd.DataFrame) -> None:
        """Set or update the order book data."""
        self.order_book_data = order_book_data
        
    def calculate_quoted_spread(self) -> pd.DataFrame:
        """
        Calculate quoted spread metrics from quote data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with timestamp, absolute spread, relative spread, and log spread
        """
        if self.quote_data is None or len(self.quote_data) == 0:
            logger.warning("No quote data available for quoted spread calculation")
            return pd.DataFrame()
        
        # Check if required columns exist
        required_cols = ['timestamp', 'bid', 'ask']
        if not all(col in self.quote_data.columns for col in required_cols):
            logger.error(f"Missing required columns for quoted spread: {required_cols}")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original data
        result = self.quote_data[required_cols].copy()
        
        # Calculate mid price
        result['mid_price'] = (result['ask'] + result['bid']) / 2
        
        # Calculate absolute spread
        result['absolute_spread'] = result['ask'] - result['bid']
        
        # Calculate relative spread (as percentage of mid price)
        result['relative_spread'] = result['absolute_spread'] / result['mid_price'] * 100
        
        # Calculate log spread
        result['log_spread'] = np.log(result['ask'] / result['bid'])
        
        # Store in metrics dictionary
        self.metrics['quoted_spread'] = result
        
        return result
    
    def calculate_depth_measures(self, levels: int = 5) -> pd.DataFrame:
        """
        Calculate market depth measures from order book data.
        
        Parameters:
        -----------
        levels : int
            Number of price levels to include in depth calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with timestamp and depth measures
        """
        if self.order_book_data is None or len(self.order_book_data) == 0:
            logger.warning("No order book data available for depth calculation")
            return pd.DataFrame()
        
        # Check if we have level information in the order book data
        if 'level' not in self.order_book_data.columns:
            logger.error("Order book data must contain 'level' column for depth calculation")
            return pd.DataFrame()
        
        # Filter to specified number of levels
        filtered_data = self.order_book_data[self.order_book_data['level'] <= levels]
        
        # Group by timestamp
        grouped = filtered_data.groupby('timestamp')
        
        # Aggregate results
        result = pd.DataFrame()
        result['timestamp'] = filtered_data['timestamp'].unique()
        
        # Total depth
        result['bid_depth'] = grouped['bid_size'].sum()
        result['ask_depth'] = grouped['ask_size'].sum()
        result['total_depth'] = result['bid_depth'] + result['ask_depth']
        
        # Depth imbalance
        result['depth_imbalance'] = (result['bid_depth'] - result['ask_depth']) / result['total_depth']
        
        # Price impact estimates
        # For a market buy order of X size, how much would price move?
        # This is a simple approximation
        test_sizes = [1, 5, 10, 20, 50]  # Example sizes, adjust based on the market
        
        for size in test_sizes:
            result[f'price_impact_buy_{size}'] = self._calculate_price_impact(filtered_data, size, 'buy')
            result[f'price_impact_sell_{size}'] = self._calculate_price_impact(filtered_data, size, 'sell')
        
        # Store in metrics dictionary
        self.metrics['market_depth'] = result
        
        return result
    
    def _calculate_price_impact(self, order_book: pd.DataFrame, size: float, side: str) -> pd.Series:
        """
        Helper method to calculate price impact for a given size and side.
        
        Parameters:
        -----------
        order_book : pd.DataFrame
            Order book data
        size : float
            Order size to calculate impact for
        side : str
            'buy' or 'sell'
            
        Returns:
        --------
        pd.Series
            Series of price impact values indexed by timestamp
        """
        impacts = {}
        
        # Group by timestamp and sort by level
        for timestamp, group in order_book.groupby('timestamp'):
            group = group.sort_values('level')
            
            if side == 'buy':
                # For buys, we walk the ask side
                remaining_size = size
                initial_price = group.iloc[0]['ask_price']
                current_price = initial_price
                
                for _, row in group.iterrows():
                    if remaining_size <= 0:
                        break
                    
                    if remaining_size <= row['ask_size']:
                        # This level can satisfy the remaining size
                        current_price = row['ask_price']
                        remaining_size = 0
                    else:
                        # Take all from this level and continue
                        remaining_size -= row['ask_size']
                        current_price = row['ask_price']
                
                impacts[timestamp] = (current_price / initial_price) - 1
                
            else:  # sell
                # For sells, we walk the bid side
                remaining_size = size
                initial_price = group.iloc[0]['bid_price']
                current_price = initial_price
                
                for _, row in group.iterrows():
                    if remaining_size <= 0:
                        break
                    
                    if remaining_size <= row['bid_size']:
                        # This level can satisfy the remaining size
                        current_price = row['bid_price']
                        remaining_size = 0
                    else:
                        # Take all from this level and continue
                        remaining_size -= row['bid_size']
                        current_price = row['bid_price']
                
                impacts[timestamp] = 1 - (current_price / initial_price)
        
        return pd.Series(impacts)
    
    def calculate_order_book_imbalance(self) -> pd.DataFrame:
        """
        Calculate order book imbalance metrics.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with timestamp and imbalance metrics
        """
        if self.order_book_data is None or len(self.order_book_data) == 0:
            logger.warning("No order book data available for imbalance calculation")
            return pd.DataFrame()
        
        # Group by timestamp
        grouped = self.order_book_data.groupby(['timestamp', 'level'])
        
        # Initialize results
        result_data = []
        
        # Process each timestamp
        for (timestamp, _), group in grouped:
            bid_volume = group['bid_size'].sum()
            ask_volume = group['ask_size'].sum()
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                volume_imbalance = (bid_volume - ask_volume) / total_volume
            else:
                volume_imbalance = 0
            
            # Weighted average price calculations
            vwap_bid = (group['bid_price'] * group['bid_size']).sum() / (group['bid_size'].sum() or 1)
            vwap_ask = (group['ask_price'] * group['ask_size']).sum() / (group['ask_size'].sum() or 1)
            
            # Calculate order book imbalance
            result_data.append({
                'timestamp': timestamp,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'volume_imbalance': volume_imbalance,
                'vwap_bid': vwap_bid,
                'vwap_ask': vwap_ask,
                'vwap_mid': (vwap_bid + vwap_ask) / 2
            })
        
        result = pd.DataFrame(result_data)
        
        # Store in metrics dictionary
        self.metrics['order_book_imbalance'] = result
        
        return result
    
    def calculate_trade_based_metrics(self, window: int = 100) -> pd.DataFrame:
        """
        Calculate trade-based microstructure metrics.
        
        Parameters:
        -----------
        window : int
            Rolling window size for calculations
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trade-based metrics
        """
        if self.trade_data is None or len(self.trade_data) == 0:
            logger.warning("No trade data available for trade metrics calculation")
            return pd.DataFrame()
        
        # Check if required columns exist
        required_cols = ['timestamp', 'price', 'volume']
        if not all(col in self.trade_data.columns for col in required_cols):
            logger.error(f"Missing required columns for trade metrics: {required_cols}")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original data
        result = self.trade_data[required_cols].copy()
        
        # Calculate returns
        result['return'] = result['price'].pct_change()
        
        # Calculate statistics over rolling windows
        result['volatility'] = result['return'].rolling(window=window).std() * np.sqrt(window)
        result['volume_weighted_price'] = result['price'] * result['volume']
        result['cumulative_volume'] = result['volume'].rolling(window=window).sum()
        result['vwap'] = result['volume_weighted_price'].rolling(window=window).sum() / result['cumulative_volume']
        
        # Calculate trade intensity
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result['time_delta'] = result['timestamp'].diff().dt.total_seconds()
        result['trade_intensity'] = 1 / result['time_delta']
        
        # Identify large trades
        mean_volume = result['volume'].mean()
        result['large_trade'] = result['volume'] > (2 * mean_volume)
        
        # Calculate price impact of trades
        result['price_impact'] = result['return'].abs() / result['volume']
        
        # Store in metrics dictionary
        self.metrics['trade_metrics'] = result
        
        return result
    
    def calculate_intraday_patterns(self, interval: str = '1H') -> pd.DataFrame:
        """
        Calculate intraday patterns in market microstructure.
        
        Parameters:
        -----------
        interval : str
            Time interval for resampling (default: 1 hour)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with intraday patterns
        """
        metrics_dfs = []
        
        # Process quote data if available
        if self.quote_data is not None and len(self.quote_data) > 0:
            # Make sure we have calculated quoted spread
            if 'quoted_spread' not in self.metrics:
                self.calculate_quoted_spread()
            
            quoted_spread = self.metrics['quoted_spread'].copy()
            quoted_spread['timestamp'] = pd.to_datetime(quoted_spread['timestamp'])
            quoted_spread.set_index('timestamp', inplace=True)
            
            # Extract time components
            quoted_spread['hour'] = quoted_spread.index.hour
            quoted_spread['minute'] = quoted_spread.index.minute
            
            # Resample by specified interval
            spread_pattern = quoted_spread.groupby(pd.Grouper(freq=interval)).agg({
                'absolute_spread': 'mean',
                'relative_spread': 'mean',
                'log_spread': 'mean',
                'hour': 'first',
                'minute': 'first'
            })
            
            # Add type indicator
            spread_pattern['metric_type'] = 'quoted_spread'
            metrics_dfs.append(spread_pattern)
        
        # Process trade data if available
        if self.trade_data is not None and len(self.trade_data) > 0:
            trade_data = self.trade_data.copy()
            trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
            trade_data.set_index('timestamp', inplace=True)
            
            # Extract time components
            trade_data['hour'] = trade_data.index.hour
            trade_data['minute'] = trade_data.index.minute
            
            # Resample by specified interval
            trade_pattern = trade_data.groupby(pd.Grouper(freq=interval)).agg({
                'volume': 'sum',
                'price': ['mean', 'std'],
                'hour': 'first',
                'minute': 'first'
            })
            
            # Flatten multi-index columns
            trade_pattern.columns = ['_'.join(col).strip() for col in trade_pattern.columns.values]
            
            # Add type indicator
            trade_pattern['metric_type'] = 'trade_metrics'
            metrics_dfs.append(trade_pattern)
        
        # Combine results if we have any
        if metrics_dfs:
            # Reset index to make timestamp a column again
            metrics_dfs = [df.reset_index() for df in metrics_dfs]
            
            # Create time of day field for sorting
            for df in metrics_dfs:
                df['time_of_day'] = df['hour'] + df['minute']/60
            
            # Store in metrics dictionary
            self.metrics['intraday_patterns'] = metrics_dfs
            
            return pd.concat(metrics_dfs, axis=0)
        else:
            logger.warning("No data available for intraday pattern calculation")
            return pd.DataFrame()
    
    def calculate_kyle_lambda(self, window: int = 20) -> pd.Series:
        """
        Calculate Kyle's Lambda, a measure of market impact.
        
        Kyle's Lambda relates price changes to order flow imbalance.
        
        Parameters:
        -----------
        window : int
            Size of rolling window for calculation
            
        Returns:
        --------
        pd.Series
            Series of Kyle's Lambda values
        """
        if self.trade_data is None or len(self.trade_data) < window:
            logger.warning("Insufficient trade data for Kyle's Lambda calculation")
            return pd.Series()
        
        # Check if required columns exist
        required_cols = ['price', 'volume']
        if not all(col in self.trade_data.columns for col in required_cols):
            logger.error(f"Missing required columns for Kyle's Lambda: {required_cols}")
            return pd.Series()
        
        # Determine trade sign if not provided
        if 'side' in self.trade_data.columns:
            # Convert side to numeric: buy = 1, sell = -1
            sign = self.trade_data['side'].map({'buy': 1, 'sell': -1})
        elif all(col in self.trade_data.columns for col in ['bid', 'ask']):
            # Infer from price relative to mid price
            mid = (self.trade_data['bid'] + self.trade_data['ask']) / 2
            sign = np.sign(self.trade_data['price'] - mid)
        else:
            # Use price changes as a crude approximation
            sign = np.sign(self.trade_data['price'].diff())
        
        # Calculate signed volume
        signed_volume = self.trade_data['volume'] * sign
        
        # Calculate price changes
        price_changes = self.trade_data['price'].diff()
        
        # Calculate Kyle's Lambda using rolling regression
        lambda_values = pd.Series(index=self.trade_data.index)
        
        for i in range(window, len(self.trade_data)):
            window_price_changes = price_changes.iloc[i-window:i]
            window_signed_volume = signed_volume.iloc[i-window:i]
            
            # Skip if we have NaN values
            if window_price_changes.isna().any() or window_signed_volume.isna().any():
                continue
            
            # Simple linear regression: price_change = lambda * signed_volume
            slope, _, _, _, _ = stats.linregress(window_signed_volume, window_price_changes)
            lambda_values.iloc[i] = slope
        
        self.metrics['kyle_lambda'] = lambda_values
        return lambda_values
    
    def calculate_market_quality_index(self) -> float:
        """
        Calculate a composite market quality index.
        
        Returns:
        --------
        float
            Market quality index between 0 and 1
        """
        components = []
        
        # 1. Quoted spread component (lower is better)
        if 'quoted_spread' in self.metrics:
            spread_data = self.metrics['quoted_spread']
            rel_spread_mean = spread_data['relative_spread'].mean()
            # Normalize: 0% spread → 1, 2% spread → 0
            spread_score = max(0, 1 - rel_spread_mean / 2)
            components.append(spread_score)
        
        # 2. Market depth component (higher is better)
        if 'market_depth' in self.metrics:
            depth_data = self.metrics['market_depth']
            # Normalize using percentile approach
            depth_mean = depth_data['total_depth'].mean()
            depth_scores = []
            
            # If we have historical data to compare against
            if hasattr(self, 'historical_depth_percentiles'):
                percentile = np.searchsorted(self.historical_depth_percentiles, depth_mean) / 100
                depth_scores.append(percentile)
            else:
                # Just use a placeholder score based on depth imbalance
                imbalance = abs(depth_data['depth_imbalance'].mean())
                depth_scores.append(1 - imbalance)
            
            if depth_scores:
                components.append(np.mean(depth_scores))
        
        # 3. Volatility component (lower is better)
        if 'trade_metrics' in self.metrics:
            trade_data = self.metrics['trade_metrics']
            vol_mean = trade_data['volatility'].mean()
            # Normalize: 0% volatility → 1, 10% volatility → 0
            vol_score = max(0, 1 - vol_mean / 0.1)
            components.append(vol_score)
        
        # 4. Price impact component (lower is better)
        if 'kyle_lambda' in self.metrics:
            lambda_mean = self.metrics['kyle_lambda'].mean()
            # Normalize using a reference value
            lambda_score = max(0, 1 - lambda_mean / 0.001)
            components.append(lambda_score)
        
        # Calculate overall quality index
        if components:
            quality_index = np.mean(components)
            self.metrics['market_quality_index'] = quality_index
            return quality_index
        else:
            logger.warning("Insufficient data to calculate market quality index")
            return np.nan
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all available market microstructure metrics.
        
        Returns:
        --------
        Dict
            Dictionary of all calculated metrics
        """
        # Calculate basic metrics
        if self.quote_data is not None:
            self.calculate_quoted_spread()
        
        if self.order_book_data is not None:
            self.calculate_depth_measures()
            self.calculate_order_book_imbalance()
        
        if self.trade_data is not None:
            self.calculate_trade_based_metrics()
            self.calculate_kyle_lambda()
        
        # Calculate composite metrics
        if self.quote_data is not None or self.trade_data is not None:
            self.calculate_intraday_patterns()
        
        # Calculate market quality index
        self.calculate_market_quality_index()
        
        return self.metrics
    
    def generate_microstructure_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive report on market microstructure metrics.
        
        Returns:
        --------
        pd.DataFrame
            Report containing microstructure metrics with descriptions
        """
        # Calculate all metrics if not already done
        if not self.metrics:
            self.calculate_all_metrics()
        
        # Create report dataframe
        report_data = []
        
        # Process quoted spread metrics
        if 'quoted_spread' in self.metrics and not self.metrics['quoted_spread'].empty:
            spread_data = self.metrics['quoted_spread']
            report_data.append({
                'Metric': 'Average Absolute Spread',
                'Value': spread_data['absolute_spread'].mean(),
                'Description': 'Average difference between best ask and best bid prices'
            })
            report_data.append({
                'Metric': 'Average Relative Spread (%)',
                'Value': spread_data['relative_spread'].mean(),
                'Description': 'Average spread as percentage of mid price'
            })
        
        # Process market depth metrics
        if 'market_depth' in self.metrics and not self.metrics['market_depth'].empty:
            depth_data = self.metrics['market_depth']
            report_data.append({
                'Metric': 'Average Market Depth',
                'Value': depth_data['total_depth'].mean(),
                'Description': 'Average total order book depth (bid + ask)'
            })
            report_data.append({
                'Metric': 'Average Depth Imbalance',
                'Value': depth_data['depth_imbalance'].mean(),
                'Description': 'Average imbalance between bid and ask sides'
            })
            
            # Include price impact metrics if available
            for col in [c for c in depth_data.columns if 'price_impact' in c]:
                size = col.split('_')[-1]
                side = col.split('_')[-2]
                report_data.append({
                    'Metric': f'Price Impact {side.capitalize()} {size}',
                    'Value': depth_data[col].mean(),
                    'Description': f'Average price impact for {side} order of size {size}'
                })
        
        # Process trade metrics
        if 'trade_metrics' in self.metrics and not self.metrics['trade_metrics'].empty:
            trade_data = self.metrics['trade_metrics']
            report_data.append({
                'Metric': 'Average Trade Size',
                'Value': trade_data['volume'].mean(),
                'Description': 'Average size of individual trades'
            })
            report_data.append({
                'Metric': 'Trade Size Volatility',
                'Value': trade_data['volume'].std(),
                'Description': 'Standard deviation of trade sizes'
            })
            report_data.append({
                'Metric': 'Price Volatility',
                'Value': trade_data['volatility'].mean(),
                'Description': 'Average price volatility'
            })
            report_data.append({
                'Metric': 'Average Trade Intensity',
                'Value': trade_data['trade_intensity'].mean(),
                'Description': 'Average number of trades per second'
            })
        
        # Include Kyle's Lambda if available
        if 'kyle_lambda' in self.metrics and isinstance(self.metrics['kyle_lambda'], pd.Series):
            lambda_mean = self.metrics['kyle_lambda'].mean()
            if not np.isnan(lambda_mean):
                report_data.append({
                    'Metric': 'Kyle\'s Lambda',
                    'Value': lambda_mean,
                    'Description': 'Price impact coefficient - higher values indicate lower liquidity'
                })
        
        # Include market quality index if available
        if 'market_quality_index' in self.metrics:
            mqi = self.metrics['market_quality_index']
            if not np.isnan(mqi):
                report_data.append({
                    'Metric': 'Market Quality Index',
                    'Value': mqi,
                    'Description': 'Composite measure of market quality (0-1 scale)'
                })
        
        # Create DataFrame
        report = pd.DataFrame(report_data)
        
        return report
    
    def plot_microstructure_measures(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots for key microstructure measures.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot image
        """
        if not self.metrics:
            logger.warning("No metrics calculated yet. Run calculate_all_metrics() first.")
            return
        
        # Set up the figure
        plt.figure(figsize=(20, 15))
        
        # Keep track of subplot position
        plot_position = 1
        
        # Plot quoted spread if available
        if 'quoted_spread' in self.metrics and not self.metrics['quoted_spread'].empty:
            plt.subplot(3, 2, plot_position)
            spread_data = self.metrics['quoted_spread'].copy()
            if 'timestamp' in spread_data.columns:
                spread_data.set_index('timestamp', inplace=True)
            
            plt.plot(spread_data.index, spread_data['relative_spread'], label='Relative Spread (%)')
            plt.title('Quoted Spread Over Time')
            plt.ylabel('Relative Spread (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_position += 1
        
        # Plot market depth if available
        if 'market_depth' in self.metrics and not self.metrics['market_depth'].empty:
            plt.subplot(3, 2, plot_position)
            depth_data = self.metrics['market_depth'].copy()
            if 'timestamp' in depth_data.columns:
                depth_data.set_index('timestamp', inplace=True)
                
            plt.plot(depth_data.index, depth_data['bid_depth'], label='Bid Depth')
            plt.plot(depth_data.index, depth_data['ask_depth'], label='Ask Depth')
            plt.title('Market Depth Over Time')
            plt.ylabel('Depth')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_position += 1
        
        # Plot price volatility if available
        if 'trade_metrics' in self.metrics and not self.metrics['trade_metrics'].empty:
            plt.subplot(3, 2, plot_position)
            trade_data = self.metrics['trade_metrics'].copy()
            if 'timestamp' in trade_data.columns:
                trade_data.set_index('timestamp', inplace=True)
                
            plt.plot(trade_data.index, trade_data['volatility'], label='Price Volatility')
            plt.title('Price Volatility Over Time')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_position += 1
        
        # Plot Kyle's Lambda if available
        if 'kyle_lambda' in self.metrics and isinstance(self.metrics['kyle_lambda'], pd.Series):
            plt.subplot(3, 2, plot_position)
            lambda_series = self.metrics['kyle_lambda'].dropna()
            
            plt.plot(lambda_series.index, lambda_series, label='Kyle\'s Lambda')
            plt.title('Kyle\'s Lambda (Price Impact) Over Time')
            plt.ylabel('Lambda')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_position += 1
        
        # Plot intraday patterns if available
        if 'intraday_patterns' in self.metrics and self.metrics['intraday_patterns']:
            plt.subplot(3, 2, plot_position)
            for df in self.metrics['intraday_patterns']:
                if 'metric_type' in df.columns and 'time_of_day' in df.columns:
                    # Plot relative spread by time of day
                    if df['metric_type'].iloc[0] == 'quoted_spread' and 'relative_spread' in df.columns:
                        plt.plot(df['time_of_day'], df['relative_spread'], 
                                 marker='o', label='Relative Spread')
                    
                    # Plot price volatility by time of day
                    if df['metric_type'].iloc[0] == 'trade_metrics' and 'price_std' in df.columns:
                        plt.plot(df['time_of_day'], df['price_std'], 
                                 marker='s', label='Price Volatility')
            
            plt.title('Intraday Patterns')
            plt.xlabel('Hour of Day')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_position += 1
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_quotes = 1000
    n_trades = 800
    n_levels = 5
    
    # Generate timestamps
    quote_timestamps = pd.date_range(start='2023-01-01', periods=n_quotes, freq='1min')
    trade_timestamps = pd.date_range(start='2023-01-01', periods=n_trades, freq='75s')
    
    # Generate price path with some trend and volatility
    mid_price = 100 + np.cumsum(np.random.normal(0, 0.02, n_quotes))
    
    # Generate spreads that vary over time
    spreads = np.abs(np.random.normal(0.1, 0.02, n_quotes))
    
    # Generate quote data
    quote_data = pd.DataFrame({
        'timestamp': quote_timestamps,
        'bid': mid_price - spreads/2,
        'ask': mid_price + spreads/2,
        'bid_size': np.random.lognormal(3, 0.5, n_quotes),
        'ask_size': np.random.lognormal(3, 0.5, n_quotes)
    })
    
    # Generate trade data
    trade_prices = []
    trade_volumes = np.random.lognormal(2, 0.8, n_trades)
    trade_sides = np.random.choice(['buy', 'sell'], size=n_trades)
    
    for i in range(n_trades):
        # Find closest quote time
        quote_idx = np.abs(quote_timestamps - trade_timestamps[i]).argmin()
        bid = quote_data.iloc[quote_idx]['bid']
        ask = quote_data.iloc[quote_idx]['ask']
        mid = (bid + ask) / 2
        
        # Determine price based on side
        if trade_sides[i] == 'buy':
            # Buys tend to be at or above the ask
            price = ask * (1 + np.random.uniform(0, 0.001))
        else:
            # Sells tend to be at or below the bid
            price = bid * (1 - np.random.uniform(0, 0.001))
        
        trade_prices.append(price)
    
    trade_data = pd.DataFrame({
        'timestamp': trade_timestamps,
        'price': trade_prices,
        'volume': trade_volumes,
        'side': trade_sides
    })
    
    # Generate order book data
    order_book_data = []
    
    for t_idx, timestamp in enumerate(quote_timestamps):
        mid = mid_price[t_idx]
        spread = spreads[t_idx]
        
        for level in range(1, n_levels + 1):
            # Price gets worse as level increases
            bid_price = mid - spread/2 - (level-1) * spread
            ask_price = mid + spread/2 + (level-1) * spread
            
            # Size tends to increase with level
            bid_size = np.random.lognormal(3 + 0.2 * (level-1), 0.5)
            ask_size = np.random.lognormal(3 + 0.2 * (level-1), 0.5)
            
            order_book_data.append({
                'timestamp': timestamp,
                'level': level,
                'bid_price': bid_price,
                'bid_size': bid_size,
                'ask_price': ask_price,
                'ask_size': ask_size
            })
    
    order_book_df = pd.DataFrame(order_book_data)
    
    # Initialize and use the MarketMicrostructure class
    ms = MarketMicrostructure(quote_data, trade_data, order_book_df)
    
    # Calculate all metrics
    ms.calculate_all_metrics()
    
    # Generate report
    report = ms.generate_microstructure_report()
    print("\nMarket Microstructure Report:")
    print(report)
    
    # Create visualization
    ms.plot_microstructure_measures() 