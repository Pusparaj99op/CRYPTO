import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class EffectiveSpread:
    """
    A class for calculating and analyzing effective spreads in financial markets.
    
    The effective spread is a measure of transaction costs that accounts for price improvement
    and represents the actual cost of trading relative to the mid-price.
    
    This class provides methods to calculate various spread metrics including:
    - Effective spread
    - Realized spread
    - Price impact
    - Implementation shortfall
    """
    
    def __init__(self, trade_data: Optional[pd.DataFrame] = None, 
                quote_data: Optional[pd.DataFrame] = None):
        """
        Initialize the EffectiveSpread analyzer.
        
        Parameters:
        -----------
        trade_data : pd.DataFrame, optional
            DataFrame containing trade execution data with columns:
            - timestamp: execution time
            - price: execution price
            - volume: execution volume
            - side: 'buy' or 'sell' (or 1/-1)
            
        quote_data : pd.DataFrame, optional
            DataFrame containing quote data with columns:
            - timestamp: quote time
            - bid: bid price
            - ask: ask price
            - bid_size: size at best bid
            - ask_size: size at best ask
        """
        self.trade_data = trade_data
        self.quote_data = quote_data
        self.results = {}
    
    def set_trade_data(self, trade_data: pd.DataFrame) -> None:
        """Set or update the trade data."""
        self.trade_data = trade_data
    
    def set_quote_data(self, quote_data: pd.DataFrame) -> None:
        """Set or update the quote data."""
        self.quote_data = quote_data
    
    def calculate_effective_spread(self, match_trades_to_quotes: bool = True) -> pd.DataFrame:
        """
        Calculate effective spread for each trade.
        
        The effective spread is calculated as 2 * |Price - Mid-price| * sign(side),
        where side is 1 for buys and -1 for sells.
        
        Parameters:
        -----------
        match_trades_to_quotes : bool
            If True, matches each trade to the most recent quote
            If False, assumes trade_data already contains bid/ask columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with effective spread calculations
        """
        if self.trade_data is None or len(self.trade_data) == 0:
            logger.error("No trade data available for effective spread calculation")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        trades = self.trade_data.copy()
        
        # Ensure trade data has required columns
        required_cols = ['price', 'timestamp']
        if not all(col in trades.columns for col in required_cols):
            logger.error(f"Trade data missing required columns: {required_cols}")
            return pd.DataFrame()
        
        # Ensure side information is available and consistent format
        if 'side' not in trades.columns and 'buy_sell' not in trades.columns:
            logger.error("Trade data must contain 'side' or 'buy_sell' column")
            return pd.DataFrame()
        
        # Standardize side column
        if 'buy_sell' in trades.columns and 'side' not in trades.columns:
            trades['side'] = trades['buy_sell']
        
        # Convert string sides to numeric if needed
        if isinstance(trades['side'].iloc[0], str):
            trades['side'] = trades['side'].map({'buy': 1, 'sell': -1})
        
        if match_trades_to_quotes and self.quote_data is not None:
            # Ensure quote data has required columns
            quote_cols = ['timestamp', 'bid', 'ask']
            if not all(col in self.quote_data.columns for col in quote_cols):
                logger.error(f"Quote data missing required columns: {quote_cols}")
                return pd.DataFrame()
            
            # Convert timestamps to datetime if they're not already
            for df in [trades, self.quote_data]:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort both dataframes by timestamp
            quotes = self.quote_data.sort_values('timestamp').copy()
            trades = trades.sort_values('timestamp')
            
            # Match each trade to most recent quote
            # Create a helper column for merging
            quotes['quote_idx'] = range(len(quotes))
            
            # For each trade, find the index of the most recent quote
            trade_quote_idx = []
            for trade_time in trades['timestamp']:
                # Find quotes that are before or at the trade time
                valid_quotes = quotes[quotes['timestamp'] <= trade_time]
                if len(valid_quotes) > 0:
                    # Get the most recent quote
                    trade_quote_idx.append(valid_quotes['quote_idx'].iloc[-1])
                else:
                    # No valid quote, use NaN
                    trade_quote_idx.append(np.nan)
            
            trades['quote_idx'] = trade_quote_idx
            
            # Filter out trades without a matching quote
            trades = trades.dropna(subset=['quote_idx'])
            
            # Convert quote_idx to int for matching
            trades['quote_idx'] = trades['quote_idx'].astype(int)
            
            # Get bid/ask for each trade
            trades['bid'] = trades['quote_idx'].map(quotes.set_index('quote_idx')['bid'])
            trades['ask'] = trades['quote_idx'].map(quotes.set_index('quote_idx')['ask'])
        
        # Calculate mid-price
        if 'bid' in trades.columns and 'ask' in trades.columns:
            trades['mid_price'] = (trades['bid'] + trades['ask']) / 2
        else:
            logger.error("Cannot calculate mid-price: missing 'bid' and 'ask' columns")
            return pd.DataFrame()
        
        # Calculate effective spread
        trades['price_diff'] = trades['price'] - trades['mid_price']
        trades['effective_spread_bps'] = 2 * trades['price_diff'] * trades['side'] / trades['mid_price'] * 10000
        trades['effective_spread'] = 2 * trades['price_diff'] * trades['side']
        
        # Calculate effective spread statistics
        spread_stats = {
            'mean_effective_spread_bps': trades['effective_spread_bps'].mean(),
            'median_effective_spread_bps': trades['effective_spread_bps'].median(),
            'std_effective_spread_bps': trades['effective_spread_bps'].std(),
            'min_effective_spread_bps': trades['effective_spread_bps'].min(),
            'max_effective_spread_bps': trades['effective_spread_bps'].max(),
        }
        
        self.results.update(spread_stats)
        self.trade_data_with_spread = trades
        
        return trades
    
    def calculate_realized_spread(self, time_horizon: str = '5min') -> pd.DataFrame:
        """
        Calculate realized spread and price impact for each trade.
        
        Realized spread measures the revenue to liquidity providers after accounting
        for adverse price movements. Price impact measures the information content of trades.
        
        Parameters:
        -----------
        time_horizon : str
            Time interval for measuring price impact (e.g., '5min', '1h')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with realized spread and price impact calculations
        """
        if not hasattr(self, 'trade_data_with_spread') or len(self.trade_data_with_spread) == 0:
            logger.warning("No processed trade data found. Running effective spread calculation first.")
            self.calculate_effective_spread()
        
        if not hasattr(self, 'trade_data_with_spread') or len(self.trade_data_with_spread) == 0:
            logger.error("Cannot calculate realized spread without effective spread data")
            return pd.DataFrame()
        
        trades = self.trade_data_with_spread.copy()
        
        # Ensure the timestamp column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(trades['timestamp']):
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        
        # Sort by timestamp
        trades = trades.sort_values('timestamp')
        
        # Calculate future mid-price for each trade
        trades['future_timestamp'] = trades['timestamp'] + pd.to_timedelta(time_horizon)
        
        # For each trade, find the mid-price after time_horizon
        future_prices = []
        
        for idx, row in trades.iterrows():
            # Find trades after the future timestamp
            future_trades = trades[trades['timestamp'] >= row['future_timestamp']]
            
            if len(future_trades) > 0:
                # Use the mid-price of the first trade after time_horizon
                future_prices.append(future_trades.iloc[0]['mid_price'])
            else:
                # No future trade available, use NaN
                future_prices.append(np.nan)
        
        trades['future_mid_price'] = future_prices
        
        # Filter out trades without a future price
        trades = trades.dropna(subset=['future_mid_price'])
        
        # Calculate realized spread (in price units and basis points)
        trades['realized_spread'] = 2 * (trades['price'] - trades['future_mid_price']) * trades['side']
        trades['realized_spread_bps'] = trades['realized_spread'] / trades['mid_price'] * 10000
        
        # Calculate price impact (in price units and basis points)
        trades['price_impact'] = trades['effective_spread'] - trades['realized_spread']
        trades['price_impact_bps'] = trades['effective_spread_bps'] - trades['realized_spread_bps']
        
        # Calculate statistics
        spread_stats = {
            'mean_realized_spread_bps': trades['realized_spread_bps'].mean(),
            'median_realized_spread_bps': trades['realized_spread_bps'].median(),
            'std_realized_spread_bps': trades['realized_spread_bps'].std(),
            'mean_price_impact_bps': trades['price_impact_bps'].mean(),
            'median_price_impact_bps': trades['price_impact_bps'].median(),
            'std_price_impact_bps': trades['price_impact_bps'].std(),
        }
        
        self.results.update(spread_stats)
        self.trade_data_with_realized = trades
        
        return trades
    
    def calculate_implementation_shortfall(self, decision_prices: pd.Series = None) -> pd.DataFrame:
        """
        Calculate implementation shortfall for a set of trades.
        
        Implementation shortfall measures the difference between the decision price
        and the actual execution price, accounting for both explicit and implicit costs.
        
        Parameters:
        -----------
        decision_prices : pd.Series, optional
            Series of prices at decision time, indexed by a decision identifier
            that matches the 'decision_id' column in the trade data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with implementation shortfall calculations
        """
        if self.trade_data is None or len(self.trade_data) == 0:
            logger.error("No trade data available for implementation shortfall calculation")
            return pd.DataFrame()
        
        trades = self.trade_data.copy()
        
        # Check if we have decision prices directly
        if decision_prices is not None:
            # Ensure trades has a decision_id column
            if 'decision_id' not in trades.columns:
                logger.error("Trade data must contain 'decision_id' column to match with decision prices")
                return pd.DataFrame()
            
            # Map decision prices to trades
            trades['decision_price'] = trades['decision_id'].map(decision_prices)
        
        # Alternative approach: use first trade in each order as the decision price
        elif 'order_id' in trades.columns and 'decision_price' not in trades.columns:
            # Group by order_id and get the first price for each order
            first_prices = trades.groupby('order_id')['price'].first()
            trades['decision_price'] = trades['order_id'].map(first_prices)
        
        # If still no decision price, we can't calculate implementation shortfall
        if 'decision_price' not in trades.columns:
            logger.error("Cannot calculate implementation shortfall without decision prices")
            return pd.DataFrame()
        
        # Ensure we have side information
        if 'side' not in trades.columns and 'buy_sell' not in trades.columns:
            logger.error("Trade data must contain 'side' or 'buy_sell' column")
            return pd.DataFrame()
        
        # Standardize side column
        if 'buy_sell' in trades.columns and 'side' not in trades.columns:
            trades['side'] = trades['buy_sell']
        
        # Convert string sides to numeric if needed
        if isinstance(trades['side'].iloc[0], str):
            trades['side'] = trades['side'].map({'buy': 1, 'sell': -1})
        
        # Calculate implementation shortfall
        trades['implementation_shortfall'] = (trades['price'] - trades['decision_price']) * trades['side']
        
        # Calculate percentage shortfall
        trades['implementation_shortfall_bps'] = trades['implementation_shortfall'] / trades['decision_price'] * 10000
        
        # Calculate volume-weighted shortfall per order if order_id exists
        if 'order_id' in trades.columns and 'volume' in trades.columns:
            # Calculate volume-weighted implementation shortfall per order
            order_shortfall = trades.groupby('order_id').apply(
                lambda x: np.sum(x['implementation_shortfall'] * x['volume']) / np.sum(x['volume'])
            )
            order_shortfall_bps = trades.groupby('order_id').apply(
                lambda x: np.sum(x['implementation_shortfall_bps'] * x['volume']) / np.sum(x['volume'])
            )
            
            # Create a summary dataframe for orders
            order_summary = pd.DataFrame({
                'order_shortfall': order_shortfall,
                'order_shortfall_bps': order_shortfall_bps
            })
            
            # Calculate overall volume-weighted average
            total_shortfall = np.sum(trades['implementation_shortfall'] * trades['volume']) / np.sum(trades['volume'])
            total_shortfall_bps = np.sum(trades['implementation_shortfall_bps'] * trades['volume']) / np.sum(trades['volume'])
            
            shortfall_stats = {
                'volume_weighted_shortfall': total_shortfall,
                'volume_weighted_shortfall_bps': total_shortfall_bps,
                'mean_order_shortfall_bps': order_shortfall_bps.mean(),
                'median_order_shortfall_bps': order_shortfall_bps.median(),
                'std_order_shortfall_bps': order_shortfall_bps.std()
            }
        else:
            # Just calculate simple statistics
            shortfall_stats = {
                'mean_implementation_shortfall_bps': trades['implementation_shortfall_bps'].mean(),
                'median_implementation_shortfall_bps': trades['implementation_shortfall_bps'].median(),
                'std_implementation_shortfall_bps': trades['implementation_shortfall_bps'].std()
            }
        
        self.results.update(shortfall_stats)
        self.trade_data_with_shortfall = trades
        
        return trades
    
    def calculate_quoted_spread(self) -> pd.DataFrame:
        """
        Calculate quoted spread statistics from quote data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with quoted spread statistics
        """
        if self.quote_data is None or len(self.quote_data) == 0:
            logger.error("No quote data available for quoted spread calculation")
            return pd.DataFrame()
        
        quotes = self.quote_data.copy()
        
        # Ensure quote data has required columns
        required_cols = ['bid', 'ask']
        if not all(col in quotes.columns for col in required_cols):
            logger.error(f"Quote data missing required columns: {required_cols}")
            return pd.DataFrame()
        
        # Calculate quoted spread
        quotes['quoted_spread'] = quotes['ask'] - quotes['bid']
        quotes['mid_price'] = (quotes['ask'] + quotes['bid']) / 2
        quotes['quoted_spread_bps'] = quotes['quoted_spread'] / quotes['mid_price'] * 10000
        
        # Calculate spread statistics
        spread_stats = {
            'mean_quoted_spread_bps': quotes['quoted_spread_bps'].mean(),
            'median_quoted_spread_bps': quotes['quoted_spread_bps'].median(),
            'std_quoted_spread_bps': quotes['quoted_spread_bps'].std(),
            'min_quoted_spread_bps': quotes['quoted_spread_bps'].min(),
            'max_quoted_spread_bps': quotes['quoted_spread_bps'].max(),
        }
        
        self.results.update(spread_stats)
        self.quote_data_with_spread = quotes
        
        return quotes
    
    def calculate_spread_capture(self) -> float:
        """
        Calculate spread capture ratio - the proportion of the quoted spread
        that is captured as effective spread.
        
        Returns:
        --------
        float
            Spread capture ratio
        """
        if not hasattr(self, 'trade_data_with_spread') or len(self.trade_data_with_spread) == 0:
            logger.warning("No processed trade data found. Running effective spread calculation first.")
            self.calculate_effective_spread()
        
        if not hasattr(self, 'quote_data_with_spread') or len(self.quote_data_with_spread) == 0:
            logger.warning("No processed quote data found. Running quoted spread calculation first.")
            self.calculate_quoted_spread()
        
        if not hasattr(self, 'trade_data_with_spread') or not hasattr(self, 'quote_data_with_spread'):
            logger.error("Cannot calculate spread capture without both effective and quoted spreads")
            return np.nan
        
        # Get mean spreads
        mean_effective_spread_bps = self.results.get('mean_effective_spread_bps')
        mean_quoted_spread_bps = self.results.get('mean_quoted_spread_bps')
        
        if mean_effective_spread_bps is None or mean_quoted_spread_bps is None:
            logger.error("Cannot calculate spread capture without both effective and quoted spreads")
            return np.nan
        
        # Calculate spread capture ratio
        spread_capture = mean_effective_spread_bps / mean_quoted_spread_bps
        
        self.results['spread_capture_ratio'] = spread_capture
        return spread_capture
    
    def calculate_effective_spread_by_size(self, size_bins: int = 5) -> pd.DataFrame:
        """
        Calculate how effective spread varies with trade size.
        
        Parameters:
        -----------
        size_bins : int
            Number of bins to divide trade sizes into
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with effective spread by size bin
        """
        if not hasattr(self, 'trade_data_with_spread') or len(self.trade_data_with_spread) == 0:
            logger.warning("No processed trade data found. Running effective spread calculation first.")
            self.calculate_effective_spread()
        
        if not hasattr(self, 'trade_data_with_spread') or len(self.trade_data_with_spread) == 0:
            logger.error("Cannot calculate spread by size without effective spread data")
            return pd.DataFrame()
        
        trades = self.trade_data_with_spread.copy()
        
        # Ensure we have volume information
        if 'volume' not in trades.columns:
            logger.error("Trade data must contain 'volume' column for size analysis")
            return pd.DataFrame()
        
        # Create size bins
        trades['size_quantile'] = pd.qcut(trades['volume'], size_bins, labels=False)
        
        # Group by size bin and calculate spread statistics
        spread_by_size = trades.groupby('size_quantile').agg({
            'volume': ['min', 'max', 'mean', 'count'],
            'effective_spread_bps': ['mean', 'median', 'std']
        })
        
        # Flatten the column hierarchy
        spread_by_size.columns = [f'{col[0]}_{col[1]}' for col in spread_by_size.columns]
        
        return spread_by_size
    
    def calculate_intraday_spread_pattern(self, interval: str = '30min') -> pd.DataFrame:
        """
        Calculate intraday pattern of spreads.
        
        Parameters:
        -----------
        interval : str
            Time interval for aggregation (e.g., '30min', '1h')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with spread statistics by time of day
        """
        # Start with effective spread
        if hasattr(self, 'trade_data_with_spread') and len(self.trade_data_with_spread) > 0:
            trades = self.trade_data_with_spread.copy()
            
            # Ensure we have a timestamp column as datetime
            if 'timestamp' not in trades.columns:
                logger.error("Trade data must contain 'timestamp' column for intraday analysis")
                return pd.DataFrame()
            
            if not pd.api.types.is_datetime64_any_dtype(trades['timestamp']):
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            
            # Extract time of day
            trades['time_of_day'] = trades['timestamp'].dt.floor(interval)
            
            # Group by time of day and calculate spread statistics
            effective_by_time = trades.groupby('time_of_day').agg({
                'effective_spread_bps': ['mean', 'median', 'std', 'count']
            })
            
            # Flatten the column hierarchy
            effective_by_time.columns = [f'effective_{col[1]}' for col in effective_by_time.columns]
        else:
            effective_by_time = pd.DataFrame()
        
        # Then do the same for quoted spread
        if hasattr(self, 'quote_data_with_spread') and len(self.quote_data_with_spread) > 0:
            quotes = self.quote_data_with_spread.copy()
            
            # Ensure we have a timestamp column as datetime
            if 'timestamp' not in quotes.columns:
                logger.warning("Quote data missing 'timestamp' column for intraday analysis")
            else:
                if not pd.api.types.is_datetime64_any_dtype(quotes['timestamp']):
                    quotes['timestamp'] = pd.to_datetime(quotes['timestamp'])
                
                # Extract time of day
                quotes['time_of_day'] = quotes['timestamp'].dt.floor(interval)
                
                # Group by time of day and calculate spread statistics
                quoted_by_time = quotes.groupby('time_of_day').agg({
                    'quoted_spread_bps': ['mean', 'median', 'std', 'count']
                })
                
                # Flatten the column hierarchy
                quoted_by_time.columns = [f'quoted_{col[1]}' for col in quoted_by_time.columns]
            
                # Combine effective and quoted spread statistics
                if not effective_by_time.empty:
                    spread_by_time = pd.merge(
                        effective_by_time, quoted_by_time, 
                        left_index=True, right_index=True,
                        how='outer'
                    )
                else:
                    spread_by_time = quoted_by_time
        else:
            if effective_by_time.empty:
                logger.error("No spread data available for intraday analysis")
                return pd.DataFrame()
            spread_by_time = effective_by_time
        
        # Convert the time_of_day index to hour:minute format for easier reading
        spread_by_time['hour'] = spread_by_time.index.hour
        spread_by_time['minute'] = spread_by_time.index.minute
        
        return spread_by_time
    
    def get_spread_report(self) -> Dict:
        """
        Generate a comprehensive report of all spread metrics.
        
        Returns:
        --------
        Dict
            Dictionary containing all spread statistics
        """
        # Ensure we've calculated the basic metrics
        if not hasattr(self, 'trade_data_with_spread'):
            self.calculate_effective_spread()
        
        if not hasattr(self, 'quote_data_with_spread') and self.quote_data is not None:
            self.calculate_quoted_spread()
        
        # Compile report
        spread_report = {
            'effective_spread_metrics': {
                k: v for k, v in self.results.items() if 'effective' in k
            },
            'quoted_spread_metrics': {
                k: v for k, v in self.results.items() if 'quoted' in k
            },
            'realized_spread_metrics': {
                k: v for k, v in self.results.items() if 'realized' in k
            },
            'price_impact_metrics': {
                k: v for k, v in self.results.items() if 'impact' in k
            },
            'implementation_shortfall_metrics': {
                k: v for k, v in self.results.items() if 'shortfall' in k
            },
            'spread_capture_ratio': self.results.get('spread_capture_ratio', np.nan)
        }
        
        return spread_report
    
    def plot_spread_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of key spread metrics.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the generated plots
        """
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Comparing quoted vs effective spread
        if (hasattr(self, 'trade_data_with_spread') and 
            hasattr(self, 'quote_data_with_spread')):
            
            plt.subplot(2, 2, 1)
            plt.title('Quoted vs Effective Spread')
            
            # Get the data
            quoted = self.results.get('mean_quoted_spread_bps', np.nan)
            effective = self.results.get('mean_effective_spread_bps', np.nan)
            
            if not np.isnan(quoted) and not np.isnan(effective):
                plt.bar(['Quoted Spread', 'Effective Spread'], [quoted, effective])
                plt.ylabel('Spread (bps)')
                
                # Add spread capture ratio
                spread_capture = self.results.get('spread_capture_ratio', np.nan)
                if not np.isnan(spread_capture):
                    plt.text(1, effective * 1.1, f'Capture: {spread_capture:.2f}', 
                             ha='center', va='bottom')
        
        # Plot 2: Effective spread by trade size
        if hasattr(self, 'trade_data_with_spread'):
            spread_by_size = self.calculate_effective_spread_by_size()
            
            if not spread_by_size.empty:
                plt.subplot(2, 2, 2)
                plt.title('Effective Spread by Trade Size')
                
                # Get mean spread by size quantile
                size_means = spread_by_size['effective_spread_bps_mean']
                size_labels = [f"Q{i+1}" for i in range(len(size_means))]
                
                plt.bar(size_labels, size_means)
                plt.ylabel('Effective Spread (bps)')
                plt.xlabel('Size Quantile')
        
        # Plot 3: Intraday pattern
        if hasattr(self, 'trade_data_with_spread'):
            intraday = self.calculate_intraday_spread_pattern()
            
            if not intraday.empty and 'effective_mean' in intraday.columns:
                plt.subplot(2, 2, 3)
                plt.title('Intraday Spread Pattern')
                
                # Format times for x-axis
                time_labels = [f"{h:02d}:{m:02d}" for h, m in 
                              zip(intraday['hour'], intraday['minute'])]
                
                plt.plot(time_labels, intraday['effective_mean'], marker='o', label='Effective')
                
                if 'quoted_mean' in intraday.columns:
                    plt.plot(time_labels, intraday['quoted_mean'], marker='x', label='Quoted')
                
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Spread (bps)')
                plt.xlabel('Time of Day')
                plt.legend()
                plt.tight_layout()
        
        # Plot 4: Effective vs Realized Spread
        if hasattr(self, 'trade_data_with_realized'):
            plt.subplot(2, 2, 4)
            plt.title('Effective vs Realized Spread vs Price Impact')
            
            # Get the data
            effective = self.results.get('mean_effective_spread_bps', np.nan)
            realized = self.results.get('mean_realized_spread_bps', np.nan)
            impact = self.results.get('mean_price_impact_bps', np.nan)
            
            if not np.isnan(effective) and not np.isnan(realized) and not np.isnan(impact):
                plt.bar(['Effective', 'Realized', 'Price Impact'], 
                        [effective, realized, impact])
                plt.ylabel('Spread/Impact (bps)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def resample_spread_time_series(self, interval: str = '1h') -> pd.DataFrame:
        """
        Resample spread time series data to a lower frequency.
        
        Parameters:
        -----------
        interval : str
            Resampling interval (e.g., '1h', '1d')
            
        Returns:
        --------
        pd.DataFrame
            Resampled spread time series
        """
        result_data = {}
        
        # Resample effective spread data if available
        if hasattr(self, 'trade_data_with_spread') and len(self.trade_data_with_spread) > 0:
            trades = self.trade_data_with_spread.copy()
            
            # Ensure timestamp is a datetime
            if not pd.api.types.is_datetime64_any_dtype(trades['timestamp']):
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            
            # Set timestamp as index
            trades = trades.set_index('timestamp')
            
            # Resample
            effective_resampled = trades['effective_spread_bps'].resample(interval).agg(['mean', 'median', 'std', 'count'])
            effective_resampled.columns = [f'effective_{col}' for col in effective_resampled.columns]
            
            result_data['effective'] = effective_resampled
        
        # Resample quoted spread data if available
        if hasattr(self, 'quote_data_with_spread') and len(self.quote_data_with_spread) > 0:
            quotes = self.quote_data_with_spread.copy()
            
            # Ensure timestamp is a datetime
            if not pd.api.types.is_datetime64_any_dtype(quotes['timestamp']):
                quotes['timestamp'] = pd.to_datetime(quotes['timestamp'])
            
            # Set timestamp as index
            quotes = quotes.set_index('timestamp')
            
            # Resample
            quoted_resampled = quotes['quoted_spread_bps'].resample(interval).agg(['mean', 'median', 'std', 'count'])
            quoted_resampled.columns = [f'quoted_{col}' for col in quoted_resampled.columns]
            
            result_data['quoted'] = quoted_resampled
        
        # Combine results
        if result_data:
            resampled = pd.concat(result_data.values(), axis=1)
            return resampled
        else:
            logger.warning("No spread data available for resampling")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_quotes = 1000
    n_trades = 200
    
    # Generate timestamps
    quote_times = pd.date_range(start='2023-01-01 09:30:00', periods=n_quotes, freq='1min')
    
    # Generate prices with a random walk
    mid_price = 100 + np.cumsum(np.random.normal(0, 0.02, n_quotes))
    
    # Generate varying spreads (tighter during high liquidity periods)
    hour_of_day = quote_times.hour
    intraday_factor = 1 + 0.5 * np.sin(np.pi * (hour_of_day - 9) / 8)  # Higher spreads at open/close
    spreads = np.random.uniform(0.05, 0.15, n_quotes) * intraday_factor
    
    # Calculate bid and ask
    bid = mid_price - spreads / 2
    ask = mid_price + spreads / 2
    
    # Generate size at best bid/ask
    bid_size = np.random.lognormal(4, 0.5, n_quotes)
    ask_size = np.random.lognormal(4, 0.5, n_quotes)
    
    # Create quote DataFrame
    quotes = pd.DataFrame({
        'timestamp': quote_times,
        'bid': bid,
        'ask': ask,
        'bid_size': bid_size,
        'ask_size': ask_size,
        'mid_price': mid_price
    })
    
    # Generate trade data
    # Random subset of quote times for trades
    trade_indices = np.sort(np.random.choice(n_quotes, n_trades, replace=False))
    trade_times = quote_times[trade_indices]
    
    # Get quote data for these trades
    trade_quotes = quotes.iloc[trade_indices]
    
    # Generate trade prices around mid-price with some price improvement
    price_improvement = np.random.uniform(0, 1, n_trades) * (spreads[trade_indices] / 2) * 0.3
    
    # Generate trade sides
    sides = np.random.choice([1, -1], n_trades)
    
    # Calculate trade prices with price improvement
    trade_prices = np.where(
        sides == 1,
        trade_quotes['mid_price'].values + (spreads[trade_indices] / 2) - price_improvement,  # buys
        trade_quotes['mid_price'].values - (spreads[trade_indices] / 2) + price_improvement   # sells
    )
    
    # Generate trade volumes
    trade_volumes = np.random.lognormal(4, 1, n_trades)
    
    # Create trade DataFrame
    trades = pd.DataFrame({
        'timestamp': trade_times,
        'price': trade_prices,
        'volume': trade_volumes,
        'side': sides,
        'bid': trade_quotes['bid'].values,
        'ask': trade_quotes['ask'].values
    })
    
    # Initialize EffectiveSpread analyzer
    spread_analyzer = EffectiveSpread(trades, quotes)
    
    # Calculate effective spread
    spread_results = spread_analyzer.calculate_effective_spread(match_trades_to_quotes=False)
    print(f"Mean effective spread: {spread_analyzer.results['mean_effective_spread_bps']:.2f} bps")
    
    # Calculate quoted spread
    quoted_results = spread_analyzer.calculate_quoted_spread()
    print(f"Mean quoted spread: {spread_analyzer.results['mean_quoted_spread_bps']:.2f} bps")
    
    # Calculate spread capture
    capture = spread_analyzer.calculate_spread_capture()
    print(f"Spread capture ratio: {capture:.2f}")
    
    # Calculate realized spread and price impact
    realized_results = spread_analyzer.calculate_realized_spread(time_horizon='5min')
    print(f"Mean realized spread: {spread_analyzer.results['mean_realized_spread_bps']:.2f} bps")
    print(f"Mean price impact: {spread_analyzer.results['mean_price_impact_bps']:.2f} bps")
    
    # Calculate spread by size
    spread_by_size = spread_analyzer.calculate_effective_spread_by_size()
    print("\nEffective spread by trade size:")
    print(spread_by_size)
    
    # Generate spread report
    report = spread_analyzer.get_spread_report()
    print("\nSpread report:")
    for category, metrics in report.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
                
    # Visualize spread metrics
    spread_analyzer.plot_spread_metrics() 