"""
Synthetic data generation for cryptocurrency market data.
This module provides tools for generating realistic synthetic
cryptocurrency market data for testing, simulation, and ML training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from scipy import stats


class TimeSeriesGenerator:
    """Base class for time series data generation."""
    
    def __init__(self, start_price=10000.0, volatility=0.02, drift=0.0001, 
                 mean_reversion=None, seasonality=None, jumps=None):
        """Initialize time series generator with model parameters.
        
        Args:
            start_price: Starting price for the time series
            volatility: Daily volatility (standard deviation of returns)
            drift: Daily drift (mean of returns)
            mean_reversion: Dict with mean_reversion parameters or None
            seasonality: Dict with seasonality parameters or None
            jumps: Dict with jump parameters or None
        """
        self.start_price = start_price
        self.volatility = volatility
        self.drift = drift
        self.mean_reversion = mean_reversion
        self.seasonality = seasonality
        self.jumps = jumps
    
    def generate_price_series(self, n_points=1000, freq='1h', seed=None):
        """Generate a price time series based on model parameters.
        
        Args:
            n_points: Number of data points to generate
            freq: Frequency of data points ('1d', '1h', etc.)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with timestamp index and price data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create timestamp index
        if freq == '1d':
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=n_points)
            dates = [start_date + timedelta(days=i) for i in range(n_points)]
            annualization = 365.0
        elif freq == '1h':
            start_date = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=n_points)
            dates = [start_date + timedelta(hours=i) for i in range(n_points)]
            annualization = 365.0 * 24
        elif freq == '1min':
            start_date = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=n_points)
            dates = [start_date + timedelta(minutes=i) for i in range(n_points)]
            annualization = 365.0 * 24 * 60
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
        
        # Scale parameters to the given frequency
        scaled_vol = self.volatility / np.sqrt(annualization)
        scaled_drift = self.drift / annualization
        
        # Initialize price array
        prices = np.zeros(n_points)
        prices[0] = self.start_price
        
        # Generate log returns
        log_returns = np.random.normal(scaled_drift, scaled_vol, n_points-1)
        
        # Add mean reversion if specified
        if self.mean_reversion is not None:
            strength = self.mean_reversion.get('strength', 0.01)
            target = self.mean_reversion.get('target', self.start_price)
            log_target = np.log(target)
            
            for i in range(n_points-1):
                # Add mean reversion component: strength * (log_target - log(current_price))
                log_returns[i] += strength * (log_target - np.log(prices[i]))
        
        # Add seasonality if specified
        if self.seasonality is not None:
            period = self.seasonality.get('period', 24)  # Default to daily (24h) cycle
            amplitude = self.seasonality.get('amplitude', 0.01)
            phase = self.seasonality.get('phase', 0)
            
            for i in range(n_points-1):
                # Add sinusoidal seasonality component
                seasonal_component = amplitude * np.sin(2 * np.pi * (i + phase) / period)
                log_returns[i] += seasonal_component
        
        # Add jumps if specified
        if self.jumps is not None:
            probability = self.jumps.get('probability', 0.01)
            mean = self.jumps.get('mean', 0)
            std = self.jumps.get('std', 0.05)
            
            # Generate random jump occurrences
            jump_mask = np.random.random(n_points-1) < probability
            
            # Generate jump sizes from normal distribution
            jumps = np.random.normal(mean, std, n_points-1) * jump_mask
            
            # Add jumps to log returns
            log_returns += jumps
        
        # Convert log returns to prices
        for i in range(1, n_points):
            prices[i] = prices[i-1] * np.exp(log_returns[i-1])
        
        # Create DataFrame with timestamp index
        df = pd.DataFrame({'price': prices}, index=dates)
        
        return df
    
    def generate_ohlcv(self, n_candles=1000, freq='1h', intrabar_points=20, seed=None):
        """Generate OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            n_candles: Number of candlesticks to generate
            freq: Frequency of candles ('1d', '1h', etc.)
            intrabar_points: Number of price points to simulate within each candle
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with timestamp index and OHLCV columns
        """
        if seed is not None:
            np.random.seed(seed)
        
        # First generate closing prices at the candle frequency
        close_df = self.generate_price_series(n_candles, freq, seed)
        closes = close_df['price'].values
        
        # Initialize arrays for OHLCV data
        opens = np.zeros(n_candles)
        highs = np.zeros(n_candles)
        lows = np.zeros(n_candles)
        volumes = np.zeros(n_candles)
        
        # Set initial open price equal to first close
        opens[0] = closes[0]
        
        # Generate intrabar points for each candle to determine highs and lows
        for i in range(n_candles):
            # Set open price (previous close for all except first candle)
            if i > 0:
                opens[i] = closes[i-1]
            
            # Generate intrabar prices
            if i == 0:
                start_price = opens[i]
            else:
                start_price = closes[i-1]
            
            target_price = closes[i]
            
            # Generate random walk from open to close
            intrabar_drift = (np.log(target_price) - np.log(start_price)) / intrabar_points
            intrabar_vol = self.volatility / np.sqrt(252 * 24 * intrabar_points)  # Scale for intrabar
            
            intrabar_returns = np.random.normal(intrabar_drift, intrabar_vol, intrabar_points)
            intrabar_prices = np.zeros(intrabar_points + 1)
            intrabar_prices[0] = start_price
            
            for j in range(intrabar_points):
                intrabar_prices[j+1] = intrabar_prices[j] * np.exp(intrabar_returns[j])
            
            # Ensure the last intrabar price matches the target close
            intrabar_prices[-1] = target_price
            
            # Set high and low
            highs[i] = np.max(intrabar_prices)
            lows[i] = np.min(intrabar_prices)
            
            # Generate volume (log-normal distribution correlated with absolute returns)
            abs_return = abs(np.log(closes[i] / opens[i]))
            volume_base = np.random.lognormal(mean=10, sigma=1)  # Base volume
            volume_multiplier = 1 + 5 * abs_return  # Higher volume for larger price moves
            volumes[i] = volume_base * volume_multiplier
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=close_df.index)
        
        return df
    
    def plot_time_series(self, df, title=None):
        """Plot the generated time series.
        
        Args:
            df: DataFrame with time series data
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price/OHLC
        if 'open' in df.columns:
            # Plot candlestick-like chart
            up = df[df.close >= df.open]
            down = df[df.close < df.open]
            
            # Plot price line
            axes[0].plot(df.index, df.close, color='black', alpha=0.3, linewidth=1)
            
            # Plot candlestick wicks
            for idx, row in df.iterrows():
                axes[0].plot([idx, idx], [row.low, row.high], color='black', alpha=0.5, linewidth=1)
            
            # Plot candlestick bodies
            width = 0.6
            axes[0].bar(up.index, up.close - up.open, width, bottom=up.open, color='green', alpha=0.5)
            axes[0].bar(down.index, down.open - down.close, width, bottom=down.close, color='red', alpha=0.5)
            
            axes[0].set_ylabel('Price')
            
            # Plot volume
            axes[1].bar(up.index, up.volume, width, color='green', alpha=0.5)
            axes[1].bar(down.index, down.volume, width, color='red', alpha=0.5)
            axes[1].set_ylabel('Volume')
        else:
            # Simple line chart for price-only series
            axes[0].plot(df.index, df.price)
            axes[0].set_ylabel('Price')
            
            # Plot returns
            returns = df.price.pct_change()
            axes[1].plot(df.index, returns, color='blue', alpha=0.6)
            axes[1].set_ylabel('Returns')
        
        if title:
            fig.suptitle(title)
        
        fig.tight_layout()
        return fig


class MarketDataGenerator:
    """Generator for complete market datasets including price, indicators, and events."""
    
    def __init__(self, base_generator=None, indicators=None, market_events=None):
        """Initialize market data generator.
        
        Args:
            base_generator: TimeSeriesGenerator for the base price series
            indicators: List of indicator configurations to generate
            market_events: Configuration for generating market events
        """
        # Create default base generator if none provided
        if base_generator is None:
            self.base_generator = TimeSeriesGenerator()
        else:
            self.base_generator = base_generator
        
        self.indicators = indicators or []
        self.market_events = market_events or {}
    
    def generate_market_data(self, n_candles=1000, freq='1h', seed=None):
        """Generate a complete market dataset.
        
        Args:
            n_candles: Number of candlesticks to generate
            freq: Frequency of candles ('1d', '1h', etc.)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with timestamp index, OHLCV data, and indicators
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate base OHLCV data
        df = self.base_generator.generate_ohlcv(n_candles, freq, seed=seed)
        
        # Add technical indicators
        for indicator in self.indicators:
            indicator_type = indicator['type']
            
            if indicator_type == 'sma':
                # Simple Moving Average
                window = indicator.get('window', 20)
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            
            elif indicator_type == 'ema':
                # Exponential Moving Average
                window = indicator.get('window', 20)
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
            elif indicator_type == 'rsi':
                # Relative Strength Index
                window = indicator.get('window', 14)
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
                rs = gain / loss
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            elif indicator_type == 'bollinger':
                # Bollinger Bands
                window = indicator.get('window', 20)
                std_dev = indicator.get('std_dev', 2)
                
                df[f'bollinger_mid_{window}'] = df['close'].rolling(window=window).mean()
                df[f'bollinger_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'bollinger_upper_{window}'] = df[f'bollinger_mid_{window}'] + std_dev * df[f'bollinger_std_{window}']
                df[f'bollinger_lower_{window}'] = df[f'bollinger_mid_{window}'] - std_dev * df[f'bollinger_std_{window}']
            
            elif indicator_type == 'macd':
                # MACD (Moving Average Convergence Divergence)
                fast = indicator.get('fast', 12)
                slow = indicator.get('slow', 26)
                signal = indicator.get('signal', 9)
                
                df[f'ema_{fast}'] = df['close'].ewm(span=fast, adjust=False).mean()
                df[f'ema_{slow}'] = df['close'].ewm(span=slow, adjust=False).mean()
                df[f'macd'] = df[f'ema_{fast}'] - df[f'ema_{slow}']
                df[f'macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
                df[f'macd_histogram'] = df['macd'] - df[f'macd_signal']
        
        # Add market events
        if 'news_impact' in self.market_events:
            # Simulate news events with price impacts
            impact_config = self.market_events['news_impact']
            frequency = impact_config.get('frequency', 0.05)  # Probability of news event per candle
            mean_impact = impact_config.get('mean_impact', 0)
            impact_std = impact_config.get('impact_std', 0.02)
            
            # Generate random news events
            news_events = np.random.random(n_candles) < frequency
            
            # Generate impact sizes
            impacts = np.zeros(n_candles)
            impacts[news_events] = np.random.normal(mean_impact, impact_std, np.sum(news_events))
            
            # Apply impacts to price series with decay
            decay_factor = impact_config.get('decay_factor', 0.9)
            news_effect = np.zeros(n_candles)
            
            for i in range(n_candles):
                if i > 0:
                    news_effect[i] = news_effect[i-1] * decay_factor
                news_effect[i] += impacts[i]
            
            # Store news impact in dataframe
            df['news_impact'] = news_effect
            df['news_event'] = news_events.astype(int)
        
        # Add market regime information
        if 'market_regimes' in self.market_events:
            regime_config = self.market_events['market_regimes']
            num_regimes = regime_config.get('num_regimes', 3)
            mean_duration = regime_config.get('mean_duration', 30)  # in candles
            
            # Generate regime transitions
            regimes = np.zeros(n_candles, dtype=int)
            current_regime = 0
            next_transition = np.random.geometric(1/mean_duration)
            
            for i in range(n_candles):
                if i >= next_transition:
                    current_regime = (current_regime + 1) % num_regimes
                    next_transition = i + np.random.geometric(1/mean_duration)
                regimes[i] = current_regime
            
            # Store regime information
            df['market_regime'] = regimes
            
            # Adjust volatility based on regime
            if regime_config.get('affect_volatility', True):
                vol_multipliers = regime_config.get('vol_multipliers', [0.7, 1.0, 1.5])
                df['regime_volatility'] = [vol_multipliers[r] for r in regimes]
        
        return df
    
    def plot_market_data(self, df, include_indicators=True, save_path=None):
        """Plot the generated market data with indicators.
        
        Args:
            df: DataFrame with market data
            include_indicators: Whether to plot technical indicators
            save_path: Path to save the figure (if None, will display)
            
        Returns:
            Matplotlib figure
        """
        indicator_types = set([ind['type'] for ind in self.indicators])
        n_indicator_plots = len(indicator_types)
        
        if 'market_regime' in df.columns:
            n_indicator_plots += 1
        
        if 'news_impact' in df.columns:
            n_indicator_plots += 1
        
        # Create figure with subplots
        fig, axes = plt.subplots(2 + n_indicator_plots, 1, figsize=(15, 12), 
                                gridspec_kw={'height_ratios': [3, 1] + [1] * n_indicator_plots})
        
        # Plot OHLC
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        
        # Plot price line
        axes[0].plot(df.index, df.close, color='black', alpha=0.3, linewidth=1)
        
        # Plot candlestick wicks
        for idx, row in df.iterrows():
            axes[0].plot([idx, idx], [row.low, row.high], color='black', alpha=0.5, linewidth=1)
        
        # Plot candlestick bodies
        width = 0.6
        axes[0].bar(up.index, up.close - up.open, width, bottom=up.open, color='green', alpha=0.5)
        axes[0].bar(down.index, down.open - down.close, width, bottom=down.close, color='red', alpha=0.5)
        
        # Add moving averages to price chart if available
        if include_indicators:
            sma_cols = [col for col in df.columns if col.startswith('sma_')]
            ema_cols = [col for col in df.columns if col.startswith('ema_') and not col.startswith('ema_1')] # exclude ema_12 for MACD
            
            for col in sma_cols:
                axes[0].plot(df.index, df[col], label=col, alpha=0.7)
            
            for col in ema_cols:
                axes[0].plot(df.index, df[col], label=col, alpha=0.7)
            
            # Add Bollinger Bands if available
            bb_mid_cols = [col for col in df.columns if col.startswith('bollinger_mid_')]
            for col in bb_mid_cols:
                window = col.split('_')[-1]
                axes[0].plot(df.index, df[col], label=f'BB Mid {window}', color='blue', alpha=0.7)
                axes[0].plot(df.index, df[f'bollinger_upper_{window}'], label=f'BB Upper {window}', 
                           color='blue', linestyle='--', alpha=0.5)
                axes[0].plot(df.index, df[f'bollinger_lower_{window}'], label=f'BB Lower {window}', 
                           color='blue', linestyle='--', alpha=0.5)
            
            if len(sma_cols + ema_cols + bb_mid_cols) > 0:
                axes[0].legend(loc='upper left')
        
        axes[0].set_ylabel('Price')
        axes[0].set_title('Synthetic Market Data')
        
        # Plot volume
        axes[1].bar(up.index, up.volume, width, color='green', alpha=0.5)
        axes[1].bar(down.index, down.volume, width, color='red', alpha=0.5)
        axes[1].set_ylabel('Volume')
        
        # Add indicator plots
        plot_idx = 2
        
        # RSI
        if 'rsi' in indicator_types and include_indicators:
            rsi_cols = [col for col in df.columns if col.startswith('rsi_')]
            for col in rsi_cols:
                axes[plot_idx].plot(df.index, df[col], label=col)
                
            # Add overbought/oversold lines
            axes[plot_idx].axhline(y=70, color='r', linestyle='-', alpha=0.3)
            axes[plot_idx].axhline(y=30, color='g', linestyle='-', alpha=0.3)
            axes[plot_idx].set_ylim(0, 100)
            axes[plot_idx].legend(loc='upper left')
            axes[plot_idx].set_ylabel('RSI')
            plot_idx += 1
        
        # MACD
        if 'macd' in indicator_types and include_indicators and 'macd' in df.columns:
            axes[plot_idx].plot(df.index, df['macd'], label='MACD', color='blue')
            axes[plot_idx].plot(df.index, df['macd_signal'], label='Signal', color='red')
            
            # Plot histogram as bar chart
            above_signal = df['macd_histogram'] >= 0
            below_signal = df['macd_histogram'] < 0
            
            axes[plot_idx].bar(df.index[above_signal], df['macd_histogram'][above_signal], 
                             width, color='green', alpha=0.5)
            axes[plot_idx].bar(df.index[below_signal], df['macd_histogram'][below_signal], 
                             width, color='red', alpha=0.5)
            
            axes[plot_idx].legend(loc='upper left')
            axes[plot_idx].set_ylabel('MACD')
            plot_idx += 1
        
        # Market regime
        if 'market_regime' in df.columns:
            axes[plot_idx].plot(df.index, df['market_regime'], drawstyle='steps-post')
            axes[plot_idx].set_ylabel('Regime')
            
            # Add volatility if available
            if 'regime_volatility' in df.columns:
                ax2 = axes[plot_idx].twinx()
                ax2.plot(df.index, df['regime_volatility'], color='red', alpha=0.5)
                ax2.set_ylabel('Vol Mult', color='red')
            
            plot_idx += 1
        
        # News impact
        if 'news_impact' in df.columns:
            ax_news = axes[plot_idx]
            ax_news.plot(df.index, df['news_impact'], label='Impact', color='blue')
            
            # Mark actual news events
            if 'news_event' in df.columns:
                event_idx = df.index[df['news_event'] == 1]
                event_impact = df['news_impact'][df['news_event'] == 1]
                ax_news.scatter(event_idx, event_impact, color='red', marker='^', s=50, label='News Event')
            
            ax_news.legend(loc='upper left')
            ax_news.set_ylabel('News Impact')
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig


class OrderBookGenerator:
    """Generator for limit order book data."""
    
    def __init__(self, price_generator=None, spread_mean=0.01, levels=10, 
                 order_size_mean=1.0, order_size_std=0.5):
        """Initialize the order book generator.
        
        Args:
            price_generator: Generator for mid-price process
            spread_mean: Mean percentage bid-ask spread
            levels: Number of price levels to generate on each side
            order_size_mean: Mean order size (log-normal distribution)
            order_size_std: Standard deviation of order size (log-normal)
        """
        if price_generator is None:
            self.price_generator = TimeSeriesGenerator(volatility=0.01, drift=0.0)
        else:
            self.price_generator = price_generator
        
        self.spread_mean = spread_mean
        self.levels = levels
        self.order_size_mean = order_size_mean
        self.order_size_std = order_size_std
    
    def generate_order_book_snapshot(self, mid_price, spread=None, depth_decay=0.1):
        """Generate a single order book snapshot.
        
        Args:
            mid_price: Current mid price
            spread: Current bid-ask spread (if None, use self.spread_mean)
            depth_decay: Decay factor for liquidity as we move away from mid price
            
        Returns:
            Tuple of (bid_prices, bid_sizes, ask_prices, ask_sizes)
        """
        if spread is None:
            # Generate random spread around the mean
            spread = abs(np.random.normal(self.spread_mean, self.spread_mean / 4))
        
        # Calculate best bid and ask prices
        half_spread = mid_price * spread / 2
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Generate price levels
        bid_prices = np.zeros(self.levels)
        ask_prices = np.zeros(self.levels)
        
        # Price tick size (0.01% of mid price)
        tick_size = mid_price * 0.0001
        
        # Generate bid price levels (decreasing)
        for i in range(self.levels):
            # Non-uniform price spacing (closer near the mid price)
            price_step = tick_size * (1 + i * 0.5)
            if i == 0:
                bid_prices[i] = best_bid
            else:
                bid_prices[i] = bid_prices[i-1] - price_step
        
        # Generate ask price levels (increasing)
        for i in range(self.levels):
            price_step = tick_size * (1 + i * 0.5)
            if i == 0:
                ask_prices[i] = best_ask
            else:
                ask_prices[i] = ask_prices[i-1] + price_step
        
        # Generate order sizes (larger near mid price, decaying outward)
        bid_sizes = np.zeros(self.levels)
        ask_sizes = np.zeros(self.levels)
        
        for i in range(self.levels):
            # Liquidity decays exponentially away from mid price
            base_size = np.random.lognormal(
                mean=self.order_size_mean, 
                sigma=self.order_size_std
            )
            
            # Apply decay based on distance from mid price
            decay_factor = np.exp(-depth_decay * i)
            
            bid_sizes[i] = base_size * decay_factor
            ask_sizes[i] = base_size * decay_factor
        
        return bid_prices, bid_sizes, ask_prices, ask_sizes
    
    def generate_order_book_series(self, n_snapshots=100, time_delta='1s', seed=None):
        """Generate a time series of order book snapshots.
        
        Args:
            n_snapshots: Number of order book snapshots to generate
            time_delta: Time between snapshots
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with order book series data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate the mid price process
        mid_price_df = self.price_generator.generate_price_series(n_snapshots, '1s', seed)
        mid_prices = mid_price_df['price'].values
        
        # Initialize arrays for order book data
        timestamps = mid_price_df.index
        bid_prices = np.zeros((n_snapshots, self.levels))
        bid_sizes = np.zeros((n_snapshots, self.levels))
        ask_prices = np.zeros((n_snapshots, self.levels))
        ask_sizes = np.zeros((n_snapshots, self.levels))
        
        # Generate order book at each timestamp
        for i in range(n_snapshots):
            # Random spread with mean reversion
            if i > 0:
                prev_spread = (ask_prices[i-1, 0] - bid_prices[i-1, 0]) / mid_prices[i-1]
                spread = prev_spread * 0.9 + self.spread_mean * 0.1 + np.random.normal(0, self.spread_mean / 10)
                spread = max(spread, self.spread_mean / 2)  # Ensure minimum spread
            else:
                spread = self.spread_mean
            
            # Generate order book snapshot
            depth_decay = 0.1 + 0.05 * np.random.randn()  # Random decay factor
            bid_p, bid_s, ask_p, ask_s = self.generate_order_book_snapshot(
                mid_prices[i], spread, depth_decay)
            
            # Store in arrays
            bid_prices[i, :] = bid_p
            bid_sizes[i, :] = bid_s
            ask_prices[i, :] = ask_p
            ask_sizes[i, :] = ask_s
        
        # Create dictionary with results
        result = {
            'timestamps': timestamps,
            'mid_prices': mid_prices,
            'bid_prices': bid_prices,
            'bid_sizes': bid_sizes,
            'ask_prices': ask_prices,
            'ask_sizes': ask_sizes
        }
        
        return result
    
    def plot_order_book(self, order_book, snapshot_idx=0, save_path=None):
        """Plot the order book at a specific snapshot.
        
        Args:
            order_book: Order book data dictionary
            snapshot_idx: Index of the snapshot to plot
            save_path: Path to save the figure (if None, will display)
            
        Returns:
            Matplotlib figure
        """
        bp = order_book['bid_prices'][snapshot_idx]
        bs = order_book['bid_sizes'][snapshot_idx]
        ap = order_book['ask_prices'][snapshot_idx]
        ass = order_book['ask_sizes'][snapshot_idx]
        mid = order_book['mid_prices'][snapshot_idx]
        timestamp = order_book['timestamps'][snapshot_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bids (negative size to plot on left side)
        ax.barh(bp, -bs, height=bp * 0.0002, color='green', alpha=0.6, label='Bids')
        
        # Plot asks
        ax.barh(ap, ass, height=ap * 0.0002, color='red', alpha=0.6, label='Asks')
        
        # Plot mid price
        ax.axhline(y=mid, color='black', linestyle='--', alpha=0.5, label='Mid Price')
        
        # Formatting
        ax.set_xlabel('Order Size')
        ax.set_ylabel('Price')
        ax.set_title(f'Order Book Snapshot at {timestamp}')
        ax.legend()
        
        # Add best bid/ask labels
        ax.annotate(f'Best Bid: {bp[0]:.2f}', xy=(0.02, 0.05), xycoords='axes fraction')
        ax.annotate(f'Best Ask: {ap[0]:.2f}', xy=(0.02, 0.95), xycoords='axes fraction', verticalalignment='top')
        ax.annotate(f'Spread: {(ap[0] - bp[0]):.2f} ({((ap[0] - bp[0]) / mid * 100):.3f}%)', 
                   xy=(0.02, 0.1), xycoords='axes fraction')
        
        if save_path:
            plt.savefig(save_path)
        
        return fig


# Example usage functions

def generate_crypto_dataset(asset_name='BTC', n_days=30, freq='1h', seed=42, include_indicators=True):
    """Generate a complete synthetic crypto dataset with OHLCV and indicators.
    
    Args:
        asset_name: Name of the crypto asset
        n_days: Number of days of data to generate
        freq: Data frequency ('1d', '1h', '1min')
        seed: Random seed for reproducibility
        include_indicators: Whether to include technical indicators
        
    Returns:
        DataFrame with complete market data
    """
    # Create base price generator with realistic parameters
    if asset_name in ['BTC', 'Bitcoin']:
        base_price = 30000.0
        vol = 0.025
        drift = 0.0001
    elif asset_name in ['ETH', 'Ethereum']:
        base_price = 2000.0
        vol = 0.03
        drift = 0.0001
    else:
        base_price = 1.0
        vol = 0.04
        drift = 0.0001
    
    # Create seasonal patterns (e.g., daily, weekly)
    seasonality = {
        'period': 24,  # Daily cycle (hours)
        'amplitude': 0.003,  # 0.3% variation
        'phase': 6  # Peak at 6 hours offset (e.g., 6 AM)
    }
    
    # Add occasional jumps/crashes
    jumps = {
        'probability': 0.005,  # 0.5% chance per candle
        'mean': 0,  # Symmetric jumps (both up and down)
        'std': 0.02  # 2% standard deviation for jumps
    }
    
    # Create price generator
    price_gen = TimeSeriesGenerator(
        start_price=base_price,
        volatility=vol,
        drift=drift,
        seasonality=seasonality,
        jumps=jumps
    )
    
    # Set up indicators if requested
    indicators = []
    if include_indicators:
        indicators = [
            {'type': 'sma', 'window': 20},
            {'type': 'sma', 'window': 50},
            {'type': 'ema', 'window': 9},
            {'type': 'ema', 'window': 21},
            {'type': 'rsi', 'window': 14},
            {'type': 'bollinger', 'window': 20, 'std_dev': 2},
            {'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9}
        ]
    
    # Add market regimes and news events
    market_events = {
        'market_regimes': {
            'num_regimes': 3,  # Bull, sideways, bear
            'mean_duration': 72,  # Average 3 days per regime at hourly frequency
            'vol_multipliers': [0.7, 1.0, 1.5]  # Different volatility per regime
        },
        'news_impact': {
            'frequency': 0.01,  # 1% chance of news per candle
            'mean_impact': 0,  # Symmetric news (both positive and negative)
            'impact_std': 0.015,  # 1.5% standard deviation for news impact
            'decay_factor': 0.85  # News effect decays by 15% each candle
        }
    }
    
    # Create full market data generator
    data_gen = MarketDataGenerator(
        base_generator=price_gen,
        indicators=indicators,
        market_events=market_events
    )
    
    # Calculate number of candles based on frequency and days
    if freq == '1h':
        n_candles = n_days * 24
    elif freq == '1min':
        n_candles = n_days * 24 * 60
    else:  # '1d'
        n_candles = n_days
    
    # Generate market data
    market_data = data_gen.generate_market_data(n_candles=n_candles, freq=freq, seed=seed)
    
    # Add asset name to dataframe
    market_data['asset'] = asset_name
    
    return market_data 