"""
Technical analysis module for cryptocurrency trading.
This package includes various technical indicators, oscillators, chart pattern recognition,
and other technical analysis tools.
"""

from . import indicators
from . import oscillators
from . import chart_patterns
from . import fibonacci
from . import elliot_wave
from . import ichimoku
from . import divergence
from . import custom_indicators

# Import common functions directly
from .indicators import (
    sma, ema, wma, macd, bollinger_bands, atr, adx, 
    vwap, price_channels, standard_deviation, moving_variance
)

from .oscillators import (
    rsi, stochastic, cci, williams_r, ultimate_oscillator,
    awesome_oscillator, momentum, rate_of_change, tsi
)

from .chart_patterns import (
    head_and_shoulders, double_top_bottom, triangle_patterns,
    rectangle_patterns, wedge_patterns, flag_pennant_patterns,
    support_resistance
)

from .fibonacci import (
    fibonacci_retracement, fibonacci_extension, fibonacci_projection,
    fibonacci_time_zones, fibonacci_fan, fibonacci_circles
)

from .elliot_wave import (
    identify_impulse_wave, identify_corrective_wave, wave_degree,
    wave_channels, wave_relationships, elliott_oscillator
)

from .ichimoku import (
    ichimoku_cloud, tenkan_sen, kijun_sen, senkou_span_a,
    senkou_span_b, chikou_span, kumo_analysis
)

from .divergence import (
    regular_divergence, hidden_divergence, exaggerated_divergence,
    find_swing_points, identify_divergence, divergence_strength
)

from .custom_indicators import (
    supertrend, donchian_channel, keltner_channel, pivot_points,
    volume_profile, market_facilitation_index, elder_force_index,
    guppy_multiple_moving_average
)
