"""
Base strategy class and framework for developing trading strategies.

This module provides the abstract base class for all trading strategies
and utility functions for strategy development.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ..core.engine import BacktestEngine
    from ..core.events import SignalEvent, MarketDataSnapshot
    from ..core.portfolio import Portfolio


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class provides the interface that all strategies must implement
    and common functionality for strategy development.
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            symbols: List of symbols this strategy trades
            parameters: Strategy-specific parameters
        """
        self.strategy_id = strategy_id
        self.symbols = set(symbols)
        self.parameters = parameters or {}
        
        # Strategy state
        self.engine: Optional['BacktestEngine'] = None
        self.is_active = True
        self.position_limits: Dict[str, float] = {}
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        
        # Data storage for indicators
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.indicator_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized strategy {strategy_id} for symbols: {symbols}")
    
    def set_engine(self, engine: 'BacktestEngine') -> None:
        """Set the backtesting engine reference."""
        self.engine = engine
    
    @abstractmethod
    def generate_signals(
        self, 
        market_data: 'MarketDataSnapshot', 
        portfolio: 'Portfolio'
    ) -> List['SignalEvent']:
        """
        Generate trading signals based on market data and portfolio state.
        
        Args:
            market_data: Current market data snapshot
            portfolio: Current portfolio state
            
        Returns:
            List of signal events
        """
        pass
    
    def update_indicators(self, market_data: 'MarketDataSnapshot') -> None:
        """
        Update technical indicators with new market data.
        
        Args:
            market_data: Current market data snapshot
        """
        # Update price history
        for symbol in self.symbols:
            if symbol in market_data.data:
                price = float(market_data.get_price(symbol, 'close'))
                self.price_history[symbol].append(price)
                
                # Limit history size for memory efficiency
                max_history = self.parameters.get('max_history', 1000)
                if len(self.price_history[symbol]) > max_history:
                    self.price_history[symbol] = self.price_history[symbol][-max_history:]
    
    def calculate_sma(self, symbol: str, period: int) -> Optional[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            symbol: Trading symbol
            period: Lookback period
            
        Returns:
            SMA value or None if insufficient data
        """
        if symbol not in self.price_history:
            return None
        
        prices = self.price_history[symbol]
        if len(prices) < period:
            return None
        
        return sum(prices[-period:]) / period
    
    def calculate_ema(self, symbol: str, period: int, alpha: Optional[float] = None) -> Optional[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            symbol: Trading symbol
            period: Lookback period
            alpha: Smoothing factor (default: 2/(period+1))
            
        Returns:
            EMA value or None if insufficient data
        """
        if symbol not in self.price_history:
            return None
        
        prices = self.price_history[symbol]
        if len(prices) < 2:
            return None
        
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        # Get cached EMA or calculate from SMA
        cache_key = f"ema_{period}"
        if symbol not in self.indicator_cache:
            self.indicator_cache[symbol] = {}
        
        if cache_key not in self.indicator_cache[symbol]:
            # Initialize with SMA
            if len(prices) >= period:
                sma = sum(prices[:period]) / period
                self.indicator_cache[symbol][cache_key] = sma
            else:
                return None
        
        # Update EMA
        prev_ema = self.indicator_cache[symbol][cache_key]
        current_price = prices[-1]
        ema = alpha * current_price + (1 - alpha) * prev_ema
        self.indicator_cache[symbol][cache_key] = ema
        
        return ema
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index.
        
        Args:
            symbol: Trading symbol
            period: Lookback period
            
        Returns:
            RSI value or None if insufficient data
        """
        if symbol not in self.price_history:
            return None
        
        prices = self.price_history[symbol]
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(0, change) for change in price_changes[-period:]]
        losses = [max(0, -change) for change in price_changes[-period:]]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(
        self, 
        symbol: str, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Optional[Dict[str, float]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            symbol: Trading symbol
            period: Lookback period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        if symbol not in self.price_history:
            return None
        
        prices = self.price_history[symbol]
        if len(prices) < period:
            return None
        
        recent_prices = prices[-period:]
        middle_band = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((price - middle_band) ** 2 for price in recent_prices) / period
        std = variance ** 0.5
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'std': std
        }
    
    def calculate_macd(
        self, 
        symbol: str, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Optional[Dict[str, float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            symbol: Trading symbol
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if symbol not in self.price_history:
            return None
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(symbol, fast_period)
        slow_ema = self.calculate_ema(symbol, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
        
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD)
        cache_key = f"macd_signal_{signal_period}"
        if symbol not in self.indicator_cache:
            self.indicator_cache[symbol] = {}
        
        if cache_key not in self.indicator_cache[symbol]:
            self.indicator_cache[symbol][cache_key] = macd_line
        
        alpha = 2.0 / (signal_period + 1)
        prev_signal = self.indicator_cache[symbol][cache_key]
        signal_line = alpha * macd_line + (1 - alpha) * prev_signal
        self.indicator_cache[symbol][cache_key] = signal_line
        
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def get_position_size(self, symbol: str, portfolio: 'Portfolio') -> int:
        """Get current position size for a symbol."""
        return portfolio.get_position_size(symbol)
    
    def get_portfolio_value(self, portfolio: 'Portfolio') -> float:
        """Get current portfolio value."""
        return float(portfolio.calculate_total_equity())
    
    def calculate_position_size(
        self, 
        symbol: str, 
        signal_strength: float,
        portfolio: 'Portfolio',
        max_position_pct: float = 0.1
    ) -> int:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (0.0 to 1.0)
            portfolio: Current portfolio state
            max_position_pct: Maximum position size as percentage of portfolio
            
        Returns:
            Position size in shares
        """
        if not self.engine or not self.engine.current_snapshot:
            return 0
        
        current_price = self.engine.current_snapshot.get_price(symbol)
        if current_price is None:
            return 0
        
        portfolio_value = portfolio.calculate_total_equity()
        max_position_value = portfolio_value * max_position_pct
        
        # Adjust by signal strength
        target_value = max_position_value * signal_strength
        
        # Check position limits
        if symbol in self.position_limits:
            limit_value = portfolio_value * self.position_limits[symbol]
            target_value = min(target_value, limit_value)
        
        position_size = int(target_value / current_price)
        return position_size
    
    def set_position_limit(self, symbol: str, limit_pct: float) -> None:
        """Set position limit for a symbol as percentage of portfolio."""
        self.position_limits[symbol] = limit_pct
    
    def is_market_open(self, timestamp: datetime) -> bool:
        """
        Check if market is open at given timestamp.
        
        This is a simple implementation. In practice, you would use
        proper market calendar libraries.
        """
        # Simple implementation: weekdays only
        return timestamp.weekday() < 5
    
    def log_signal(self, signal: 'SignalEvent') -> None:
        """Log signal generation for debugging."""
        self.signals_generated += 1
        logger.debug(f"Strategy {self.strategy_id} generated signal: "
                    f"{signal.symbol} {getattr(signal.signal_type, 'value', signal.signal_type)} "
                    f"strength={signal.strength:.2f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            'strategy_id': self.strategy_id,
            'symbols': list(self.symbols),
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'is_active': self.is_active,
            'parameters': self.parameters
        }
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        self.signals_generated = 0
        self.trades_executed = 0
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.indicator_cache = {}
        self.is_active = True
        
        logger.info(f"Strategy {self.strategy_id} reset")
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        return (f"{self.__class__.__name__}(id={self.strategy_id}, "
                f"symbols={list(self.symbols)}, "
                f"signals={self.signals_generated})")
