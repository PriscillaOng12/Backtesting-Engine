"""
Momentum strategy implementation.

This module implements a comprehensive momentum strategy with
multiple timeframes and technical indicators.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal

from .base import BaseStrategy
from ..core.events import SignalEvent, OrderSide, MarketEvent
from ..core.portfolio import Portfolio


logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Multi-timeframe momentum strategy.
    
    Uses price momentum, volume confirmation, and risk management
    to generate trading signals across multiple timeframes.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 symbols: List[str],
                 fast_ma_period: int = 10,
                 slow_ma_period: int = 30,
                 momentum_period: int = 20,
                 volume_ma_period: int = 20,
                 min_momentum_threshold: float = 0.02,
                 volume_confirmation: bool = True,
                 position_size: float = 0.05,
                 stop_loss: Optional[float] = 0.05,
                 take_profit: Optional[float] = 0.15,
                 max_holding_days: Optional[int] = 60):
        """
        Initialize momentum strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        symbols : List[str]
            List of symbols to trade
        fast_ma_period : int
            Fast moving average period
        slow_ma_period : int
            Slow moving average period
        momentum_period : int
            Momentum calculation period
        volume_ma_period : int
            Volume moving average period
        min_momentum_threshold : float
            Minimum momentum for signal generation
        volume_confirmation : bool
            Whether to require volume confirmation
        position_size : float
            Position size as fraction of portfolio
        stop_loss : float, optional
            Stop loss percentage
        take_profit : float, optional
            Take profit percentage
        max_holding_days : int, optional
            Maximum holding period in days
        """
        parameters = {
            'fast_ma_period': fast_ma_period,
            'slow_ma_period': slow_ma_period,
            'momentum_period': momentum_period,
            'volume_ma_period': volume_ma_period,
            'min_momentum_threshold': min_momentum_threshold,
            'volume_confirmation': volume_confirmation,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_holding_days': max_holding_days
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        # Strategy parameters
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.momentum_period = momentum_period
        self.volume_ma_period = volume_ma_period
        self.min_momentum_threshold = min_momentum_threshold
        self.volume_confirmation = volume_confirmation
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_days = max_holding_days
        
        # Technical indicators
        self.fast_ma = {symbol: [] for symbol in symbols}
        self.slow_ma = {symbol: [] for symbol in symbols}
        self.momentum = {symbol: [] for symbol in symbols}
        self.volume_ma = {symbol: [] for symbol in symbols}
        self.volume_history = {symbol: [] for symbol in symbols}
        
        # Position tracking
        self.entry_dates = {symbol: None for symbol in symbols}
        self.entry_prices = {symbol: None for symbol in symbols}
        
        logger.info(f"Momentum strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """
        Generate momentum-based trading signals.
        
        Parameters
        ----------
        market_data : MarketEvent
            Current market data
        portfolio : Portfolio
            Current portfolio state
            
        Returns
        -------
        List[SignalEvent]
            List of generated signals
        """
        signals = []
        
        try:
            # Update indicators
            self.update_indicators(market_data)
            
            for symbol in self.symbols:
                if symbol not in market_data.data:
                    continue
                
                # Get current market data
                current_data = market_data.data[symbol]
                current_price = current_data.get('close')
                current_volume = current_data.get('volume', 0)
                
                if not current_price:
                    continue
                
                # Update volume history
                self.volume_history[symbol].append(current_volume)
                if len(self.volume_history[symbol]) > self.volume_ma_period:
                    self.volume_history[symbol].pop(0)
                
                # Calculate indicators
                self._update_technical_indicators(symbol, current_price, current_volume)
                
                # Check if we have enough data
                if not self._has_sufficient_data(symbol):
                    continue
                
                # Get current position
                current_position = self.get_position_size(symbol, portfolio)
                
                # Generate signals based on momentum
                if current_position == 0:
                    # Look for entry signals
                    entry_signal = self._check_entry_signal(symbol, current_price, market_data.timestamp)
                    if entry_signal:
                        signals.append(entry_signal)
                        self.entry_dates[symbol] = market_data.timestamp
                        self.entry_prices[symbol] = current_price
                else:
                    # Look for exit signals
                    exit_signal = self._check_exit_signal(symbol, current_price, current_position, 
                                                        market_data.timestamp, portfolio)
                    if exit_signal:
                        signals.append(exit_signal)
                        self.entry_dates[symbol] = None
                        self.entry_prices[symbol] = None
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return []
    
    def _update_technical_indicators(self, symbol: str, price: float, volume: float) -> None:
        """Update technical indicators for a symbol."""
        try:
            # Fast MA
            self.fast_ma[symbol] = self.calculate_sma(
                self.price_history[symbol], self.fast_ma_period
            )
            
            # Slow MA
            self.slow_ma[symbol] = self.calculate_sma(
                self.price_history[symbol], self.slow_ma_period
            )
            
            # Momentum
            if len(self.price_history[symbol]) >= self.momentum_period:
                current_price = self.price_history[symbol][-1]
                past_price = self.price_history[symbol][-self.momentum_period]
                momentum_value = (current_price - past_price) / past_price
                
                self.momentum[symbol].append(momentum_value)
                if len(self.momentum[symbol]) > 100:  # Keep history manageable
                    self.momentum[symbol].pop(0)
            
            # Volume MA
            if len(self.volume_history[symbol]) >= self.volume_ma_period:
                vol_ma = np.mean(self.volume_history[symbol][-self.volume_ma_period:])
                self.volume_ma[symbol].append(vol_ma)
                if len(self.volume_ma[symbol]) > 100:
                    self.volume_ma[symbol].pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}")
    
    def _has_sufficient_data(self, symbol: str) -> bool:
        """Check if we have sufficient data for signal generation."""
        return (len(self.price_history[symbol]) >= self.slow_ma_period and
                len(self.momentum[symbol]) >= 1 and
                (not self.volume_confirmation or len(self.volume_ma[symbol]) >= 1))
    
    def _check_entry_signal(self, symbol: str, current_price: float, 
                          timestamp: datetime) -> Optional[SignalEvent]:
        """Check for momentum entry signals."""
        try:
            # Get latest indicator values
            fast_ma = self.fast_ma[symbol][-1] if self.fast_ma[symbol] else None
            slow_ma = self.slow_ma[symbol][-1] if self.slow_ma[symbol] else None
            momentum_val = self.momentum[symbol][-1] if self.momentum[symbol] else None
            
            if not all([fast_ma, slow_ma, momentum_val]):
                return None
            
            # Primary momentum condition
            momentum_bullish = momentum_val > self.min_momentum_threshold
            
            # Moving average crossover condition
            ma_bullish = fast_ma > slow_ma
            
            # Price above fast MA condition
            price_above_ma = current_price > fast_ma
            
            # Volume confirmation (if enabled)
            volume_confirmed = True
            if self.volume_confirmation and self.volume_ma[symbol]:
                current_volume = self.volume_history[symbol][-1] if self.volume_history[symbol] else 0
                avg_volume = self.volume_ma[symbol][-1]
                volume_confirmed = current_volume > avg_volume * 1.2  # 20% above average
            
            # Additional momentum filters
            recent_momentum = self.momentum[symbol][-min(5, len(self.momentum[symbol])):]
            momentum_accelerating = len(recent_momentum) >= 2 and recent_momentum[-1] > recent_momentum[-2]
            
            # Combine all conditions
            if (momentum_bullish and ma_bullish and price_above_ma and 
                volume_confirmed and momentum_accelerating):
                
                # Calculate signal strength based on momentum magnitude
                signal_strength = min(1.0, abs(momentum_val) / (self.min_momentum_threshold * 2))
                
                return SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=OrderSide.BUY,
                    strength=signal_strength,
                    target_percent=self.position_size,
                    metadata={
                        'momentum': momentum_val,
                        'fast_ma': fast_ma,
                        'slow_ma': slow_ma,
                        'volume_confirmed': volume_confirmed,
                        'entry_reason': 'momentum_breakout'
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking entry signal for {symbol}: {e}")
            return None
    
    def _check_exit_signal(self, symbol: str, current_price: float, 
                         current_position: int, timestamp: datetime,
                         portfolio: Portfolio) -> Optional[SignalEvent]:
        """Check for momentum exit signals."""
        try:
            # Get entry information
            entry_price = self.entry_prices[symbol]
            entry_date = self.entry_dates[symbol]
            
            if not entry_price or not entry_date:
                return None
            
            # Calculate position P&L
            if current_position > 0:  # Long position
                pnl_percent = (current_price - entry_price) / entry_price
            else:  # Short position
                pnl_percent = (entry_price - current_price) / entry_price
            
            # Stop loss check
            if self.stop_loss and pnl_percent <= -self.stop_loss:
                return SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=OrderSide.SELL if current_position > 0 else OrderSide.BUY,
                    strength=1.0,
                    target_percent=0.0,
                    metadata={
                        'pnl_percent': pnl_percent,
                        'exit_reason': 'stop_loss',
                        'entry_price': entry_price
                    }
                )
            
            # Take profit check
            if self.take_profit and pnl_percent >= self.take_profit:
                return SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=OrderSide.SELL if current_position > 0 else OrderSide.BUY,
                    strength=1.0,
                    target_percent=0.0,
                    metadata={
                        'pnl_percent': pnl_percent,
                        'exit_reason': 'take_profit',
                        'entry_price': entry_price
                    }
                )
            
            # Maximum holding period check
            if self.max_holding_days:
                days_held = (timestamp - entry_date).days
                if days_held >= self.max_holding_days:
                    return SignalEvent(
                        timestamp=timestamp,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=OrderSide.SELL if current_position > 0 else OrderSide.BUY,
                        strength=1.0,
                        target_percent=0.0,
                        metadata={
                            'pnl_percent': pnl_percent,
                            'exit_reason': 'max_holding_period',
                            'days_held': days_held
                        }
                    )
            
            # Momentum reversal check
            momentum_val = self.momentum[symbol][-1] if self.momentum[symbol] else 0
            
            # For long positions, exit if momentum turns negative
            if current_position > 0 and momentum_val < -self.min_momentum_threshold * 0.5:
                return SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=OrderSide.SELL,
                    strength=1.0,
                    target_percent=0.0,
                    metadata={
                        'pnl_percent': pnl_percent,
                        'exit_reason': 'momentum_reversal',
                        'momentum': momentum_val
                    }
                )
            
            # Moving average exit condition
            fast_ma = self.fast_ma[symbol][-1] if self.fast_ma[symbol] else None
            slow_ma = self.slow_ma[symbol][-1] if self.slow_ma[symbol] else None
            
            if fast_ma and slow_ma and current_position > 0:
                # Exit long if fast MA crosses below slow MA
                if fast_ma < slow_ma:
                    return SignalEvent(
                        timestamp=timestamp,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=OrderSide.SELL,
                        strength=1.0,
                        target_percent=0.0,
                        metadata={
                            'pnl_percent': pnl_percent,
                            'exit_reason': 'ma_crossover',
                            'fast_ma': fast_ma,
                            'slow_ma': slow_ma
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit signal for {symbol}: {e}")
            return None
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for analysis."""
        state = super().get_strategy_state()
        
        # Add momentum-specific state
        state.update({
            'current_momentum': {symbol: self.momentum[symbol][-1] if self.momentum[symbol] else 0 
                               for symbol in self.symbols},
            'current_ma_signals': {symbol: {
                'fast_ma': self.fast_ma[symbol][-1] if self.fast_ma[symbol] else 0,
                'slow_ma': self.slow_ma[symbol][-1] if self.slow_ma[symbol] else 0,
                'ma_bullish': (self.fast_ma[symbol][-1] > self.slow_ma[symbol][-1] 
                             if self.fast_ma[symbol] and self.slow_ma[symbol] else False)
            } for symbol in self.symbols},
            'active_positions': {symbol: {
                'entry_date': self.entry_dates[symbol],
                'entry_price': self.entry_prices[symbol]
            } for symbol in self.symbols if self.entry_dates[symbol]}
        })
        
        return state


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using multiple indicators.
    
    Combines trend identification with momentum confirmation
    for robust signal generation.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 symbols: List[str],
                 trend_period: int = 50,
                 atr_period: int = 20,
                 breakout_multiplier: float = 2.0,
                 position_size: float = 0.04):
        """
        Initialize trend following strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        symbols : List[str]
            List of symbols to trade
        trend_period : int
            Period for trend calculation
        atr_period : int
            Average True Range period
        breakout_multiplier : float
            ATR multiplier for breakout detection
        position_size : float
            Position size as fraction of portfolio
        """
        parameters = {
            'trend_period': trend_period,
            'atr_period': atr_period,
            'breakout_multiplier': breakout_multiplier,
            'position_size': position_size
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier
        self.position_size = position_size
        
        # Technical indicators
        self.atr = {symbol: [] for symbol in symbols}
        self.trend_ma = {symbol: [] for symbol in symbols}
        self.high_history = {symbol: [] for symbol in symbols}
        self.low_history = {symbol: [] for symbol in symbols}
        
        logger.info(f"Trend following strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """Generate trend following signals."""
        signals = []
        
        try:
            self.update_indicators(market_data)
            
            for symbol in self.symbols:
                if symbol not in market_data.data:
                    continue
                
                current_data = market_data.data[symbol]
                current_price = current_data.get('close')
                current_high = current_data.get('high', current_price)
                current_low = current_data.get('low', current_price)
                
                if not current_price:
                    continue
                
                # Update price histories
                self.high_history[symbol].append(current_high)
                self.low_history[symbol].append(current_low)
                
                # Keep histories manageable
                for hist in [self.high_history[symbol], self.low_history[symbol]]:
                    if len(hist) > self.trend_period * 2:
                        hist.pop(0)
                
                # Calculate indicators
                self._update_trend_indicators(symbol, current_price, current_high, current_low)
                
                # Check for signals
                if len(self.price_history[symbol]) >= self.trend_period:
                    signal = self._check_trend_signal(symbol, current_price, market_data.timestamp, portfolio)
                    if signal:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trend signals: {e}")
            return []
    
    def _update_trend_indicators(self, symbol: str, price: float, high: float, low: float) -> None:
        """Update trend indicators."""
        try:
            # Trend MA
            self.trend_ma[symbol] = self.calculate_sma(self.price_history[symbol], self.trend_period)
            
            # ATR calculation
            if len(self.price_history[symbol]) >= 2:
                prev_close = self.price_history[symbol][-2]
                
                # True Range calculation
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                
                # Update ATR
                if not self.atr[symbol]:
                    self.atr[symbol].append(true_range)
                else:
                    # Exponential moving average of True Range
                    alpha = 2.0 / (self.atr_period + 1)
                    new_atr = alpha * true_range + (1 - alpha) * self.atr[symbol][-1]
                    self.atr[symbol].append(new_atr)
                
                # Keep ATR history manageable
                if len(self.atr[symbol]) > 100:
                    self.atr[symbol].pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating trend indicators for {symbol}: {e}")
    
    def _check_trend_signal(self, symbol: str, current_price: float, 
                          timestamp: datetime, portfolio: Portfolio) -> Optional[SignalEvent]:
        """Check for trend following signals."""
        try:
            if not self.trend_ma[symbol] or not self.atr[symbol]:
                return None
            
            trend_ma = self.trend_ma[symbol][-1]
            current_atr = self.atr[symbol][-1]
            current_position = self.get_position_size(symbol, portfolio)
            
            # Calculate breakout thresholds
            upper_threshold = trend_ma + (self.breakout_multiplier * current_atr)
            lower_threshold = trend_ma - (self.breakout_multiplier * current_atr)
            
            # Long entry condition
            if current_position == 0 and current_price > upper_threshold:
                return SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=OrderSide.BUY,
                    strength=1.0,
                    target_percent=self.position_size,
                    metadata={
                        'trend_ma': trend_ma,
                        'atr': current_atr,
                        'breakout_level': upper_threshold,
                        'signal_type': 'trend_breakout_long'
                    }
                )
            
            # Exit long position if price falls below trend
            elif current_position > 0 and current_price < trend_ma:
                return SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=OrderSide.SELL,
                    strength=1.0,
                    target_percent=0.0,
                    metadata={
                        'trend_ma': trend_ma,
                        'exit_reason': 'trend_reversal'
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking trend signal for {symbol}: {e}")
            return None
