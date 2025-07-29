"""
Position sizing algorithms for risk management.

This module implements various position sizing methods including
Kelly criterion, fixed fractional, and volatility targeting.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core.portfolio import Portfolio
from ..core.events import MarketEvent, SignalEvent


logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    
    target_quantity: int
    target_value: Decimal
    risk_percent: float
    sizing_method: str
    confidence: float  # 0-1, confidence in the sizing
    metadata: Dict[str, Any] = None


class BasePositionSizer(ABC):
    """Abstract base class for position sizing algorithms."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate_position_size(self, signal: SignalEvent, portfolio: Portfolio,
                              market_data: MarketEvent, **kwargs) -> PositionSizeResult:
        """
        Calculate position size for a signal.
        
        Parameters
        ----------
        signal : SignalEvent
            Trading signal
        portfolio : Portfolio
            Current portfolio state
        market_data : MarketEvent
            Current market data
        **kwargs
            Additional parameters
            
        Returns
        -------
        PositionSizeResult
            Position sizing result
        """
        pass


class FixedFractionalSizer(BasePositionSizer):
    """
    Fixed fractional position sizing.
    
    Allocates a fixed percentage of portfolio value to each position.
    """
    
    def __init__(self, fraction: float = 0.02, max_positions: int = 50):
        """
        Initialize fixed fractional sizer.
        
        Parameters
        ----------
        fraction : float
            Fraction of portfolio to risk per position (default 2%)
        max_positions : int
            Maximum number of positions to hold
        """
        super().__init__("Fixed Fractional")
        self.fraction = fraction
        self.max_positions = max_positions
        
        logger.info(f"Initialized {self.name} sizer: fraction={fraction}, "
                   f"max_positions={max_positions}")
    
    def calculate_position_size(self, signal: SignalEvent, portfolio: Portfolio,
                              market_data: MarketEvent, **kwargs) -> PositionSizeResult:
        """Calculate position size using fixed fraction."""
        try:
            # Get portfolio value
            portfolio_value = portfolio.calculate_total_equity()
            if portfolio_value <= 0:
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            # Adjust fraction based on number of positions
            current_positions = len(portfolio.positions)
            if current_positions >= self.max_positions:
                # Don't open new positions if at limit
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            # Calculate target value
            adjusted_fraction = min(self.fraction, 1.0 / (current_positions + 1))
            target_value = portfolio_value * Decimal(str(adjusted_fraction))
            
            # Get current price
            current_price = market_data.data.get(signal.symbol, {}).get('close')
            if not current_price or current_price <= 0:
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            # Calculate quantity
            target_quantity = int(target_value / current_price)
            
            # Adjust for signal strength
            if hasattr(signal, 'strength') and signal.strength:
                target_quantity = int(target_quantity * signal.strength)
            
            # Ensure we don't exceed cash
            required_cash = target_quantity * current_price
            available_cash = portfolio.cash
            
            if required_cash > available_cash:
                target_quantity = int(available_cash / current_price)
                target_value = target_quantity * current_price
            
            risk_percent = float(target_value / portfolio_value) if portfolio_value > 0 else 0.0
            
            return PositionSizeResult(
                target_quantity=target_quantity,
                target_value=target_value,
                risk_percent=risk_percent,
                sizing_method=self.name,
                confidence=0.8,
                metadata={
                    'fraction_used': adjusted_fraction,
                    'signal_strength': getattr(signal, 'strength', 1.0)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in fixed fractional sizing: {e}")
            return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)


class KellyCriterionSizer(BasePositionSizer):
    """
    Kelly Criterion position sizing.
    
    Optimizes position size based on historical win rate and risk/reward ratio.
    """
    
    def __init__(self, lookback_periods: int = 100, max_kelly_fraction: float = 0.25):
        """
        Initialize Kelly criterion sizer.
        
        Parameters
        ----------
        lookback_periods : int
            Number of historical trades to analyze
        max_kelly_fraction : float
            Maximum Kelly fraction to use (to avoid over-leverage)
        """
        super().__init__("Kelly Criterion")
        self.lookback_periods = lookback_periods
        self.max_kelly_fraction = max_kelly_fraction
        
        logger.info(f"Initialized {self.name} sizer: lookback={lookback_periods}, "
                   f"max_fraction={max_kelly_fraction}")
    
    def calculate_position_size(self, signal: SignalEvent, portfolio: Portfolio,
                              market_data: MarketEvent, **kwargs) -> PositionSizeResult:
        """Calculate position size using Kelly criterion."""
        try:
            # Get historical trade data for this symbol
            trade_history = self._get_trade_history(signal.symbol, portfolio)
            
            if len(trade_history) < 10:  # Need minimum trades for Kelly
                # Fall back to fixed fractional
                fallback_sizer = FixedFractionalSizer(0.02)
                result = fallback_sizer.calculate_position_size(signal, portfolio, market_data)
                result.sizing_method = f"{self.name} (fallback)"
                result.confidence = 0.3
                return result
            
            # Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(trade_history)
            
            # Apply maximum limit
            kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
            kelly_fraction = max(kelly_fraction, 0.0)  # No negative sizing
            
            # Calculate position size
            portfolio_value = portfolio.calculate_total_equity()
            if portfolio_value <= 0:
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            target_value = portfolio_value * Decimal(str(kelly_fraction))
            
            # Get current price
            current_price = market_data.data.get(signal.symbol, {}).get('close')
            if not current_price or current_price <= 0:
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            target_quantity = int(target_value / current_price)
            
            # Ensure we don't exceed cash
            required_cash = target_quantity * current_price
            if required_cash > portfolio.cash:
                target_quantity = int(portfolio.cash / current_price)
                target_value = target_quantity * current_price
            
            risk_percent = float(target_value / portfolio_value) if portfolio_value > 0 else 0.0
            
            return PositionSizeResult(
                target_quantity=target_quantity,
                target_value=target_value,
                risk_percent=risk_percent,
                sizing_method=self.name,
                confidence=0.9 if len(trade_history) >= 50 else 0.6,
                metadata={
                    'kelly_fraction': kelly_fraction,
                    'trade_count': len(trade_history),
                    'win_rate': self._calculate_win_rate(trade_history)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Kelly criterion sizing: {e}")
            return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
    
    def _get_trade_history(self, symbol: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Get historical trade data for Kelly calculation."""
        # This would extract completed trades for the symbol
        # For now, return empty list
        return []
    
    def _calculate_kelly_fraction(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate Kelly fraction from trade history."""
        if not trade_history:
            return 0.0
        
        # Extract returns
        returns = [trade.get('return', 0.0) for trade in trade_history]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not wins or not losses:
            return 0.0
        
        # Calculate Kelly parameters
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        return max(0.0, kelly_fraction)
    
    def _calculate_win_rate(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trade history."""
        if not trade_history:
            return 0.0
        
        wins = sum(1 for trade in trade_history if trade.get('return', 0.0) > 0)
        return wins / len(trade_history)


class VolatilityTargetingSizer(BasePositionSizer):
    """
    Volatility targeting position sizing.
    
    Adjusts position size to target a specific portfolio volatility.
    """
    
    def __init__(self, target_volatility: float = 0.15, lookback_days: int = 60):
        """
        Initialize volatility targeting sizer.
        
        Parameters
        ----------
        target_volatility : float
            Target annualized portfolio volatility (default 15%)
        lookback_days : int
            Number of days to use for volatility calculation
        """
        super().__init__("Volatility Targeting")
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        
        logger.info(f"Initialized {self.name} sizer: target_vol={target_volatility}, "
                   f"lookback={lookback_days}")
    
    def calculate_position_size(self, signal: SignalEvent, portfolio: Portfolio,
                              market_data: MarketEvent, **kwargs) -> PositionSizeResult:
        """Calculate position size using volatility targeting."""
        try:
            # Get historical price data for volatility calculation
            price_history = self._get_price_history(signal.symbol, market_data)
            
            if len(price_history) < self.lookback_days:
                # Fall back to fixed fractional
                fallback_sizer = FixedFractionalSizer(0.02)
                result = fallback_sizer.calculate_position_size(signal, portfolio, market_data)
                result.sizing_method = f"{self.name} (fallback)"
                result.confidence = 0.3
                return result
            
            # Calculate asset volatility
            returns = np.diff(np.log(price_history))
            asset_volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            if asset_volatility <= 0:
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            # Calculate position size for target volatility
            portfolio_value = portfolio.calculate_total_equity()
            if portfolio_value <= 0:
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            # Position sizing: target_vol / asset_vol * portfolio_value
            position_fraction = self.target_volatility / asset_volatility
            position_fraction = min(position_fraction, 1.0)  # Cap at 100%
            
            target_value = portfolio_value * Decimal(str(position_fraction))
            
            # Get current price
            current_price = market_data.data.get(signal.symbol, {}).get('close')
            if not current_price or current_price <= 0:
                return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
            
            target_quantity = int(target_value / current_price)
            
            # Ensure we don't exceed cash
            required_cash = target_quantity * current_price
            if required_cash > portfolio.cash:
                target_quantity = int(portfolio.cash / current_price)
                target_value = target_quantity * current_price
            
            risk_percent = float(target_value / portfolio_value) if portfolio_value > 0 else 0.0
            
            return PositionSizeResult(
                target_quantity=target_quantity,
                target_value=target_value,
                risk_percent=risk_percent,
                sizing_method=self.name,
                confidence=0.8,
                metadata={
                    'asset_volatility': asset_volatility,
                    'position_fraction': position_fraction,
                    'target_volatility': self.target_volatility
                }
            )
            
        except Exception as e:
            logger.error(f"Error in volatility targeting sizing: {e}")
            return PositionSizeResult(0, Decimal('0'), 0.0, self.name, 0.0)
    
    def _get_price_history(self, symbol: str, market_data: MarketEvent) -> List[float]:
        """Get historical prices for volatility calculation."""
        # This would extract price history from data handler
        # For now, return dummy data
        return [100.0 + i + np.random.normal(0, 2) for i in range(self.lookback_days)]


class PositionSizer:
    """
    Main position sizing coordinator.
    
    Manages multiple position sizing algorithms and provides
    a unified interface for position size calculation.
    """
    
    def __init__(self, default_sizer: BasePositionSizer):
        """
        Initialize position sizer.
        
        Parameters
        ----------
        default_sizer : BasePositionSizer
            Default position sizing algorithm
        """
        self.default_sizer = default_sizer
        self.strategy_sizers: Dict[str, BasePositionSizer] = {}
        self.symbol_sizers: Dict[str, BasePositionSizer] = {}
        
        logger.info(f"Position sizer initialized with default: {default_sizer.name}")
    
    def add_strategy_sizer(self, strategy_id: str, sizer: BasePositionSizer) -> None:
        """Add strategy-specific position sizer."""
        self.strategy_sizers[strategy_id] = sizer
        logger.info(f"Added strategy sizer for {strategy_id}: {sizer.name}")
    
    def add_symbol_sizer(self, symbol: str, sizer: BasePositionSizer) -> None:
        """Add symbol-specific position sizer."""
        self.symbol_sizers[symbol] = sizer
        logger.info(f"Added symbol sizer for {symbol}: {sizer.name}")
    
    def calculate_position_size(self, signal: SignalEvent, portfolio: Portfolio,
                              market_data: MarketEvent) -> PositionSizeResult:
        """
        Calculate position size using appropriate sizer.
        
        Parameters
        ----------
        signal : SignalEvent
            Trading signal
        portfolio : Portfolio
            Current portfolio state
        market_data : MarketEvent
            Current market data
            
        Returns
        -------
        PositionSizeResult
            Position sizing result
        """
        try:
            # Choose appropriate sizer
            sizer = self._get_sizer(signal)
            
            # Calculate position size
            result = sizer.calculate_position_size(signal, portfolio, market_data)
            
            logger.debug(f"Calculated position size for {signal.symbol}: "
                        f"{result.target_quantity} shares using {result.sizing_method}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return PositionSizeResult(0, Decimal('0'), 0.0, "Error", 0.0)
    
    def _get_sizer(self, signal: SignalEvent) -> BasePositionSizer:
        """Get appropriate sizer for signal."""
        # Priority: symbol-specific > strategy-specific > default
        if signal.symbol in self.symbol_sizers:
            return self.symbol_sizers[signal.symbol]
        elif signal.strategy_id in self.strategy_sizers:
            return self.strategy_sizers[signal.strategy_id]
        else:
            return self.default_sizer
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing configuration."""
        return {
            'default_sizer': self.default_sizer.name,
            'strategy_sizers': {sid: sizer.name for sid, sizer in self.strategy_sizers.items()},
            'symbol_sizers': {symbol: sizer.name for symbol, sizer in self.symbol_sizers.items()}
        }
