"""
Benchmark strategies for performance comparison.

This module implements standard benchmark strategies used in
quantitative finance for performance evaluation.
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


class BuyAndHoldStrategy(BaseStrategy):
    """
    Simple buy-and-hold benchmark strategy.
    
    Buys all symbols at the start and holds them throughout
    the backtesting period.
    """
    
    def __init__(self, 
                 strategy_id: str = "buy_and_hold",
                 symbols: List[str] = None,
                 equal_weight: bool = True,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize buy-and-hold strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        symbols : List[str]
            List of symbols to hold
        equal_weight : bool
            Whether to use equal weighting
        weights : Dict[str, float], optional
            Custom weights for symbols (must sum to 1.0)
        """
        if symbols is None:
            symbols = []
        
        parameters = {
            'equal_weight': equal_weight,
            'weights': weights or {}
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        self.equal_weight = equal_weight
        self.weights = weights or {}
        self.initialized = False
        
        # Validate weights
        if not equal_weight and weights:
            if not np.isclose(sum(weights.values()), 1.0, rtol=1e-3):
                logger.warning("Weights do not sum to 1.0, normalizing...")
                total_weight = sum(weights.values())
                self.weights = {symbol: weight/total_weight for symbol, weight in weights.items()}
        
        logger.info(f"Buy-and-hold strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """
        Generate buy-and-hold signals.
        
        Only generates signals once at the beginning.
        """
        if self.initialized:
            return []
        
        signals = []
        
        try:
            # Calculate target weights
            if self.equal_weight:
                target_weight = 1.0 / len(self.symbols) if self.symbols else 0.0
                weights = {symbol: target_weight for symbol in self.symbols}
            else:
                weights = self.weights
            
            # Generate buy signals for all symbols
            for symbol in self.symbols:
                if symbol in market_data.data:
                    target_percent = weights.get(symbol, 0.0)
                    
                    if target_percent > 0:
                        signals.append(SignalEvent(
                            timestamp=market_data.timestamp,
                            strategy_id=self.strategy_id,
                            symbol=symbol,
                            signal_type=OrderSide.BUY,
                            strength=1.0,
                            target_percent=target_percent,
                            metadata={
                                'strategy_type': 'buy_and_hold',
                                'initial_allocation': True,
                                'target_weight': target_percent
                            }
                        ))
            
            self.initialized = True
            logger.info(f"Buy-and-hold strategy initialized with {len(signals)} positions")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating buy-and-hold signals: {e}")
            return []


class EqualWeightRebalancingStrategy(BaseStrategy):
    """
    Equal weight rebalancing strategy.
    
    Maintains equal weights across all symbols by rebalancing
    at regular intervals.
    """
    
    def __init__(self, 
                 strategy_id: str = "equal_weight_rebalancing",
                 symbols: List[str] = None,
                 rebalance_frequency: int = 20,
                 rebalance_threshold: float = 0.05):
        """
        Initialize equal weight rebalancing strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        symbols : List[str]
            List of symbols to trade
        rebalance_frequency : int
            Rebalancing frequency in trading days
        rebalance_threshold : float
            Threshold for rebalancing (deviation from target weight)
        """
        if symbols is None:
            symbols = []
        
        parameters = {
            'rebalance_frequency': rebalance_frequency,
            'rebalance_threshold': rebalance_threshold
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_threshold = rebalance_threshold
        self.last_rebalance_date: Optional[datetime] = None
        self.target_weight = 1.0 / len(symbols) if symbols else 0.0
        self.day_counter = 0
        
        logger.info(f"Equal weight rebalancing strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """
        Generate rebalancing signals.
        """
        signals = []
        
        try:
            self.day_counter += 1
            
            # Check if it's time to rebalance
            should_rebalance = False
            
            # Time-based rebalancing
            if self.day_counter % self.rebalance_frequency == 0:
                should_rebalance = True
                
            # Threshold-based rebalancing
            if not should_rebalance:
                should_rebalance = self._check_rebalance_threshold(portfolio)
            
            if should_rebalance:
                signals = self._generate_rebalance_signals(market_data, portfolio)
                self.last_rebalance_date = market_data.timestamp
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating rebalancing signals: {e}")
            return []
    
    def _check_rebalance_threshold(self, portfolio: Portfolio) -> bool:
        """Check if any position deviates beyond the threshold."""
        try:
            total_value = portfolio.total_value
            
            if total_value <= 0:
                return False
            
            for symbol in self.symbols:
                current_value = portfolio.positions.get(symbol, 0) * portfolio.current_prices.get(symbol, 0)
                current_weight = current_value / total_value
                
                deviation = abs(current_weight - self.target_weight)
                if deviation > self.rebalance_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rebalance threshold: {e}")
            return False
    
    def _generate_rebalance_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """Generate signals to rebalance to target weights."""
        signals = []
        
        try:
            for symbol in self.symbols:
                if symbol in market_data.data:
                    signals.append(SignalEvent(
                        timestamp=market_data.timestamp,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=OrderSide.BUY,  # Will be determined by execution system
                        strength=1.0,
                        target_percent=self.target_weight,
                        metadata={
                            'strategy_type': 'equal_weight_rebalancing',
                            'rebalance_signal': True,
                            'target_weight': self.target_weight,
                            'day_counter': self.day_counter
                        }
                    ))
            
            logger.info(f"Generated {len(signals)} rebalancing signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating rebalance signals: {e}")
            return []


class RiskParityStrategy(BaseStrategy):
    """
    Risk parity strategy.
    
    Allocates capital based on risk contribution rather than
    capital weight, targeting equal risk contribution from each asset.
    """
    
    def __init__(self, 
                 strategy_id: str = "risk_parity",
                 symbols: List[str] = None,
                 lookback_period: int = 60,
                 rebalance_frequency: int = 20,
                 min_weight: float = 0.05,
                 max_weight: float = 0.5):
        """
        Initialize risk parity strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        symbols : List[str]
            List of symbols to trade
        lookback_period : int
            Period for volatility calculation
        rebalance_frequency : int
            Rebalancing frequency in trading days
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        """
        if symbols is None:
            symbols = []
        
        parameters = {
            'lookback_period': lookback_period,
            'rebalance_frequency': rebalance_frequency,
            'min_weight': min_weight,
            'max_weight': max_weight
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.day_counter = 0
        self.target_weights: Dict[str, float] = {}
        
        logger.info(f"Risk parity strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """
        Generate risk parity signals.
        """
        signals = []
        
        try:
            self.day_counter += 1
            self.update_indicators(market_data)
            
            # Rebalance at specified frequency
            if self.day_counter % self.rebalance_frequency == 0:
                self._calculate_risk_parity_weights()
                signals = self._generate_rebalance_signals(market_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating risk parity signals: {e}")
            return []
    
    def _calculate_risk_parity_weights(self) -> None:
        """Calculate risk parity weights based on inverse volatility."""
        try:
            # Calculate volatilities
            volatilities = {}
            
            for symbol in self.symbols:
                if len(self.price_history[symbol]) >= self.lookback_period:
                    prices = np.array(self.price_history[symbol][-self.lookback_period:])
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    volatilities[symbol] = max(volatility, 1e-6)  # Avoid division by zero
            
            if not volatilities:
                logger.warning("No volatilities calculated, using equal weights")
                equal_weight = 1.0 / len(self.symbols)
                self.target_weights = {symbol: equal_weight for symbol in self.symbols}
                return
            
            # Calculate inverse volatility weights
            inv_vol_weights = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
            total_inv_vol = sum(inv_vol_weights.values())
            
            # Normalize weights
            raw_weights = {symbol: weight / total_inv_vol for symbol, weight in inv_vol_weights.items()}
            
            # Apply constraints
            self.target_weights = {}
            for symbol in self.symbols:
                if symbol in raw_weights:
                    weight = max(self.min_weight, min(self.max_weight, raw_weights[symbol]))
                    self.target_weights[symbol] = weight
                else:
                    self.target_weights[symbol] = self.min_weight
            
            # Renormalize to ensure weights sum to 1
            total_weight = sum(self.target_weights.values())
            if total_weight > 0:
                self.target_weights = {symbol: weight / total_weight 
                                     for symbol, weight in self.target_weights.items()}
            
            logger.info(f"Risk parity weights calculated: {self.target_weights}")
            
        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / len(self.symbols)
            self.target_weights = {symbol: equal_weight for symbol in self.symbols}
    
    def _generate_rebalance_signals(self, market_data: MarketEvent) -> List[SignalEvent]:
        """Generate signals to rebalance to target weights."""
        signals = []
        
        try:
            for symbol in self.symbols:
                if symbol in market_data.data and symbol in self.target_weights:
                    signals.append(SignalEvent(
                        timestamp=market_data.timestamp,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=OrderSide.BUY,
                        strength=1.0,
                        target_percent=self.target_weights[symbol],
                        metadata={
                            'strategy_type': 'risk_parity',
                            'rebalance_signal': True,
                            'target_weight': self.target_weights[symbol],
                            'day_counter': self.day_counter
                        }
                    ))
            
            logger.info(f"Generated {len(signals)} risk parity rebalancing signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating risk parity rebalance signals: {e}")
            return []


class SixtyFortyStrategy(BaseStrategy):
    """
    Classic 60/40 portfolio strategy.
    
    Maintains 60% stocks and 40% bonds allocation with periodic rebalancing.
    """
    
    def __init__(self, 
                 strategy_id: str = "sixty_forty",
                 stock_symbols: List[str] = None,
                 bond_symbols: List[str] = None,
                 stock_weight: float = 0.6,
                 bond_weight: float = 0.4,
                 rebalance_frequency: int = 60):
        """
        Initialize 60/40 strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        stock_symbols : List[str]
            List of stock symbols
        bond_symbols : List[str]
            List of bond symbols
        stock_weight : float
            Target weight for stocks
        bond_weight : float
            Target weight for bonds
        rebalance_frequency : int
            Rebalancing frequency in trading days
        """
        if stock_symbols is None:
            stock_symbols = []
        if bond_symbols is None:
            bond_symbols = []
        
        symbols = stock_symbols + bond_symbols
        
        parameters = {
            'stock_weight': stock_weight,
            'bond_weight': bond_weight,
            'rebalance_frequency': rebalance_frequency
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        self.stock_symbols = stock_symbols
        self.bond_symbols = bond_symbols
        self.stock_weight = stock_weight
        self.bond_weight = bond_weight
        self.rebalance_frequency = rebalance_frequency
        self.day_counter = 0
        
        # Calculate individual weights
        stock_individual_weight = stock_weight / len(stock_symbols) if stock_symbols else 0.0
        bond_individual_weight = bond_weight / len(bond_symbols) if bond_symbols else 0.0
        
        self.target_weights = {}
        for symbol in stock_symbols:
            self.target_weights[symbol] = stock_individual_weight
        for symbol in bond_symbols:
            self.target_weights[symbol] = bond_individual_weight
        
        logger.info(f"60/40 strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """
        Generate 60/40 allocation signals.
        """
        signals = []
        
        try:
            self.day_counter += 1
            
            # Initial allocation or periodic rebalancing
            if self.day_counter == 1 or self.day_counter % self.rebalance_frequency == 0:
                signals = self._generate_allocation_signals(market_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating 60/40 signals: {e}")
            return []
    
    def _generate_allocation_signals(self, market_data: MarketEvent) -> List[SignalEvent]:
        """Generate signals for 60/40 allocation."""
        signals = []
        
        try:
            for symbol in self.symbols:
                if symbol in market_data.data and symbol in self.target_weights:
                    signals.append(SignalEvent(
                        timestamp=market_data.timestamp,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=OrderSide.BUY,
                        strength=1.0,
                        target_percent=self.target_weights[symbol],
                        metadata={
                            'strategy_type': 'sixty_forty',
                            'asset_class': 'stock' if symbol in self.stock_symbols else 'bond',
                            'target_weight': self.target_weights[symbol],
                            'allocation_signal': True
                        }
                    ))
            
            logger.info(f"Generated {len(signals)} 60/40 allocation signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating 60/40 allocation signals: {e}")
            return []
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for analysis."""
        state = super().get_strategy_state()
        
        # Add 60/40 specific state
        state.update({
            'stock_symbols': self.stock_symbols,
            'bond_symbols': self.bond_symbols,
            'stock_weight': self.stock_weight,
            'bond_weight': self.bond_weight,
            'target_weights': self.target_weights,
            'rebalance_frequency': self.rebalance_frequency
        })
        
        return state
