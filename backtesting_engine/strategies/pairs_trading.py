"""
Pairs trading strategy implementation.

This module implements statistical arbitrage through pairs trading
using cointegration and mean reversion techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from .base import BaseStrategy
from ..core.events import SignalEvent, OrderSide, MarketEvent
from ..core.portfolio import Portfolio


logger = logging.getLogger(__name__)


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage pairs trading strategy.
    
    Identifies cointegrated pairs and trades the spread when it
    deviates from its historical mean.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 symbols: List[str],
                 lookback_period: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0,
                 min_half_life: int = 5,
                 max_half_life: int = 30,
                 position_size: float = 0.05,
                 cointegration_pvalue: float = 0.05):
        """
        Initialize pairs trading strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        symbols : List[str]
            List of symbols to analyze for pairs
        lookback_period : int
            Period for cointegration analysis
        entry_threshold : float
            Z-score threshold for entry
        exit_threshold : float
            Z-score threshold for exit
        stop_loss_threshold : float
            Z-score threshold for stop loss
        min_half_life : int
            Minimum half-life for mean reversion (days)
        max_half_life : int
            Maximum half-life for mean reversion (days)
        position_size : float
            Position size as fraction of portfolio
        cointegration_pvalue : float
            Maximum p-value for cointegration test
        """
        parameters = {
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'stop_loss_threshold': stop_loss_threshold,
            'min_half_life': min_half_life,
            'max_half_life': max_half_life,
            'position_size': position_size,
            'cointegration_pvalue': cointegration_pvalue
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        # Strategy parameters
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.position_size = position_size
        self.cointegration_pvalue = cointegration_pvalue
        
        # Pairs tracking
        self.pairs: List[Tuple[str, str]] = []
        self.pair_coefficients: Dict[Tuple[str, str], float] = {}
        self.pair_spreads: Dict[Tuple[str, str], List[float]] = {}
        self.pair_z_scores: Dict[Tuple[str, str], List[float]] = {}
        self.active_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Last analysis timestamp
        self.last_analysis_date: Optional[datetime] = None
        self.analysis_frequency_days = 7  # Re-analyze pairs weekly
        
        logger.info(f"Pairs trading strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """
        Generate pairs trading signals.
        
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
            # Update price history
            self.update_indicators(market_data)
            
            # Check if we need to re-analyze pairs
            if self._should_analyze_pairs(market_data.timestamp):
                self._analyze_pairs()
                self.last_analysis_date = market_data.timestamp
            
            # Generate signals for existing pairs
            for pair in self.pairs:
                pair_signals = self._generate_pair_signals(pair, market_data, portfolio)
                signals.extend(pair_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating pairs trading signals: {e}")
            return []
    
    def _should_analyze_pairs(self, current_date: datetime) -> bool:
        """Check if pairs should be re-analyzed."""
        if self.last_analysis_date is None:
            return True
        
        days_since_analysis = (current_date - self.last_analysis_date).days
        return days_since_analysis >= self.analysis_frequency_days
    
    def _analyze_pairs(self) -> None:
        """Analyze all possible pairs for cointegration."""
        try:
            logger.info("Analyzing pairs for cointegration...")
            
            # Clear existing pairs
            self.pairs.clear()
            self.pair_coefficients.clear()
            self.pair_spreads.clear()
            self.pair_z_scores.clear()
            
            # Check if we have enough data
            min_data_length = max(self.lookback_period, 30)
            valid_symbols = [symbol for symbol in self.symbols 
                           if len(self.price_history[symbol]) >= min_data_length]
            
            if len(valid_symbols) < 2:
                logger.warning("Insufficient data for pairs analysis")
                return
            
            # Test all possible pairs
            for i, symbol1 in enumerate(valid_symbols):
                for symbol2 in valid_symbols[i+1:]:
                    if self._test_cointegration(symbol1, symbol2):
                        pair = (symbol1, symbol2)
                        self.pairs.append(pair)
                        logger.info(f"Found cointegrated pair: {symbol1} - {symbol2}")
            
            logger.info(f"Identified {len(self.pairs)} cointegrated pairs")
            
        except Exception as e:
            logger.error(f"Error analyzing pairs: {e}")
    
    def _test_cointegration(self, symbol1: str, symbol2: str) -> bool:
        """Test if two symbols are cointegrated."""
        try:
            # Get price series
            prices1 = np.array(self.price_history[symbol1][-self.lookback_period:])
            prices2 = np.array(self.price_history[symbol2][-self.lookback_period:])
            
            if len(prices1) != len(prices2) or len(prices1) < 30:
                return False
            
            # Perform cointegration test
            score, pvalue, _ = coint(prices1, prices2)
            
            if pvalue <= self.cointegration_pvalue:
                # Calculate hedge ratio using linear regression
                hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)
                
                # Calculate spread
                spread = prices1 - hedge_ratio * prices2
                
                # Test mean reversion properties
                half_life = self._calculate_half_life(spread)
                
                if self.min_half_life <= half_life <= self.max_half_life:
                    # Store pair information
                    pair = (symbol1, symbol2)
                    self.pair_coefficients[pair] = hedge_ratio
                    
                    # Initialize spread and z-score histories
                    self.pair_spreads[pair] = spread.tolist()
                    self.pair_z_scores[pair] = self._calculate_z_scores(spread).tolist()
                    
                    logger.debug(f"Cointegrated pair found: {symbol1}-{symbol2}, "
                               f"p-value: {pvalue:.4f}, half-life: {half_life:.1f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error testing cointegration for {symbol1}-{symbol2}: {e}")
            return False
    
    def _calculate_hedge_ratio(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate hedge ratio using linear regression."""
        try:
            # Use OLS regression: prices1 = alpha + beta * prices2 + error
            X = sm.add_constant(prices2)
            model = sm.OLS(prices1, X).fit()
            return model.params[1]  # Beta coefficient
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}")
            return 1.0
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """Calculate half-life of mean reversion."""
        try:
            # Calculate lagged spread
            spread_lag = spread[:-1]
            spread_diff = np.diff(spread)
            
            # Regression: spread_diff = alpha + beta * spread_lag + error
            X = sm.add_constant(spread_lag)
            model = sm.OLS(spread_diff, X).fit()
            
            beta = model.params[1]
            
            if beta < 0:
                half_life = -np.log(2) / beta
                return half_life
            else:
                return float('inf')  # No mean reversion
                
        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return float('inf')
    
    def _calculate_z_scores(self, spread: np.ndarray) -> np.ndarray:
        """Calculate z-scores for the spread."""
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)
        
        if std_spread == 0:
            return np.zeros_like(spread)
        
        return (spread - mean_spread) / std_spread
    
    def _generate_pair_signals(self, pair: Tuple[str, str], 
                             market_data: MarketEvent, 
                             portfolio: Portfolio) -> List[SignalEvent]:
        """Generate signals for a specific pair."""
        signals = []
        
        try:
            symbol1, symbol2 = pair
            
            # Check if both symbols have current data
            if symbol1 not in market_data.data or symbol2 not in market_data.data:
                return signals
            
            price1 = market_data.data[symbol1].get('close')
            price2 = market_data.data[symbol2].get('close')
            
            if not price1 or not price2:
                return signals
            
            # Calculate current spread
            hedge_ratio = self.pair_coefficients[pair]
            current_spread = price1 - hedge_ratio * price2
            
            # Update spread history
            self.pair_spreads[pair].append(current_spread)
            if len(self.pair_spreads[pair]) > self.lookback_period:
                self.pair_spreads[pair].pop(0)
            
            # Calculate current z-score
            recent_spreads = np.array(self.pair_spreads[pair][-min(30, len(self.pair_spreads[pair])):])
            current_z_score = self._calculate_z_scores(recent_spreads)[-1]
            
            # Update z-score history
            self.pair_z_scores[pair].append(current_z_score)
            if len(self.pair_z_scores[pair]) > self.lookback_period:
                self.pair_z_scores[pair].pop(0)
            
            # Get current positions
            position1 = self.get_position_size(symbol1, portfolio)
            position2 = self.get_position_size(symbol2, portfolio)
            
            # Check for entry signals
            if pair not in self.active_pairs:
                entry_signals = self._check_pair_entry(pair, current_z_score, market_data.timestamp)
                if entry_signals:
                    signals.extend(entry_signals)
                    self.active_pairs[pair] = {
                        'entry_z_score': current_z_score,
                        'entry_time': market_data.timestamp,
                        'entry_spread': current_spread
                    }
            else:
                # Check for exit signals
                exit_signals = self._check_pair_exit(pair, current_z_score, market_data.timestamp)
                if exit_signals:
                    signals.extend(exit_signals)
                    del self.active_pairs[pair]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for pair {pair}: {e}")
            return []
    
    def _check_pair_entry(self, pair: Tuple[str, str], z_score: float, 
                        timestamp: datetime) -> List[SignalEvent]:
        """Check for pair entry signals."""
        signals = []
        symbol1, symbol2 = pair
        
        try:
            # Entry conditions based on z-score
            if z_score > self.entry_threshold:
                # Spread is high: short symbol1, long symbol2
                signals.append(SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol1,
                    signal_type=OrderSide.SELL,
                    strength=min(1.0, abs(z_score) / self.entry_threshold),
                    target_percent=self.position_size,
                    metadata={
                        'pair_trade': True,
                        'pair_symbol': symbol2,
                        'z_score': z_score,
                        'trade_type': 'pairs_short_long'
                    }
                ))
                
                signals.append(SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol2,
                    signal_type=OrderSide.BUY,
                    strength=min(1.0, abs(z_score) / self.entry_threshold),
                    target_percent=self.position_size,
                    metadata={
                        'pair_trade': True,
                        'pair_symbol': symbol1,
                        'z_score': z_score,
                        'trade_type': 'pairs_short_long'
                    }
                ))
                
            elif z_score < -self.entry_threshold:
                # Spread is low: long symbol1, short symbol2
                signals.append(SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol1,
                    signal_type=OrderSide.BUY,
                    strength=min(1.0, abs(z_score) / self.entry_threshold),
                    target_percent=self.position_size,
                    metadata={
                        'pair_trade': True,
                        'pair_symbol': symbol2,
                        'z_score': z_score,
                        'trade_type': 'pairs_long_short'
                    }
                ))
                
                signals.append(SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol2,
                    signal_type=OrderSide.SELL,
                    strength=min(1.0, abs(z_score) / self.entry_threshold),
                    target_percent=self.position_size,
                    metadata={
                        'pair_trade': True,
                        'pair_symbol': symbol1,
                        'z_score': z_score,
                        'trade_type': 'pairs_long_short'
                    }
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error checking pair entry for {pair}: {e}")
            return []
    
    def _check_pair_exit(self, pair: Tuple[str, str], z_score: float, 
                       timestamp: datetime) -> List[SignalEvent]:
        """Check for pair exit signals."""
        signals = []
        symbol1, symbol2 = pair
        
        try:
            pair_info = self.active_pairs[pair]
            entry_z_score = pair_info['entry_z_score']
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # Normal exit: z-score reverts to near zero
            if abs(z_score) <= self.exit_threshold:
                should_exit = True
                exit_reason = "mean_reversion"
            
            # Stop loss: z-score moves further against position
            elif abs(z_score) >= self.stop_loss_threshold:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Time-based exit (optional)
            days_in_trade = (timestamp - pair_info['entry_time']).days
            if days_in_trade >= 30:  # Maximum 30 days in trade
                should_exit = True
                exit_reason = "time_exit"
            
            if should_exit:
                # Exit both positions
                signals.append(SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol1,
                    signal_type=OrderSide.SELL if entry_z_score < 0 else OrderSide.BUY,
                    strength=1.0,
                    target_percent=0.0,
                    metadata={
                        'pair_trade': True,
                        'pair_symbol': symbol2,
                        'exit_reason': exit_reason,
                        'entry_z_score': entry_z_score,
                        'exit_z_score': z_score
                    }
                ))
                
                signals.append(SignalEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    symbol=symbol2,
                    signal_type=OrderSide.SELL if entry_z_score > 0 else OrderSide.BUY,
                    strength=1.0,
                    target_percent=0.0,
                    metadata={
                        'pair_trade': True,
                        'pair_symbol': symbol1,
                        'exit_reason': exit_reason,
                        'entry_z_score': entry_z_score,
                        'exit_z_score': z_score
                    }
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error checking pair exit for {pair}: {e}")
            return []
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for analysis."""
        state = super().get_strategy_state()
        
        # Add pairs-specific state
        state.update({
            'active_pairs': len(self.pairs),
            'pairs_list': [f"{p[0]}-{p[1]}" for p in self.pairs],
            'current_positions': len(self.active_pairs),
            'active_pair_details': {
                f"{p[0]}-{p[1]}": {
                    'current_z_score': self.pair_z_scores[p][-1] if self.pair_z_scores[p] else 0,
                    'entry_z_score': self.active_pairs[p]['entry_z_score'],
                    'days_in_trade': (datetime.now() - self.active_pairs[p]['entry_time']).days
                } for p in self.active_pairs
            },
            'pair_statistics': {
                f"{p[0]}-{p[1]}": {
                    'hedge_ratio': self.pair_coefficients[p],
                    'current_spread': self.pair_spreads[p][-1] if self.pair_spreads[p] else 0,
                    'spread_volatility': np.std(self.pair_spreads[p]) if len(self.pair_spreads[p]) > 1 else 0
                } for p in self.pairs
            }
        })
        
        return state
