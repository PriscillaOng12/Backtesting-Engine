"""
Risk metrics calculation and monitoring.

This module provides comprehensive risk metrics calculation
including VaR, CVaR, drawdown analysis, and portfolio risk measures.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import scipy.stats as stats

from ..core.portfolio import Portfolio


logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics."""
    
    # Volatility metrics
    daily_volatility: float
    annualized_volatility: float
    volatility_of_volatility: float
    
    # VaR metrics
    var_95_daily: float
    var_99_daily: float
    cvar_95_daily: float
    cvar_99_daily: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    drawdown_frequency: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Tail risk metrics
    tail_ratio: float
    gain_to_pain_ratio: float
    pain_index: float
    
    # Time-based metrics
    up_capture_ratio: Optional[float] = None
    down_capture_ratio: Optional[float] = None
    
    # Calculation metadata
    calculation_date: datetime = None
    lookback_days: int = None
    confidence_level: float = None


class RiskMetricsCalculator:
    """
    Comprehensive risk metrics calculator.
    
    Calculates various risk metrics from portfolio returns
    and equity curves.
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize risk metrics calculator.
        
        Parameters
        ----------
        lookback_days : int
            Number of days to use for calculations (default 1 year)
        """
        self.lookback_days = lookback_days
        logger.info(f"Risk metrics calculator initialized with {lookback_days} day lookback")
    
    def calculate_metrics(self, portfolio: Portfolio, 
                         benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyze
        benchmark_returns : pd.Series, optional
            Benchmark returns for relative metrics
            
        Returns
        -------
        RiskMetrics
            Comprehensive risk metrics
        """
        try:
            # Get portfolio returns
            equity_curve = portfolio.get_equity_curve()
            if len(equity_curve) < 2:
                return self._empty_metrics()
            
            returns = self._calculate_returns(equity_curve)
            if len(returns) < 10:
                return self._empty_metrics()
            
            # Use only recent data if specified
            if len(returns) > self.lookback_days:
                returns = returns[-self.lookback_days:]
            
            # Calculate metrics
            metrics = RiskMetrics(
                # Volatility metrics
                daily_volatility=self._calculate_daily_volatility(returns),
                annualized_volatility=self._calculate_annualized_volatility(returns),
                volatility_of_volatility=self._calculate_vol_of_vol(returns),
                
                # VaR metrics
                var_95_daily=self._calculate_var(returns, 0.05),
                var_99_daily=self._calculate_var(returns, 0.01),
                cvar_95_daily=self._calculate_cvar(returns, 0.05),
                cvar_99_daily=self._calculate_cvar(returns, 0.01),
                
                # Drawdown metrics
                max_drawdown=self._calculate_max_drawdown(equity_curve),
                max_drawdown_duration=self._calculate_max_dd_duration(equity_curve),
                avg_drawdown=self._calculate_avg_drawdown(equity_curve),
                drawdown_frequency=self._calculate_dd_frequency(equity_curve),
                
                # Distribution metrics
                skewness=float(stats.skew(returns)),
                kurtosis=float(stats.kurtosis(returns)),
                jarque_bera_stat=float(stats.jarque_bera(returns)[0]),
                jarque_bera_pvalue=float(stats.jarque_bera(returns)[1]),
                
                # Tail risk metrics
                tail_ratio=self._calculate_tail_ratio(returns),
                gain_to_pain_ratio=self._calculate_gain_to_pain(returns),
                pain_index=self._calculate_pain_index(equity_curve),
                
                # Metadata
                calculation_date=datetime.now(),
                lookback_days=len(returns),
                confidence_level=0.95
            )
            
            # Add benchmark-relative metrics if provided
            if benchmark_returns is not None:
                metrics.up_capture_ratio = self._calculate_up_capture(returns, benchmark_returns)
                metrics.down_capture_ratio = self._calculate_down_capture(returns, benchmark_returns)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._empty_metrics()
    
    def calculate_rolling_metrics(self, portfolio: Portfolio, 
                                window_days: int = 60) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyze
        window_days : int
            Rolling window size in days
            
        Returns
        -------
        pd.DataFrame
            Rolling risk metrics
        """
        try:
            equity_curve = portfolio.get_equity_curve()
            if len(equity_curve) < window_days:
                return pd.DataFrame()
            
            returns = self._calculate_returns(equity_curve)
            
            # Calculate rolling metrics
            rolling_data = []
            
            for i in range(window_days, len(returns)):
                window_returns = returns[i-window_days:i]
                window_equity = equity_curve[i-window_days:i+1]
                
                metrics = {
                    'date': i,  # Would be actual date in real implementation
                    'volatility': self._calculate_annualized_volatility(window_returns),
                    'var_95': self._calculate_var(window_returns, 0.05),
                    'max_drawdown': self._calculate_max_drawdown(window_equity),
                    'skewness': float(stats.skew(window_returns)),
                    'kurtosis': float(stats.kurtosis(window_returns))
                }
                rolling_data.append(metrics)
            
            return pd.DataFrame(rolling_data)
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()
    
    def calculate_stress_scenarios(self, portfolio: Portfolio,
                                 scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate portfolio performance under stress scenarios.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to stress test
        scenarios : dict
            Stress scenarios with asset shocks
            
        Returns
        -------
        dict
            Stress test results
        """
        results = {}
        
        try:
            for scenario_name, asset_shocks in scenarios.items():
                # Apply shocks to current positions
                stressed_value = Decimal('0')
                
                for symbol, position in portfolio.positions.items():
                    current_price = portfolio.get_current_price(symbol)
                    if current_price and symbol in asset_shocks:
                        shock = asset_shocks[symbol]
                        stressed_price = current_price * (1 + shock)
                        stressed_value += position.quantity * stressed_price
                    elif current_price:
                        stressed_value += position.quantity * current_price
                
                # Add cash
                stressed_value += portfolio.cash
                
                # Calculate impact
                current_value = portfolio.calculate_total_equity()
                if current_value > 0:
                    impact = float((stressed_value - current_value) / current_value)
                else:
                    impact = 0.0
                
                results[scenario_name] = {
                    'portfolio_impact': impact,
                    'value_change': float(stressed_value - current_value),
                    'stressed_value': float(stressed_value)
                }
                
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
        
        return results
    
    def _calculate_returns(self, equity_curve: List[float]) -> np.ndarray:
        """Calculate returns from equity curve."""
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        return returns[np.isfinite(returns)]  # Remove any NaN or inf values
    
    def _calculate_daily_volatility(self, returns: np.ndarray) -> float:
        """Calculate daily volatility."""
        return float(np.std(returns)) if len(returns) > 1 else 0.0
    
    def _calculate_annualized_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        daily_vol = self._calculate_daily_volatility(returns)
        return daily_vol * np.sqrt(252)
    
    def _calculate_vol_of_vol(self, returns: np.ndarray, window: int = 20) -> float:
        """Calculate volatility of volatility."""
        if len(returns) < window * 2:
            return 0.0
        
        # Calculate rolling volatility
        rolling_vols = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            rolling_vols.append(np.std(window_returns))
        
        return float(np.std(rolling_vols)) if rolling_vols else 0.0
    
    def _calculate_var(self, returns: np.ndarray, alpha: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, alpha * 100))
    
    def _calculate_cvar(self, returns: np.ndarray, alpha: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, alpha)
        tail_returns = returns[returns <= var]
        
        return float(np.mean(tail_returns)) if len(tail_returns) > 0 else var
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        
        return float(np.min(drawdowns))
    
    def _calculate_max_dd_duration(self, equity_curve: List[float]) -> int:
        """Calculate maximum drawdown duration in days."""
        if len(equity_curve) < 2:
            return 0
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        
        # Find drawdown periods
        in_drawdown = equity_array < running_max
        
        # Calculate consecutive drawdown periods
        dd_durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    dd_durations.append(current_duration)
                    current_duration = 0
        
        # Don't forget the last period if it ends in drawdown
        if current_duration > 0:
            dd_durations.append(current_duration)
        
        return max(dd_durations) if dd_durations else 0
    
    def _calculate_avg_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate average drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        
        # Only consider negative drawdowns
        negative_drawdowns = drawdowns[drawdowns < 0]
        
        return float(np.mean(negative_drawdowns)) if len(negative_drawdowns) > 0 else 0.0
    
    def _calculate_dd_frequency(self, equity_curve: List[float]) -> float:
        """Calculate drawdown frequency (fraction of time in drawdown)."""
        if len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        in_drawdown = equity_array < running_max
        
        return float(np.mean(in_drawdown))
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        if len(returns) == 0:
            return 0.0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        return float(p95 / abs(p5)) if p5 != 0 else 0.0
    
    def _calculate_gain_to_pain(self, returns: np.ndarray) -> float:
        """Calculate gain to pain ratio."""
        if len(returns) == 0:
            return 0.0
        
        total_gain = np.sum(returns[returns > 0])
        total_pain = abs(np.sum(returns[returns < 0]))
        
        return float(total_gain / total_pain) if total_pain != 0 else 0.0
    
    def _calculate_pain_index(self, equity_curve: List[float]) -> float:
        """Calculate pain index (average squared drawdown)."""
        if len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        
        # Square the drawdowns and take average
        squared_drawdowns = drawdowns ** 2
        
        return float(np.mean(squared_drawdowns))
    
    def _calculate_up_capture(self, returns: np.ndarray, 
                            benchmark_returns: np.ndarray) -> float:
        """Calculate up capture ratio."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        # Calculate up capture
        up_market = benchmark_returns > 0
        if not np.any(up_market):
            return 0.0
        
        portfolio_up = np.mean(returns[up_market])
        benchmark_up = np.mean(benchmark_returns[up_market])
        
        return float(portfolio_up / benchmark_up) if benchmark_up != 0 else 0.0
    
    def _calculate_down_capture(self, returns: np.ndarray, 
                              benchmark_returns: np.ndarray) -> float:
        """Calculate down capture ratio."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        # Calculate down capture
        down_market = benchmark_returns < 0
        if not np.any(down_market):
            return 0.0
        
        portfolio_down = np.mean(returns[down_market])
        benchmark_down = np.mean(benchmark_returns[down_market])
        
        return float(portfolio_down / benchmark_down) if benchmark_down != 0 else 0.0
    
    def _empty_metrics(self) -> RiskMetrics:
        """Return empty risk metrics."""
        return RiskMetrics(
            daily_volatility=0.0,
            annualized_volatility=0.0,
            volatility_of_volatility=0.0,
            var_95_daily=0.0,
            var_99_daily=0.0,
            cvar_95_daily=0.0,
            cvar_99_daily=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            avg_drawdown=0.0,
            drawdown_frequency=0.0,
            skewness=0.0,
            kurtosis=0.0,
            jarque_bera_stat=0.0,
            jarque_bera_pvalue=1.0,
            tail_ratio=0.0,
            gain_to_pain_ratio=0.0,
            pain_index=0.0,
            calculation_date=datetime.now(),
            lookback_days=0,
            confidence_level=0.95
        )
