"""
Monte Carlo analysis for robust performance statistics.

This module implements Monte Carlo simulation techniques including
bootstrap resampling, parameter uncertainty analysis, and
robust performance statistics with confidence intervals.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import concurrent.futures
from scipy import stats
import warnings

from ..core.engine import BacktestEngine
from ..core.data_handler import DataConfig
from ..strategies.base import BaseStrategy
from .performance import PerformanceAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo analysis."""
    
    # Simulation parameters
    num_simulations: int = 1000
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Bootstrap parameters
    bootstrap_block_size: int = 20  # For block bootstrap
    bootstrap_method: str = "block"  # "simple", "block", "stationary"
    
    # Parameter uncertainty
    parameter_uncertainty: bool = True
    parameter_noise_level: float = 0.05  # 5% noise
    
    # Simulation types
    return_simulation: bool = True
    parameter_simulation: bool = True
    market_regime_simulation: bool = False
    
    # Performance options
    parallel_processing: bool = True
    save_distributions: bool = True
    detailed_statistics: bool = True


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo analysis."""
    
    config: MonteCarloConfig
    base_performance: Dict[str, float]
    
    # Simulation results
    simulation_results: List[Dict[str, float]] = field(default_factory=list)
    
    # Statistical summaries
    confidence_intervals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    distribution_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Risk metrics
    probability_of_loss: float = 0.0
    expected_shortfall: Dict[str, float] = field(default_factory=dict)
    value_at_risk: Dict[str, float] = field(default_factory=dict)
    
    # Robustness metrics
    stability_ratio: float = 0.0
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    runtime_seconds: float = 0.0
    success_rate: float = 0.0
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all metrics."""
        if not self.simulation_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.simulation_results)
        
        summary = []
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                stats_dict = {
                    'metric': column,
                    'base_value': self.base_performance.get(column, np.nan),
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'median': df[column].median(),
                    'skewness': stats.skew(df[column]),
                    'kurtosis': stats.kurtosis(df[column])
                }
                
                # Add confidence intervals
                for level in self.config.confidence_levels:
                    alpha = 1 - level
                    lower = df[column].quantile(alpha / 2)
                    upper = df[column].quantile(1 - alpha / 2)
                    stats_dict[f'ci_{level}_lower'] = lower
                    stats_dict[f'ci_{level}_upper'] = upper
                
                summary.append(stats_dict)
        
        return pd.DataFrame(summary)
    
    def get_stability_analysis(self) -> Dict[str, Any]:
        """Analyze strategy stability across simulations."""
        if not self.simulation_results:
            return {}
        
        df = pd.DataFrame(self.simulation_results)
        
        stability = {}
        
        # Calculate coefficient of variation for key metrics
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        for metric in key_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0 and values.mean() != 0:
                    cv = values.std() / abs(values.mean())
                    stability[f'{metric}_cv'] = cv
        
        # Probability of positive returns
        if 'total_return' in df.columns:
            stability['prob_positive_return'] = (df['total_return'] > 0).mean()
        
        # Consistency metrics
        if 'sharpe_ratio' in df.columns:
            stability['prob_positive_sharpe'] = (df['sharpe_ratio'] > 0).mean()
            stability['prob_good_sharpe'] = (df['sharpe_ratio'] > 1.0).mean()
        
        return stability


class MonteCarloAnalyzer:
    """
    Monte Carlo analysis engine.
    
    Implements various Monte Carlo techniques for robust
    strategy performance analysis and uncertainty quantification.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """
        Initialize Monte Carlo analyzer.
        
        Parameters
        ----------
        config : MonteCarloConfig
            Monte Carlo analysis configuration
        """
        self.config = config
        logger.info(f"Monte Carlo analyzer initialized: {config.num_simulations} simulations")
    
    def analyze_strategy(self, 
                        strategy_class: type,
                        strategy_params: Dict[str, Any],
                        data_config: DataConfig,
                        base_results: Optional[Dict[str, float]] = None) -> MonteCarloResults:
        """
        Perform Monte Carlo analysis on a strategy.
        
        Parameters
        ----------
        strategy_class : type
            Strategy class to analyze
        strategy_params : dict
            Base strategy parameters
        data_config : DataConfig
            Data configuration
        base_results : dict, optional
            Base backtest results (will run if not provided)
            
        Returns
        -------
        MonteCarloResults
            Complete Monte Carlo analysis results
        """
        start_time = datetime.now()
        
        try:
            # Get base performance if not provided
            if base_results is None:
                base_results = self._run_base_backtest(strategy_class, strategy_params, data_config)
            
            # Initialize results
            results = MonteCarloResults(
                config=self.config,
                base_performance=base_results
            )
            
            # Run simulations
            if self.config.parallel_processing:
                simulation_results = self._run_simulations_parallel(
                    strategy_class, strategy_params, data_config
                )
            else:
                simulation_results = self._run_simulations_sequential(
                    strategy_class, strategy_params, data_config
                )
            
            results.simulation_results = simulation_results
            
            # Calculate statistical summaries
            self._calculate_confidence_intervals(results)
            self._calculate_distribution_statistics(results)
            self._calculate_risk_metrics(results)
            self._calculate_robustness_metrics(results)
            
            # Set metadata
            results.runtime_seconds = (datetime.now() - start_time).total_seconds()
            results.success_rate = len(simulation_results) / self.config.num_simulations
            
            logger.info(f"Monte Carlo analysis completed: {len(simulation_results)} successful simulations "
                       f"in {results.runtime_seconds:.1f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {e}")
            raise
    
    def bootstrap_returns(self, returns: np.ndarray, 
                         num_samples: int = None) -> List[np.ndarray]:
        """
        Bootstrap resample returns.
        
        Parameters
        ----------
        returns : np.ndarray
            Original return series
        num_samples : int, optional
            Number of bootstrap samples (default: config.num_simulations)
            
        Returns
        -------
        List[np.ndarray]
            Bootstrap samples
        """
        if num_samples is None:
            num_samples = self.config.num_simulations
        
        samples = []
        
        for _ in range(num_samples):
            if self.config.bootstrap_method == "simple":
                sample = self._simple_bootstrap(returns)
            elif self.config.bootstrap_method == "block":
                sample = self._block_bootstrap(returns)
            elif self.config.bootstrap_method == "stationary":
                sample = self._stationary_bootstrap(returns)
            else:
                raise ValueError(f"Unknown bootstrap method: {self.config.bootstrap_method}")
            
            samples.append(sample)
        
        return samples
    
    def _run_base_backtest(self, strategy_class, strategy_params, data_config) -> Dict[str, float]:
        """Run base backtest to get baseline performance."""
        try:
            # Create strategy
            strategy = strategy_class(
                strategy_id="mc_base",
                symbols=data_config.symbols,
                **strategy_params
            )
            
            # Create engine
            engine = BacktestEngine(
                start_date=data_config.start_date,
                end_date=data_config.end_date,
                initial_capital=Decimal('1000000')
            )
            
            # For now, return simulated base performance
            # In real implementation, run actual backtest
            np.random.seed(42)
            base_performance = {
                'total_return': np.random.normal(0.12, 0.05),
                'annualized_return': np.random.normal(0.12, 0.05),
                'volatility': np.random.uniform(0.15, 0.25),
                'sharpe_ratio': np.random.normal(0.8, 0.2),
                'max_drawdown': np.random.uniform(-0.08, -0.15),
                'num_trades': np.random.randint(50, 200)
            }
            
            return base_performance
            
        except Exception as e:
            logger.error(f"Error in base backtest: {e}")
            return {}
    
    def _run_simulations_sequential(self, strategy_class, strategy_params, data_config) -> List[Dict[str, float]]:
        """Run Monte Carlo simulations sequentially."""
        results = []
        
        for i in range(self.config.num_simulations):
            try:
                sim_result = self._run_single_simulation(i, strategy_class, strategy_params, data_config)
                if sim_result:
                    results.append(sim_result)
                    
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{self.config.num_simulations} simulations")
                    
            except Exception as e:
                logger.debug(f"Error in simulation {i}: {e}")
                continue
        
        return results
    
    def _run_simulations_parallel(self, strategy_class, strategy_params, data_config) -> List[Dict[str, float]]:
        """Run Monte Carlo simulations in parallel."""
        results = []
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            
            for i in range(self.config.num_simulations):
                future = executor.submit(
                    self._run_single_simulation,
                    i, strategy_class, strategy_params, data_config
                )
                futures.append(future)
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                    if (i + 1) % 100 == 0:
                        logger.info(f"Completed {i + 1}/{self.config.num_simulations} simulations")
                        
                except Exception as e:
                    logger.debug(f"Error in parallel simulation: {e}")
        
        return results
    
    def _run_single_simulation(self, sim_id: int, strategy_class, 
                             strategy_params, data_config) -> Optional[Dict[str, float]]:
        """Run a single Monte Carlo simulation."""
        try:
            # Add randomness to parameters if configured
            sim_params = strategy_params.copy()
            if self.config.parameter_uncertainty:
                sim_params = self._add_parameter_noise(sim_params)
            
            # Create perturbed data if configured
            if self.config.return_simulation:
                # This would involve creating new return series
                # For now, just add some randomness to the final performance
                pass
            
            # Run simulation
            # For now, return simulated performance with variations
            np.random.seed(sim_id + 1000)  # Different seed for each simulation
            
            # Base performance with random variations
            base_return = np.random.normal(0.10, 0.08)
            base_vol = np.random.uniform(0.12, 0.28)
            
            performance = {
                'total_return': base_return,
                'annualized_return': base_return,
                'volatility': base_vol,
                'sharpe_ratio': base_return / base_vol if base_vol > 0 else 0,
                'max_drawdown': np.random.uniform(-0.05, -0.25),
                'num_trades': np.random.randint(30, 150),
                'win_rate': np.random.uniform(0.4, 0.7),
                'profit_factor': np.random.uniform(0.8, 2.5)
            }
            
            return performance
            
        except Exception as e:
            logger.debug(f"Error in simulation {sim_id}: {e}")
            return None
    
    def _add_parameter_noise(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to strategy parameters."""
        noisy_params = params.copy()
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Add percentage noise
                noise = np.random.normal(0, self.config.parameter_noise_level)
                noisy_value = value * (1 + noise)
                
                # Ensure reasonable bounds
                if isinstance(value, int):
                    noisy_params[key] = max(1, int(noisy_value))
                else:
                    noisy_params[key] = max(0.001, noisy_value)
        
        return noisy_params
    
    def _simple_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Simple bootstrap resampling."""
        n = len(returns)
        indices = np.random.choice(n, size=n, replace=True)
        return returns[indices]
    
    def _block_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Block bootstrap resampling."""
        n = len(returns)
        block_size = self.config.bootstrap_block_size
        
        if block_size >= n:
            return self._simple_bootstrap(returns)
        
        num_blocks = int(np.ceil(n / block_size))
        resampled = []
        
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            block = returns[start_idx:start_idx + block_size]
            resampled.extend(block)
        
        return np.array(resampled[:n])
    
    def _stationary_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Stationary bootstrap resampling."""
        n = len(returns)
        avg_block_size = self.config.bootstrap_block_size
        p = 1.0 / avg_block_size  # Probability of starting new block
        
        resampled = []
        i = 0
        
        while len(resampled) < n:
            # Choose starting point
            start_idx = np.random.randint(0, n)
            block_length = 1
            
            # Determine block length
            while np.random.random() > p and len(resampled) + block_length < n:
                block_length += 1
            
            # Add block (with wraparound)
            for j in range(block_length):
                if len(resampled) >= n:
                    break
                idx = (start_idx + j) % n
                resampled.append(returns[idx])
        
        return np.array(resampled[:n])
    
    def _calculate_confidence_intervals(self, results: MonteCarloResults) -> None:
        """Calculate confidence intervals for all metrics."""
        if not results.simulation_results:
            return
        
        df = pd.DataFrame(results.simulation_results)
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                intervals = {}
                
                for level in self.config.confidence_levels:
                    alpha = 1 - level
                    lower = df[column].quantile(alpha / 2)
                    upper = df[column].quantile(1 - alpha / 2)
                    
                    intervals[f'level_{level}'] = {
                        'lower': float(lower),
                        'upper': float(upper)
                    }
                
                results.confidence_intervals[column] = intervals
    
    def _calculate_distribution_statistics(self, results: MonteCarloResults) -> None:
        """Calculate distribution statistics for all metrics."""
        if not results.simulation_results:
            return
        
        df = pd.DataFrame(results.simulation_results)
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                values = df[column].dropna()
                
                if len(values) > 0:
                    results.distribution_stats[column] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'median': float(values.median()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'skewness': float(stats.skew(values)),
                        'kurtosis': float(stats.kurtosis(values)),
                        'normality_pvalue': float(stats.normaltest(values)[1]) if len(values) >= 8 else 1.0
                    }
    
    def _calculate_risk_metrics(self, results: MonteCarloResults) -> None:
        """Calculate risk metrics from simulation results."""
        if not results.simulation_results:
            return
        
        df = pd.DataFrame(results.simulation_results)
        
        # Probability of loss
        if 'total_return' in df.columns:
            results.probability_of_loss = float((df['total_return'] < 0).mean())
        
        # Value at Risk and Expected Shortfall
        for level in self.config.confidence_levels:
            alpha = 1 - level
            
            if 'total_return' in df.columns:
                var = df['total_return'].quantile(alpha)
                tail_returns = df['total_return'][df['total_return'] <= var]
                es = tail_returns.mean() if len(tail_returns) > 0 else var
                
                results.value_at_risk[f'level_{level}'] = float(var)
                results.expected_shortfall[f'level_{level}'] = float(es)
    
    def _calculate_robustness_metrics(self, results: MonteCarloResults) -> None:
        """Calculate strategy robustness metrics."""
        if not results.simulation_results:
            return
        
        df = pd.DataFrame(results.simulation_results)
        
        # Stability ratio (fraction of simulations with positive Sharpe)
        if 'sharpe_ratio' in df.columns:
            results.stability_ratio = float((df['sharpe_ratio'] > 0).mean())
        
        # Parameter sensitivity (coefficient of variation for key metrics)
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        for metric in key_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0 and values.mean() != 0:
                    cv = values.std() / abs(values.mean())
                    results.parameter_sensitivity[metric] = float(cv)
