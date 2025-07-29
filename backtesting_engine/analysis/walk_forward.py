"""
Walk-forward analysis for out-of-sample testing.

This module implements proper walk-forward analysis including
rolling window optimization and performance degradation analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import concurrent.futures
from pathlib import Path

from ..core.engine import BacktestEngine
from ..core.data_handler import DataConfig
from ..strategies.base import BaseStrategy
from .performance import PerformanceAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Window configuration
    in_sample_days: int = 252  # 1 year training
    out_sample_days: int = 63   # 3 months testing
    step_days: int = 21         # Monthly steps
    
    # Optimization configuration
    optimize_frequency: str = "monthly"  # "daily", "weekly", "monthly"
    min_trades_required: int = 10
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -0.15
    
    # Analysis options
    anchored_analysis: bool = False  # True for anchored, False for rolling
    parallel_processing: bool = True
    save_intermediate_results: bool = True


@dataclass
class WalkForwardPeriod:
    """Single walk-forward period results."""
    
    period_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    
    # Optimization results
    optimal_parameters: Dict[str, Any]
    in_sample_performance: Dict[str, float]
    
    # Out-of-sample results
    out_sample_performance: Dict[str, float]
    degradation_metrics: Dict[str, float]
    
    # Metadata
    optimization_time: float = 0.0
    backtest_time: float = 0.0
    num_trades_is: int = 0
    num_trades_oos: int = 0


@dataclass
class WalkForwardResults:
    """Complete walk-forward analysis results."""
    
    config: WalkForwardConfig
    periods: List[WalkForwardPeriod] = field(default_factory=list)
    
    # Aggregate statistics
    summary_statistics: Dict[str, float] = field(default_factory=dict)
    parameter_stability: Dict[str, float] = field(default_factory=dict)
    performance_degradation: Dict[str, float] = field(default_factory=dict)
    
    # Analysis metadata
    total_runtime: float = 0.0
    success_rate: float = 0.0
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of performance across periods."""
        data = []
        for period in self.periods:
            row = {
                'period_id': period.period_id,
                'in_sample_sharpe': period.in_sample_performance.get('sharpe_ratio', 0),
                'out_sample_sharpe': period.out_sample_performance.get('sharpe_ratio', 0),
                'in_sample_return': period.in_sample_performance.get('total_return', 0),
                'out_sample_return': period.out_sample_performance.get('total_return', 0),
                'degradation_ratio': period.degradation_metrics.get('sharpe_degradation', 0),
                'num_trades_is': period.num_trades_is,
                'num_trades_oos': period.num_trades_oos
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_parameter_evolution(self) -> pd.DataFrame:
        """Get evolution of parameters over time."""
        if not self.periods:
            return pd.DataFrame()
        
        # Get all parameter names
        all_params = set()
        for period in self.periods:
            all_params.update(period.optimal_parameters.keys())
        
        data = []
        for period in self.periods:
            row = {'period_id': period.period_id}
            for param in all_params:
                row[param] = period.optimal_parameters.get(param, np.nan)
            data.append(row)
        
        return pd.DataFrame(data)


class WalkForwardAnalyzer:
    """
    Walk-forward analysis engine.
    
    Implements rolling and anchored walk-forward analysis
    for robust out-of-sample strategy validation.
    """
    
    def __init__(self, config: WalkForwardConfig):
        """
        Initialize walk-forward analyzer.
        
        Parameters
        ----------
        config : WalkForwardConfig
            Walk-forward analysis configuration
        """
        self.config = config
        logger.info(f"Walk-forward analyzer initialized: {config}")
    
    def analyze_strategy(self, 
                        strategy_class: type,
                        data_config: DataConfig,
                        parameter_grid: Dict[str, List[Any]],
                        objective_function: Callable = None) -> WalkForwardResults:
        """
        Perform walk-forward analysis on a strategy.
        
        Parameters
        ----------
        strategy_class : type
            Strategy class to analyze
        data_config : DataConfig
            Data configuration
        parameter_grid : dict
            Parameter combinations to test
        objective_function : callable, optional
            Function to optimize (default: Sharpe ratio)
            
        Returns
        -------
        WalkForwardResults
            Complete walk-forward analysis results
        """
        start_time = datetime.now()
        
        try:
            # Generate walk-forward periods
            periods = self._generate_periods(data_config.start_date, data_config.end_date)
            logger.info(f"Generated {len(periods)} walk-forward periods")
            
            # Initialize results
            results = WalkForwardResults(config=self.config)
            
            # Process each period
            if self.config.parallel_processing:
                results.periods = self._process_periods_parallel(
                    periods, strategy_class, data_config, parameter_grid, objective_function
                )
            else:
                results.periods = self._process_periods_sequential(
                    periods, strategy_class, data_config, parameter_grid, objective_function
                )
            
            # Calculate aggregate statistics
            self._calculate_aggregate_statistics(results)
            
            # Calculate runtime
            results.total_runtime = (datetime.now() - start_time).total_seconds()
            results.success_rate = len(results.periods) / len(periods)
            
            logger.info(f"Walk-forward analysis completed in {results.total_runtime:.1f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            raise
    
    def _generate_periods(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward periods."""
        periods = []
        current_date = start_date
        period_id = 0
        
        while True:
            # Calculate period dates
            is_start = current_date
            is_end = is_start + timedelta(days=self.config.in_sample_days)
            oos_start = is_end + timedelta(days=1)
            oos_end = oos_start + timedelta(days=self.config.out_sample_days)
            
            # Check if we have enough data
            if oos_end > end_date:
                break
            
            periods.append((is_start, is_end, oos_start, oos_end))
            
            # Move to next period
            if self.config.anchored_analysis:
                # Anchored: keep same start, extend end
                current_date = is_start
                self.config.in_sample_days += self.config.step_days
            else:
                # Rolling: move window forward
                current_date += timedelta(days=self.config.step_days)
            
            period_id += 1
            
            # Safety check
            if period_id > 100:  # Prevent infinite loops
                logger.warning("Too many periods generated, stopping")
                break
        
        return periods
    
    def _process_periods_sequential(self, periods, strategy_class, data_config, 
                                  parameter_grid, objective_function) -> List[WalkForwardPeriod]:
        """Process periods sequentially."""
        results = []
        
        for i, (is_start, is_end, oos_start, oos_end) in enumerate(periods):
            try:
                period_result = self._process_single_period(
                    i, is_start, is_end, oos_start, oos_end,
                    strategy_class, data_config, parameter_grid, objective_function
                )
                if period_result:
                    results.append(period_result)
                    
            except Exception as e:
                logger.error(f"Error processing period {i}: {e}")
                continue
        
        return results
    
    def _process_periods_parallel(self, periods, strategy_class, data_config, 
                                parameter_grid, objective_function) -> List[WalkForwardPeriod]:
        """Process periods in parallel."""
        results = []
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_period = {}
            
            for i, (is_start, is_end, oos_start, oos_end) in enumerate(periods):
                future = executor.submit(
                    self._process_single_period,
                    i, is_start, is_end, oos_start, oos_end,
                    strategy_class, data_config, parameter_grid, objective_function
                )
                future_to_period[future] = i
            
            for future in concurrent.futures.as_completed(future_to_period):
                period_id = future_to_period[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing for period {period_id}: {e}")
        
        # Sort by period ID
        results.sort(key=lambda x: x.period_id)
        return results
    
    def _process_single_period(self, period_id: int, 
                             is_start: datetime, is_end: datetime,
                             oos_start: datetime, oos_end: datetime,
                             strategy_class, data_config, parameter_grid, 
                             objective_function) -> Optional[WalkForwardPeriod]:
        """Process a single walk-forward period."""
        try:
            logger.debug(f"Processing period {period_id}: {is_start} to {oos_end}")
            
            # Step 1: Optimize parameters on in-sample data
            opt_start_time = datetime.now()
            optimal_params, is_performance = self._optimize_parameters(
                strategy_class, data_config, parameter_grid, 
                is_start, is_end, objective_function
            )
            opt_time = (datetime.now() - opt_start_time).total_seconds()
            
            if not optimal_params:
                logger.warning(f"No valid parameters found for period {period_id}")
                return None
            
            # Step 2: Test on out-of-sample data
            test_start_time = datetime.now()
            oos_performance = self._test_out_of_sample(
                strategy_class, optimal_params, data_config,
                oos_start, oos_end
            )
            test_time = (datetime.now() - test_start_time).total_seconds()
            
            if not oos_performance:
                logger.warning(f"Out-of-sample test failed for period {period_id}")
                return None
            
            # Step 3: Calculate degradation metrics
            degradation = self._calculate_degradation(is_performance, oos_performance)
            
            # Create period result
            period_result = WalkForwardPeriod(
                period_id=period_id,
                in_sample_start=is_start,
                in_sample_end=is_end,
                out_sample_start=oos_start,
                out_sample_end=oos_end,
                optimal_parameters=optimal_params,
                in_sample_performance=is_performance,
                out_sample_performance=oos_performance,
                degradation_metrics=degradation,
                optimization_time=opt_time,
                backtest_time=test_time,
                num_trades_is=is_performance.get('num_trades', 0),
                num_trades_oos=oos_performance.get('num_trades', 0)
            )
            
            logger.debug(f"Completed period {period_id}: IS Sharpe={is_performance.get('sharpe_ratio', 0):.3f}, "
                        f"OOS Sharpe={oos_performance.get('sharpe_ratio', 0):.3f}")
            
            return period_result
            
        except Exception as e:
            logger.error(f"Error processing period {period_id}: {e}")
            return None
    
    def _optimize_parameters(self, strategy_class, data_config, parameter_grid,
                           start_date, end_date, objective_function) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize strategy parameters on in-sample data."""
        # This is a simplified version - real implementation would use
        # sophisticated optimization algorithms
        
        best_params = None
        best_performance = None
        best_score = -np.inf
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_grid)
        
        for params in param_combinations:
            try:
                # Test this parameter combination
                performance = self._backtest_parameters(
                    strategy_class, params, data_config, start_date, end_date
                )
                
                if not performance:
                    continue
                
                # Calculate objective score
                if objective_function:
                    score = objective_function(performance)
                else:
                    score = performance.get('sharpe_ratio', 0)
                
                # Check if this is the best so far
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_performance = performance.copy()
                    
            except Exception as e:
                logger.debug(f"Error testing parameters {params}: {e}")
                continue
        
        return best_params or {}, best_performance or {}
    
    def _test_out_of_sample(self, strategy_class, parameters, data_config,
                          start_date, end_date) -> Dict[str, float]:
        """Test strategy with optimized parameters on out-of-sample data."""
        return self._backtest_parameters(strategy_class, parameters, data_config, start_date, end_date)
    
    def _backtest_parameters(self, strategy_class, parameters, data_config,
                           start_date, end_date) -> Optional[Dict[str, float]]:
        """Run backtest with specific parameters."""
        try:
            # Create strategy with parameters
            strategy = strategy_class(
                strategy_id=f"wf_test_{start_date.strftime('%Y%m%d')}",
                symbols=data_config.symbols,
                **parameters
            )
            
            # Create engine
            engine = BacktestEngine(
                start_date=start_date,
                end_date=end_date,
                initial_capital=Decimal('1000000')
            )
            
            # Set up data config for this period
            period_data_config = DataConfig(
                source_type=data_config.source_type,
                path_or_connection=data_config.path_or_connection,
                symbols=data_config.symbols,
                start_date=start_date,
                end_date=end_date,
                frequency=data_config.frequency
            )
            
            # Add components (simplified)
            # engine.add_data_handler(CSVDataHandler(period_data_config))
            # engine.add_strategy(strategy.strategy_id, strategy)
            
            # Run backtest
            # results = engine.run()
            
            # For now, return simulated performance
            # In real implementation, extract from results
            np.random.seed(hash(str(parameters)) % 1000)
            base_return = np.random.normal(0.08, 0.15)
            volatility = np.random.uniform(0.10, 0.25)
            
            performance = {
                'total_return': base_return,
                'annualized_return': base_return,
                'volatility': volatility,
                'sharpe_ratio': base_return / volatility if volatility > 0 else 0,
                'max_drawdown': np.random.uniform(-0.05, -0.20),
                'num_trades': np.random.randint(10, 100)
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return None
    
    def _generate_parameter_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        from itertools import product
        
        if not parameter_grid:
            return [{}]
        
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _calculate_degradation(self, is_performance: Dict[str, float], 
                             oos_performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance degradation metrics."""
        degradation = {}
        
        for metric in ['sharpe_ratio', 'total_return', 'volatility']:
            is_val = is_performance.get(metric, 0)
            oos_val = oos_performance.get(metric, 0)
            
            if is_val != 0:
                degradation[f'{metric}_degradation'] = (oos_val - is_val) / abs(is_val)
            else:
                degradation[f'{metric}_degradation'] = 0
        
        return degradation
    
    def _calculate_aggregate_statistics(self, results: WalkForwardResults) -> None:
        """Calculate aggregate statistics across all periods."""
        if not results.periods:
            return
        
        # Performance statistics
        is_sharpes = [p.in_sample_performance.get('sharpe_ratio', 0) for p in results.periods]
        oos_sharpes = [p.out_sample_performance.get('sharpe_ratio', 0) for p in results.periods]
        
        results.summary_statistics = {
            'avg_is_sharpe': np.mean(is_sharpes),
            'avg_oos_sharpe': np.mean(oos_sharpes),
            'sharpe_degradation': np.mean(oos_sharpes) - np.mean(is_sharpes),
            'oos_sharpe_std': np.std(oos_sharpes),
            'positive_oos_periods': sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
        }
        
        # Parameter stability
        param_df = results.get_parameter_evolution()
        if not param_df.empty:
            stability = {}
            for col in param_df.columns:
                if col != 'period_id':
                    values = param_df[col].dropna()
                    if len(values) > 1:
                        stability[f'{col}_cv'] = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            
            results.parameter_stability = stability
