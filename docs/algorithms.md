# Mathematical Models & Algorithms Documentation

## Overview

This document details the mathematical rigor and algorithmic sophistication underlying the backtesting engine, demonstrating the quantitative depth expected by top-tier trading firms and the computational complexity understanding valued by tech companies.

## Financial Mathematics Implementation

### 1. Risk-Adjusted Performance Metrics

#### Sharpe Ratio with Statistical Significance

The traditional Sharpe ratio calculation often ignores statistical significance. Our implementation includes proper hypothesis testing:

```python
def calculate_sharpe_ratio_with_significance(
    returns: np.ndarray, 
    risk_free_rate: float = 0.02,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate Sharpe ratio with statistical significance testing and confidence intervals.
    
    Mathematical Foundation:
    - Sharpe = √252 * (μ_excess / σ_excess)
    - t-statistic = Sharpe * √n for significance testing
    - Confidence interval using Johnson distribution approximation
    """
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    # Basic Sharpe calculation
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    sharpe = np.sqrt(252) * mean_excess / std_excess
    
    # Statistical significance test
    n = len(returns)
    t_statistic = sharpe * np.sqrt(n)
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), n - 1))
    
    # Confidence interval using Jobson-Korkie method
    alpha = 1 - confidence_level
    gamma_3 = stats.skew(excess_returns)
    gamma_4 = stats.kurtosis(excess_returns, fisher=False)
    
    # Variance of Sharpe ratio estimator
    sharpe_variance = (1 + 0.5 * sharpe**2 - gamma_3 * sharpe + 
                      (gamma_4 - 3)/4 * sharpe**2) / n
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    margin_error = z_alpha * np.sqrt(sharpe_variance)
    
    return {
        'sharpe_ratio': sharpe,
        'p_value': p_value,
        'is_significant': p_value < (1 - confidence_level),
        'confidence_interval': (sharpe - margin_error, sharpe + margin_error),
        't_statistic': t_statistic,
        'sample_size': n
    }
```

#### Maximum Drawdown with Recovery Analysis

```python
def calculate_detailed_drawdown_analysis(equity_curve: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive drawdown analysis including:
    - Maximum drawdown and duration
    - Average drawdown statistics  
    - Recovery time analysis
    - Drawdown clustering analysis
    """
    # Calculate running maximum (peak values)
    peaks = equity_curve.expanding().max()
    
    # Calculate drawdown series
    drawdowns = (equity_curve - peaks) / peaks
    
    # Find drawdown periods
    in_drawdown = drawdowns < 0
    drawdown_periods = []
    
    start_idx = None
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and start_idx is None:
            start_idx = i
        elif not is_dd and start_idx is not None:
            drawdown_periods.append((start_idx, i-1))
            start_idx = None
    
    # Handle case where drawdown continues to end
    if start_idx is not None:
        drawdown_periods.append((start_idx, len(drawdowns)-1))
    
    # Analyze each drawdown period
    drawdown_stats = []
    for start, end in drawdown_periods:
        period_drawdowns = drawdowns.iloc[start:end+1]
        max_dd = period_drawdowns.min()
        duration = end - start + 1
        
        # Find recovery point (if any)
        recovery_idx = None
        if end < len(equity_curve) - 1:
            for i in range(end + 1, len(equity_curve)):
                if equity_curve.iloc[i] >= peaks.iloc[start]:
                    recovery_idx = i
                    break
        
        recovery_time = (recovery_idx - end) if recovery_idx else None
        
        drawdown_stats.append({
            'start_date': equity_curve.index[start],
            'end_date': equity_curve.index[end],
            'max_drawdown': max_dd,
            'duration_days': duration,
            'recovery_time_days': recovery_time,
            'peak_value': peaks.iloc[start],
            'trough_value': equity_curve.iloc[start:end+1].min()
        })
    
    return {
        'max_drawdown': drawdowns.min(),
        'max_drawdown_duration': max(dd['duration_days'] for dd in drawdown_stats) if drawdown_stats else 0,
        'average_drawdown': drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0,
        'drawdown_periods': drawdown_stats,
        'num_drawdown_periods': len(drawdown_periods),
        'time_in_drawdown_pct': (drawdowns < 0).sum() / len(drawdowns) * 100,
        'max_recovery_time': max((dd['recovery_time_days'] for dd in drawdown_stats 
                                if dd['recovery_time_days'] is not None), default=None)
    }
```

### 2. Value at Risk (VaR) Implementation

#### Historical Simulation VaR

```python
def calculate_historical_var(
    returns: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99, 0.995],
    window_size: int = 252
) -> Dict[str, float]:
    """
    Calculate Value at Risk using historical simulation method.
    
    Advantages over parametric VaR:
    - No distributional assumptions
    - Captures tail risk and skewness
    - Includes actual historical extreme events
    """
    var_results = {}
    
    for confidence_level in confidence_levels:
        alpha = 1 - confidence_level
        
        # Use rolling window for dynamic VaR
        rolling_var = []
        
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            var_percentile = np.percentile(window_returns, alpha * 100)
            rolling_var.append(var_percentile)
        
        # Current VaR (latest window)
        latest_window = returns[-window_size:]
        current_var = np.percentile(latest_window, alpha * 100)
        
        var_results[f'VaR_{int(confidence_level*100)}'] = current_var
        var_results[f'VaR_{int(confidence_level*100)}_rolling'] = np.array(rolling_var)
    
    return var_results

def calculate_conditional_var(
    returns: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    CVaR = E[R | R <= VaR]
    """
    cvar_results = {}
    
    for confidence_level in confidence_levels:
        alpha = 1 - confidence_level
        var_threshold = np.percentile(returns, alpha * 100)
        
        # Calculate expected value of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
        
        cvar_results[f'CVaR_{int(confidence_level*100)}'] = cvar
    
    return cvar_results
```

### 3. Dynamic Volatility Modeling

#### GARCH(1,1) Implementation

```python
class GARCHVolatilityModel:
    """
    GARCH(1,1) volatility estimation for dynamic risk management.
    
    Model: σ²t = ω + α*ε²t-1 + β*σ²t-1
    
    Where:
    - σ²t: conditional variance at time t
    - ω: long-term variance component
    - α: reaction coefficient (innovation impact)
    - β: persistence coefficient (past volatility impact)
    """
    
    def __init__(self):
        self.omega = None
        self.alpha = None 
        self.beta = None
        self.fitted = False
        
    def fit(self, returns: pd.Series) -> Dict[str, float]:
        """Fit GARCH(1,1) model using maximum likelihood estimation"""
        
        def garch_likelihood(params: np.ndarray) -> float:
            """Negative log-likelihood function for optimization"""
            omega, alpha, beta = params
            
            # Parameter constraints
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1:
                return 1e6
            
            n = len(returns)
            sigma2 = np.zeros(n)
            
            # Initial variance (unconditional variance)
            sigma2[0] = np.var(returns)
            
            # GARCH recursion
            for t in range(1, n):
                sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
            
            # Log-likelihood calculation
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * sigma2) + returns**2 / sigma2
            )
            
            return -log_likelihood  # Minimize negative log-likelihood
        
        # Initial parameter guess
        initial_params = [0.01, 0.1, 0.8]
        
        # Optimization with constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # omega > 0
            {'type': 'ineq', 'fun': lambda x: x[1]},  # alpha > 0
            {'type': 'ineq', 'fun': lambda x: x[2]},  # beta > 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2]}  # alpha + beta < 1
        ]
        
        result = minimize(
            garch_likelihood,
            initial_params,
            method='SLSQP',
            constraints=constraints
        )
        
        if result.success:
            self.omega, self.alpha, self.beta = result.x
            self.fitted = True
            
            return {
                'omega': self.omega,
                'alpha': self.alpha,
                'beta': self.beta,
                'log_likelihood': -result.fun,
                'persistence': self.alpha + self.beta
            }
        else:
            raise RuntimeError("GARCH model fitting failed")
    
    def forecast_volatility(
        self, 
        returns: pd.Series, 
        horizon: int = 1
    ) -> np.ndarray:
        """Forecast volatility for given horizon"""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Calculate current conditional variance
        n = len(returns)
        current_sigma2 = (self.omega + 
                         self.alpha * returns.iloc[-1]**2 + 
                         self.beta * self._calculate_last_variance(returns))
        
        # Multi-step ahead forecasting
        forecasts = np.zeros(horizon)
        persistence = self.alpha + self.beta
        
        for h in range(horizon):
            if h == 0:
                forecasts[h] = np.sqrt(current_sigma2)
            else:
                # Forecast converges to long-term volatility
                long_term_var = self.omega / (1 - persistence)
                forecasts[h] = np.sqrt(
                    long_term_var + 
                    (current_sigma2 - long_term_var) * persistence**h
                )
        
        return forecasts
    
    def _calculate_last_variance(self, returns: pd.Series) -> float:
        """Calculate the last conditional variance for forecasting"""
        # Simplified: use sample variance of recent window
        return np.var(returns.tail(30))
```

### 4. Market Microstructure Models

#### Market Impact and Slippage

```python
class AlmgrenChrissImpactModel:
    """
    Implementation of Almgren-Chriss optimal execution model.
    
    Market impact = σ * f(trading_rate) * g(market_conditions)
    
    Where:
    - σ: volatility
    - f(): impact function (typically square-root)
    - g(): market condition adjustment
    """
    
    def __init__(
        self,
        temporary_impact_coeff: float = 0.5,
        permanent_impact_coeff: float = 0.1,
        volatility_multiplier: float = 1.0
    ):
        self.temp_coeff = temporary_impact_coeff
        self.perm_coeff = permanent_impact_coeff
        self.vol_mult = volatility_multiplier
    
    def calculate_market_impact(
        self,
        order_size: int,
        average_daily_volume: float,
        volatility: float,
        spread: float,
        participation_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate both temporary and permanent market impact.
        
        Based on: "Optimal execution of portfolio transactions"
        by Almgren & Chriss (2001)
        """
        
        # Participation rate (order size / daily volume)
        participation = order_size / average_daily_volume
        
        # Temporary impact (reverts after execution)
        temp_impact = (
            self.temp_coeff * 
            spread * 
            np.sqrt(participation / participation_rate) *
            self.vol_mult * volatility
        )
        
        # Permanent impact (persists)
        perm_impact = (
            self.perm_coeff *
            volatility *
            np.sqrt(participation) *
            self.vol_mult
        )
        
        # Total expected slippage
        total_impact = temp_impact + perm_impact
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'total_impact': total_impact,
            'participation_rate': participation,
            'impact_basis_points': total_impact * 10000  # Convert to bps
        }

class LiquidityAdjustedSlippage:
    """
    Dynamic slippage model that adjusts based on market liquidity conditions.
    """
    
    def __init__(self):
        self.liquidity_indicators = {}
        
    def update_liquidity_metrics(
        self,
        symbol: str,
        bid_ask_spread: float,
        market_depth: float,
        recent_volume_profile: np.ndarray
    ):
        """Update liquidity indicators for more accurate slippage estimation"""
        
        # Volume-weighted average spread
        vwap_spread = np.average(
            bid_ask_spread,
            weights=recent_volume_profile
        )
        
        # Market depth ratio (depth relative to average)
        avg_depth = np.mean(market_depth)
        depth_ratio = market_depth / avg_depth if avg_depth > 0 else 1.0
        
        # Volume consistency (lower variance = more predictable)
        volume_cv = np.std(recent_volume_profile) / np.mean(recent_volume_profile)
        
        self.liquidity_indicators[symbol] = {
            'vwap_spread': vwap_spread,
            'depth_ratio': depth_ratio,
            'volume_cv': volume_cv,
            'liquidity_score': self._calculate_liquidity_score(
                vwap_spread, depth_ratio, volume_cv
            )
        }
    
    def _calculate_liquidity_score(
        self,
        spread: float,
        depth_ratio: float,
        volume_cv: float
    ) -> float:
        """Composite liquidity score (higher = more liquid)"""
        # Normalize components (lower spread = higher liquidity)
        spread_component = 1 / (1 + spread * 100)  # Convert to bps
        depth_component = min(depth_ratio, 2.0) / 2.0  # Cap at 2x average
        consistency_component = 1 / (1 + volume_cv)
        
        # Weighted average
        liquidity_score = (
            0.4 * spread_component +
            0.35 * depth_component +
            0.25 * consistency_component
        )
        
        return liquidity_score
    
    def estimate_slippage(
        self,
        symbol: str,
        order_size: int,
        market_price: float,
        urgency_factor: float = 1.0
    ) -> float:
        """Estimate slippage based on current liquidity conditions"""
        
        if symbol not in self.liquidity_indicators:
            # Fallback to simple percentage slippage
            return market_price * 0.001 * urgency_factor
        
        indicators = self.liquidity_indicators[symbol]
        
        # Base slippage from spread
        base_slippage = indicators['vwap_spread'] / 2
        
        # Adjust for market depth
        depth_adjustment = 1 / indicators['depth_ratio']
        
        # Adjust for order size impact
        size_impact = np.sqrt(order_size / 1000)  # Assume 1000 share baseline
        
        # Adjust for market volatility
        volatility_adjustment = 1 + (1 - indicators['liquidity_score'])
        
        # Combined slippage estimate
        estimated_slippage = (
            base_slippage *
            depth_adjustment *
            size_impact *
            volatility_adjustment *
            urgency_factor
        )
        
        return estimated_slippage * market_price
```

### 5. Statistical Significance and Multiple Testing

#### Robust Strategy Evaluation

```python
class StrategyStatisticalTesting:
    """
    Comprehensive statistical testing framework for strategy evaluation.
    
    Addresses common pitfalls:
    - Multiple hypothesis testing
    - Data snooping bias
    - Overfitting detection
    - Statistical significance of performance metrics
    """
    
    def __init__(self):
        self.test_results = {}
        
    def test_strategy_significance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        alpha: float = 0.05,
        method: str = 'bonferroni'
    ) -> Dict[str, Any]:
        """
        Test statistical significance of strategy performance vs benchmark.
        """
        
        # Excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        # Basic statistics
        n_obs = len(excess_returns)
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        # T-test for mean excess return
        t_stat = mean_excess / (std_excess / np.sqrt(n_obs))
        p_value_ttest = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 1))
        
        # Wilcoxon signed-rank test (non-parametric)
        wilcoxon_stat, p_value_wilcoxon = stats.wilcoxon(excess_returns)
        
        # Bootstrap confidence interval for mean excess return
        bootstrap_means = []
        for _ in range(1000):
            boot_sample = np.random.choice(excess_returns, size=n_obs, replace=True)
            bootstrap_means.append(np.mean(boot_sample))
        
        ci_lower = np.percentile(bootstrap_means, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        # Multiple testing correction
        p_values = [p_value_ttest, p_value_wilcoxon]
        
        if method == 'bonferroni':
            corrected_alpha = alpha / len(p_values)
            is_significant = all(p < corrected_alpha for p in p_values)
        elif method == 'holm':
            # Holm-Bonferroni step-down method
            sorted_p = sorted(p_values)
            is_significant = True
            for i, p in enumerate(sorted_p):
                if p > alpha / (len(p_values) - i):
                    is_significant = False
                    break
        
        return {
            'mean_excess_return': mean_excess,
            'excess_return_volatility': std_excess,
            't_statistic': t_stat,
            'p_value_ttest': p_value_ttest,
            'p_value_wilcoxon': p_value_wilcoxon,
            'is_statistically_significant': is_significant,
            'confidence_interval': (ci_lower, ci_upper),
            'multiple_testing_method': method,
            'corrected_alpha': corrected_alpha if method == 'bonferroni' else alpha
        }
    
    def detect_overfitting(
        self,
        in_sample_results: Dict[str, float],
        out_sample_results: Dict[str, float],
        min_out_sample_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect potential overfitting using multiple approaches.
        """
        
        # Performance degradation analysis
        metrics_comparison = {}
        degradation_scores = []
        
        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown']:
            if metric in in_sample_results and metric in out_sample_results:
                is_value = in_sample_results[metric]
                oos_value = out_sample_results[metric]
                
                # Calculate relative degradation
                if metric == 'max_drawdown':
                    # For drawdown, improvement is reduction (negative degradation)
                    degradation = (oos_value - is_value) / abs(is_value)
                else:
                    # For returns/sharpe, degradation is reduction
                    degradation = (is_value - oos_value) / abs(is_value)
                
                metrics_comparison[metric] = {
                    'in_sample': is_value,
                    'out_sample': oos_value,
                    'degradation_pct': degradation * 100
                }
                degradation_scores.append(degradation)
        
        # Overall degradation score
        avg_degradation = np.mean(degradation_scores) if degradation_scores else 0
        
        # Overfitting indicators
        high_degradation = avg_degradation > 0.3  # >30% performance loss
        inconsistent_ranking = self._check_ranking_consistency(
            in_sample_results, out_sample_results
        )
        
        # White's Reality Check (simplified)
        reality_check_pvalue = self._whites_reality_check(
            in_sample_results, out_sample_results
        )
        
        overfitting_risk = "HIGH" if (
            high_degradation or inconsistent_ranking or reality_check_pvalue < 0.05
        ) else "LOW"
        
        return {
            'metrics_comparison': metrics_comparison,
            'average_degradation_pct': avg_degradation * 100,
            'overfitting_risk': overfitting_risk,
            'reality_check_pvalue': reality_check_pvalue,
            'recommendations': self._generate_overfitting_recommendations(
                avg_degradation, reality_check_pvalue
            )
        }
    
    def _whites_reality_check(
        self,
        in_sample: Dict[str, float],
        out_sample: Dict[str, float]
    ) -> float:
        """
        Simplified implementation of White's Reality Check.
        Tests if best in-sample performance is due to luck.
        """
        # This is a simplified version - full implementation would require
        # multiple strategy comparisons and bootstrap simulation
        
        if 'sharpe_ratio' in in_sample and 'sharpe_ratio' in out_sample:
            is_sharpe = in_sample['sharpe_ratio']
            oos_sharpe = out_sample['sharpe_ratio']
            
            # Simulate p-value based on degradation magnitude
            degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
            
            # Higher degradation suggests higher probability of luck
            simulated_pvalue = min(0.5, degradation * 2) if degradation > 0 else 0.95
            
            return simulated_pvalue
        
        return 0.5  # Neutral if insufficient data
```
