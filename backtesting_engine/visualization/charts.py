"""
Publication-quality charts and visualizations.

This module provides comprehensive charting capabilities for
backtesting results including equity curves, drawdown charts,
performance attribution, and risk analysis visualizations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..core.portfolio import Portfolio
from ..analysis.performance import BacktestResults, PerformanceMetrics
from ..analysis.walk_forward import WalkForwardResults
from ..analysis.monte_carlo import MonteCarloResults


logger = logging.getLogger(__name__)

# Set style preferences
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ChartGenerator:
    """
    Comprehensive chart generation for backtesting results.
    
    Supports both static (matplotlib) and interactive (plotly) charts
    with publication-quality output.
    """
    
    def __init__(self, style: str = "plotly", theme: str = "white"):
        """
        Initialize chart generator.
        
        Parameters
        ----------
        style : str
            Chart library to use ("plotly" or "matplotlib")
        theme : str
            Color theme ("white", "dark", "presentation")
        """
        self.style = style
        self.theme = theme
        self.fig_size = (12, 8)
        
        # Configure themes
        if style == "plotly":
            if theme == "dark":
                self.template = "plotly_dark"
            elif theme == "presentation":
                self.template = "presentation"
            else:
                self.template = "plotly_white"
        
        logger.info(f"Chart generator initialized: {style} style, {theme} theme")
    
    def create_equity_curve(self, portfolio: Portfolio, 
                          benchmark_data: Optional[pd.Series] = None,
                          title: str = "Portfolio Equity Curve") -> Union[go.Figure, plt.Figure]:
        """
        Create equity curve chart.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio with equity history
        benchmark_data : pd.Series, optional
            Benchmark comparison data
        title : str
            Chart title
            
        Returns
        -------
        Union[go.Figure, plt.Figure]
            Chart figure
        """
        try:
            equity_curve = portfolio.get_equity_curve()
            if len(equity_curve) < 2:
                logger.warning("Insufficient data for equity curve")
                return self._empty_chart(title)
            
            # Create date index (simplified for demo)
            dates = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
            
            if self.style == "plotly":
                return self._create_equity_curve_plotly(equity_curve, dates, benchmark_data, title)
            else:
                return self._create_equity_curve_matplotlib(equity_curve, dates, benchmark_data, title)
                
        except Exception as e:
            logger.error(f"Error creating equity curve: {e}")
            return self._empty_chart(title)
    
    def create_drawdown_chart(self, portfolio: Portfolio,
                            title: str = "Portfolio Drawdown") -> Union[go.Figure, plt.Figure]:
        """
        Create drawdown chart.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio with equity history
        title : str
            Chart title
            
        Returns
        -------
        Union[go.Figure, plt.Figure]
            Chart figure
        """
        try:
            equity_curve = portfolio.get_equity_curve()
            if len(equity_curve) < 2:
                return self._empty_chart(title)
            
            # Calculate drawdowns
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdowns = (equity_array - running_max) / running_max * 100
            
            dates = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
            
            if self.style == "plotly":
                return self._create_drawdown_plotly(drawdowns, dates, title)
            else:
                return self._create_drawdown_matplotlib(drawdowns, dates, title)
                
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            return self._empty_chart(title)
    
    def create_returns_distribution(self, portfolio: Portfolio,
                                  title: str = "Returns Distribution") -> Union[go.Figure, plt.Figure]:
        """
        Create returns distribution chart.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio with equity history
        title : str
            Chart title
            
        Returns
        -------
        Union[go.Figure, plt.Figure]
            Chart figure
        """
        try:
            equity_curve = portfolio.get_equity_curve()
            if len(equity_curve) < 2:
                return self._empty_chart(title)
            
            # Calculate returns
            returns = np.diff(equity_curve) / equity_curve[:-1] * 100
            
            if self.style == "plotly":
                return self._create_returns_distribution_plotly(returns, title)
            else:
                return self._create_returns_distribution_matplotlib(returns, title)
                
        except Exception as e:
            logger.error(f"Error creating returns distribution: {e}")
            return self._empty_chart(title)
    
    def create_rolling_metrics(self, portfolio: Portfolio, window_days: int = 60,
                             title: str = "Rolling Performance Metrics") -> Union[go.Figure, plt.Figure]:
        """
        Create rolling performance metrics chart.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio with equity history
        window_days : int
            Rolling window size
        title : str
            Chart title
            
        Returns
        -------
        Union[go.Figure, plt.Figure]
            Chart figure
        """
        try:
            equity_curve = portfolio.get_equity_curve()
            if len(equity_curve) < window_days:
                return self._empty_chart(title)
            
            # Calculate rolling metrics
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            rolling_sharpe = []
            rolling_vol = []
            
            for i in range(window_days, len(returns)):
                window_returns = returns[i-window_days:i]
                vol = np.std(window_returns) * np.sqrt(252)
                ret = np.mean(window_returns) * 252
                sharpe = ret / vol if vol > 0 else 0
                
                rolling_sharpe.append(sharpe)
                rolling_vol.append(vol)
            
            dates = pd.date_range(start='2020-01-01', periods=len(rolling_sharpe), freq='D')
            
            if self.style == "plotly":
                return self._create_rolling_metrics_plotly(rolling_sharpe, rolling_vol, dates, title)
            else:
                return self._create_rolling_metrics_matplotlib(rolling_sharpe, rolling_vol, dates, title)
                
        except Exception as e:
            logger.error(f"Error creating rolling metrics: {e}")
            return self._empty_chart(title)
    
    def create_monte_carlo_results(self, mc_results: MonteCarloResults,
                                 title: str = "Monte Carlo Analysis") -> Union[go.Figure, plt.Figure]:
        """
        Create Monte Carlo results visualization.
        
        Parameters
        ----------
        mc_results : MonteCarloResults
            Monte Carlo analysis results
        title : str
            Chart title
            
        Returns
        -------
        Union[go.Figure, plt.Figure]
            Chart figure
        """
        try:
            if not mc_results.simulation_results:
                return self._empty_chart(title)
            
            df = pd.DataFrame(mc_results.simulation_results)
            
            if self.style == "plotly":
                return self._create_monte_carlo_plotly(df, mc_results.base_performance, title)
            else:
                return self._create_monte_carlo_matplotlib(df, mc_results.base_performance, title)
                
        except Exception as e:
            logger.error(f"Error creating Monte Carlo chart: {e}")
            return self._empty_chart(title)
    
    def create_walk_forward_results(self, wf_results: WalkForwardResults,
                                  title: str = "Walk Forward Analysis") -> Union[go.Figure, plt.Figure]:
        """
        Create walk-forward analysis visualization.
        
        Parameters
        ----------
        wf_results : WalkForwardResults
            Walk-forward analysis results
        title : str
            Chart title
            
        Returns
        -------
        Union[go.Figure, plt.Figure]
            Chart figure
        """
        try:
            if not wf_results.periods:
                return self._empty_chart(title)
            
            performance_df = wf_results.get_performance_summary()
            
            if self.style == "plotly":
                return self._create_walk_forward_plotly(performance_df, title)
            else:
                return self._create_walk_forward_matplotlib(performance_df, title)
                
        except Exception as e:
            logger.error(f"Error creating walk-forward chart: {e}")
            return self._empty_chart(title)
    
    # Plotly implementations
    def _create_equity_curve_plotly(self, equity_curve: List[float], dates: pd.DatetimeIndex,
                                  benchmark_data: Optional[pd.Series], title: str) -> go.Figure:
        """Create equity curve using plotly."""
        fig = go.Figure()
        
        # Add portfolio equity curve
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark if provided
        if benchmark_data is not None:
            fig.add_trace(go.Scatter(
                x=dates[:len(benchmark_data)],
                y=benchmark_data.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template=self.template,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_drawdown_plotly(self, drawdowns: np.ndarray, dates: pd.DatetimeIndex, title: str) -> go.Figure:
        """Create drawdown chart using plotly."""
        fig = go.Figure()
        
        # Add drawdown area
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red'),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=self.template,
            yaxis=dict(range=[min(drawdowns) * 1.1, 5])
        )
        
        return fig
    
    def _create_returns_distribution_plotly(self, returns: np.ndarray, title: str) -> go.Figure:
        """Create returns distribution using plotly."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Histogram', 'Q-Q Plot'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns', opacity=0.7),
            row=1, col=1
        )
        
        # Normal distribution overlay
        x_norm = np.linspace(min(returns), max(returns), 100)
        y_norm = len(returns) * (x_norm[1] - x_norm[0]) * \
                 (1/np.sqrt(2*np.pi*np.var(returns))) * \
                 np.exp(-0.5 * ((x_norm - np.mean(returns))/np.std(returns))**2)
        
        fig.add_trace(
            go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Dist',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Q-Q plot
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm", plot=None)
        
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Data', 
                      marker=dict(color='blue', size=4)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', 
                      name='Normal Line', line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.template,
            showlegend=True
        )
        
        return fig
    
    def _create_rolling_metrics_plotly(self, rolling_sharpe: List[float], rolling_vol: List[float],
                                     dates: pd.DatetimeIndex, title: str) -> go.Figure:
        """Create rolling metrics chart using plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=['Rolling Sharpe Ratio', 'Rolling Volatility'],
            vertical_spacing=0.1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(x=dates, y=rolling_sharpe, mode='lines', name='Sharpe Ratio',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(x=dates, y=rolling_vol, mode='lines', name='Volatility',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=600
        )
        
        return fig
    
    def _create_monte_carlo_plotly(self, df: pd.DataFrame, base_performance: Dict[str, float], 
                                 title: str) -> go.Figure:
        """Create Monte Carlo results using plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Total Return Distribution', 'Sharpe Ratio Distribution',
                          'Risk vs Return', 'Drawdown Distribution']
        )
        
        # Total return distribution
        if 'total_return' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['total_return'] * 100, nbinsx=50, name='Total Return',
                           opacity=0.7),
                row=1, col=1
            )
            
            # Add base performance line
            base_return = base_performance.get('total_return', 0) * 100
            fig.add_vline(x=base_return, line_dash="dash", line_color="red",
                         annotation_text=f"Base: {base_return:.1f}%", row=1, col=1)
        
        # Sharpe ratio distribution
        if 'sharpe_ratio' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['sharpe_ratio'], nbinsx=50, name='Sharpe Ratio',
                           opacity=0.7),
                row=1, col=2
            )
            
            base_sharpe = base_performance.get('sharpe_ratio', 0)
            fig.add_vline(x=base_sharpe, line_dash="dash", line_color="red",
                         annotation_text=f"Base: {base_sharpe:.2f}", row=1, col=2)
        
        # Risk vs return scatter
        if 'total_return' in df.columns and 'volatility' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['volatility'] * 100, y=df['total_return'] * 100,
                          mode='markers', name='Simulations', 
                          marker=dict(size=4, opacity=0.6)),
                row=2, col=1
            )
            
            # Add base performance point
            base_vol = base_performance.get('volatility', 0) * 100
            base_ret = base_performance.get('total_return', 0) * 100
            fig.add_trace(
                go.Scatter(x=[base_vol], y=[base_ret], mode='markers',
                          name='Base Performance', marker=dict(size=12, color='red')),
                row=2, col=1
            )
        
        # Max drawdown distribution
        if 'max_drawdown' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['max_drawdown'] * 100, nbinsx=50, name='Max Drawdown',
                           opacity=0.7),
                row=2, col=2
            )
            
            base_dd = base_performance.get('max_drawdown', 0) * 100
            fig.add_vline(x=base_dd, line_dash="dash", line_color="red",
                         annotation_text=f"Base: {base_dd:.1f}%", row=2, col=2)
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_walk_forward_plotly(self, performance_df: pd.DataFrame, title: str) -> go.Figure:
        """Create walk-forward results using plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['In-Sample vs Out-of-Sample Sharpe', 'Performance Degradation',
                          'Return Comparison', 'Trade Count Evolution']
        )
        
        # In-sample vs out-of-sample Sharpe
        if 'in_sample_sharpe' in performance_df.columns and 'out_sample_sharpe' in performance_df.columns:
            fig.add_trace(
                go.Scatter(x=performance_df['period_id'], y=performance_df['in_sample_sharpe'],
                          mode='lines+markers', name='In-Sample Sharpe'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=performance_df['period_id'], y=performance_df['out_sample_sharpe'],
                          mode='lines+markers', name='Out-of-Sample Sharpe'),
                row=1, col=1
            )
        
        # Performance degradation
        if 'degradation_ratio' in performance_df.columns:
            fig.add_trace(
                go.Bar(x=performance_df['period_id'], y=performance_df['degradation_ratio'],
                      name='Degradation Ratio'),
                row=1, col=2
            )
        
        # Return comparison
        if 'in_sample_return' in performance_df.columns and 'out_sample_return' in performance_df.columns:
            fig.add_trace(
                go.Scatter(x=performance_df['period_id'], y=performance_df['in_sample_return'] * 100,
                          mode='lines+markers', name='In-Sample Return'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=performance_df['period_id'], y=performance_df['out_sample_return'] * 100,
                          mode='lines+markers', name='Out-of-Sample Return'),
                row=2, col=1
            )
        
        # Trade count evolution
        if 'num_trades_is' in performance_df.columns and 'num_trades_oos' in performance_df.columns:
            fig.add_trace(
                go.Bar(x=performance_df['period_id'], y=performance_df['num_trades_is'],
                      name='In-Sample Trades', opacity=0.7),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=performance_df['period_id'], y=performance_df['num_trades_oos'],
                      name='Out-of-Sample Trades', opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=800
        )
        
        return fig
    
    # Matplotlib implementations (simplified)
    def _create_equity_curve_matplotlib(self, equity_curve: List[float], dates: pd.DatetimeIndex,
                                      benchmark_data: Optional[pd.Series], title: str) -> plt.Figure:
        """Create equity curve using matplotlib."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.plot(dates, equity_curve, label='Portfolio', linewidth=2)
        
        if benchmark_data is not None:
            ax.plot(dates[:len(benchmark_data)], benchmark_data.values, 
                   label='Benchmark', linestyle='--', linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _create_drawdown_matplotlib(self, drawdowns: np.ndarray, dates: pd.DatetimeIndex, title: str) -> plt.Figure:
        """Create drawdown chart using matplotlib."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
        ax.plot(dates, drawdowns, color='red', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _create_returns_distribution_matplotlib(self, returns: np.ndarray, title: str) -> plt.Figure:
        """Create returns distribution using matplotlib."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, density=True)
        
        # Normal distribution overlay
        x_norm = np.linspace(min(returns), max(returns), 100)
        y_norm = (1/np.sqrt(2*np.pi*np.var(returns))) * \
                 np.exp(-0.5 * ((x_norm - np.mean(returns))/np.std(returns))**2)
        ax1.plot(x_norm, y_norm, 'r--', label='Normal Distribution')
        
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def _create_rolling_metrics_matplotlib(self, rolling_sharpe: List[float], rolling_vol: List[float],
                                         dates: pd.DatetimeIndex, title: str) -> plt.Figure:
        """Create rolling metrics using matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Rolling Sharpe
        ax1.plot(dates, rolling_sharpe, color='blue', linewidth=2)
        ax1.set_title('Rolling Sharpe Ratio')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Rolling Volatility
        ax2.plot(dates, rolling_vol, color='red', linewidth=2)
        ax2.set_title('Rolling Volatility')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def _create_monte_carlo_matplotlib(self, df: pd.DataFrame, base_performance: Dict[str, float], 
                                     title: str) -> plt.Figure:
        """Create Monte Carlo results using matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total return distribution
        if 'total_return' in df.columns:
            axes[0, 0].hist(df['total_return'] * 100, bins=50, alpha=0.7)
            base_return = base_performance.get('total_return', 0) * 100
            axes[0, 0].axvline(base_return, color='red', linestyle='--', 
                              label=f'Base: {base_return:.1f}%')
            axes[0, 0].set_title('Total Return Distribution')
            axes[0, 0].set_xlabel('Return (%)')
            axes[0, 0].legend()
        
        # Sharpe ratio distribution
        if 'sharpe_ratio' in df.columns:
            axes[0, 1].hist(df['sharpe_ratio'], bins=50, alpha=0.7)
            base_sharpe = base_performance.get('sharpe_ratio', 0)
            axes[0, 1].axvline(base_sharpe, color='red', linestyle='--',
                              label=f'Base: {base_sharpe:.2f}')
            axes[0, 1].set_title('Sharpe Ratio Distribution')
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].legend()
        
        # Risk vs return scatter
        if 'total_return' in df.columns and 'volatility' in df.columns:
            axes[1, 0].scatter(df['volatility'] * 100, df['total_return'] * 100, 
                              alpha=0.6, s=20)
            base_vol = base_performance.get('volatility', 0) * 100
            base_ret = base_performance.get('total_return', 0) * 100
            axes[1, 0].scatter(base_vol, base_ret, color='red', s=100, 
                              label='Base Performance')
            axes[1, 0].set_title('Risk vs Return')
            axes[1, 0].set_xlabel('Volatility (%)')
            axes[1, 0].set_ylabel('Return (%)')
            axes[1, 0].legend()
        
        # Max drawdown distribution
        if 'max_drawdown' in df.columns:
            axes[1, 1].hist(df['max_drawdown'] * 100, bins=50, alpha=0.7)
            base_dd = base_performance.get('max_drawdown', 0) * 100
            axes[1, 1].axvline(base_dd, color='red', linestyle='--',
                              label=f'Base: {base_dd:.1f}%')
            axes[1, 1].set_title('Max Drawdown Distribution')
            axes[1, 1].set_xlabel('Max Drawdown (%)')
            axes[1, 1].legend()
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def _create_walk_forward_matplotlib(self, performance_df: pd.DataFrame, title: str) -> plt.Figure:
        """Create walk-forward results using matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        x = performance_df['period_id']
        
        # In-sample vs out-of-sample Sharpe
        if 'in_sample_sharpe' in performance_df.columns and 'out_sample_sharpe' in performance_df.columns:
            axes[0, 0].plot(x, performance_df['in_sample_sharpe'], 'bo-', label='In-Sample')
            axes[0, 0].plot(x, performance_df['out_sample_sharpe'], 'ro-', label='Out-of-Sample')
            axes[0, 0].set_title('Sharpe Ratio Evolution')
            axes[0, 0].set_xlabel('Period')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].legend()
        
        # Performance degradation
        if 'degradation_ratio' in performance_df.columns:
            axes[0, 1].bar(x, performance_df['degradation_ratio'], alpha=0.7)
            axes[0, 1].set_title('Performance Degradation')
            axes[0, 1].set_xlabel('Period')
            axes[0, 1].set_ylabel('Degradation Ratio')
        
        # Return comparison
        if 'in_sample_return' in performance_df.columns and 'out_sample_return' in performance_df.columns:
            axes[1, 0].plot(x, performance_df['in_sample_return'] * 100, 'bo-', label='In-Sample')
            axes[1, 0].plot(x, performance_df['out_sample_return'] * 100, 'ro-', label='Out-of-Sample')
            axes[1, 0].set_title('Return Evolution')
            axes[1, 0].set_xlabel('Period')
            axes[1, 0].set_ylabel('Return (%)')
            axes[1, 0].legend()
        
        # Trade count evolution
        if 'num_trades_is' in performance_df.columns and 'num_trades_oos' in performance_df.columns:
            width = 0.35
            axes[1, 1].bar(x - width/2, performance_df['num_trades_is'], width, 
                          label='In-Sample', alpha=0.7)
            axes[1, 1].bar(x + width/2, performance_df['num_trades_oos'], width, 
                          label='Out-of-Sample', alpha=0.7)
            axes[1, 1].set_title('Trade Count Evolution')
            axes[1, 1].set_xlabel('Period')
            axes[1, 1].set_ylabel('Number of Trades')
            axes[1, 1].legend()
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def _empty_chart(self, title: str) -> Union[go.Figure, plt.Figure]:
        """Create empty chart with message."""
        if self.style == "plotly":
            fig = go.Figure()
            fig.update_layout(
                title=title,
                annotations=[dict(text="No data available", 
                                x=0.5, y=0.5, xref="paper", yref="paper",
                                showarrow=False, font=dict(size=20))]
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=20)
            ax.set_title(title)
            return fig
    
    def save_chart(self, fig: Union[go.Figure, plt.Figure], 
                  filename: str, format: str = "png") -> None:
        """
        Save chart to file.
        
        Parameters
        ----------
        fig : Union[go.Figure, plt.Figure]
            Chart figure
        filename : str
            Output filename
        format : str
            Output format ("png", "html", "pdf", "svg")
        """
        try:
            if isinstance(fig, go.Figure):
                if format == "html":
                    fig.write_html(filename)
                else:
                    fig.write_image(filename, format=format)
            else:
                fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
            
            logger.info(f"Chart saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
