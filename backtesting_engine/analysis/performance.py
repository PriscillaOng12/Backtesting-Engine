"""
Performance analysis and results generation.

This module provides comprehensive performance analysis tools including
metrics calculation, visualization, and report generation.
"""

import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..core.portfolio import Portfolio, PortfolioSummary
from ..core.events import FillEvent, OrderSide


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Returns
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    cvar_95: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Other metrics
    periods: int
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    beta: Optional[float] = None
    alpha: Optional[float] = None

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis engine.
    
    Calculates various performance metrics, risk measures, and generates
    visualizations for backtesting results.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(
        self,
        equity_curve: List[Tuple[datetime, Decimal]],
        trades: List[FillEvent],
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: List of (timestamp, equity_value) tuples
            trades: List of completed trades
            benchmark_returns: Benchmark returns for beta/alpha calculation
            
        Returns:
            PerformanceMetrics object
        """
        if not equity_curve or len(equity_curve) < 2:
            raise ValueError("Insufficient equity curve data")
        
        # Convert to DataFrame for easier analysis
        if not equity_curve or len(equity_curve) < 2:
            # Create minimal valid equity curve
            now = datetime.now()
            equity_curve = [(now, Decimal('1000000')), (now + timedelta(days=1), Decimal('1000000'))]
        
        # Create a dictionary to store unique timestamps (keeping last values)
        equity_dict = {}
        for timestamp, equity in equity_curve:
            equity_dict[timestamp] = float(equity)
        
        # Convert to lists for DataFrame creation
        timestamps = list(equity_dict.keys())
        equities = list(equity_dict.values())
        
        # Calculate returns before creating DataFrame
        returns = [None]  # First return is NaN/None
        for i in range(1, len(equities)):
            if equities[i-1] != 0:
                returns.append((equities[i] - equities[i-1]) / equities[i-1])
            else:
                returns.append(0.0)
        
        # Create DataFrame with all data at once to avoid reindexing issues
        data_dict = {
            'timestamp': timestamps, 
            'equity': equities,
            'returns': returns
        }
        df = pd.DataFrame(data_dict)
        df = df.set_index('timestamp').sort_index()
        
        # Basic metrics
        total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
        periods = len(df)
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        
        annualized_return = (1 + total_return) ** (1 / max(years, 1/365.25)) - 1
        volatility = df['returns'].std() * math.sqrt(252)  # Annualized
        
        # Risk-adjusted returns
        excess_returns = df['returns'] - (self.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / df['returns'].std() * math.sqrt(252) if df['returns'].std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = df['returns'][df['returns'] < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_std * math.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown analysis
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        # Drawdown duration
        dd_duration = self._calculate_drawdown_duration(df['drawdown'])
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(df['returns'], 5)
        var_99 = np.percentile(df['returns'], 1)
        cvar_95 = df['returns'][df['returns'] <= var_95].mean()
        
        # Benchmark comparison
        beta, alpha = None, None
        if benchmark_returns is not None:
            beta, alpha = self._calculate_beta_alpha(df['returns'], benchmark_returns)
        
        # Trading metrics
        trade_metrics = self._calculate_trade_metrics(trades)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            periods=periods,
            start_date=df.index[0],
            end_date=df.index[-1],
            initial_capital=df['equity'].iloc[0],
            final_capital=df['equity'].iloc[-1],
            **trade_metrics
        )
    
    def _calculate_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdown_series < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_beta_alpha(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate beta and alpha relative to benchmark."""
        # Align the series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return 0.0, 0.0
        
        # Calculate beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha (annualized)
        portfolio_return = aligned_returns.mean() * 252
        benchmark_return = aligned_benchmark.mean() * 252
        alpha = portfolio_return - beta * benchmark_return
        
        return beta, alpha
    
    def _calculate_trade_metrics(self, trades: List[FillEvent]) -> Dict[str, Any]:
        """Calculate trading-specific metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Group trades by symbol to calculate P&L
        symbol_trades = {}
        for trade in trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        # Calculate trade P&L
        trade_pnls = []
        for symbol, symbol_trade_list in symbol_trades.items():
            # Simple P&L calculation (this is simplified)
            for i in range(0, len(symbol_trade_list), 2):
                if i + 1 < len(symbol_trade_list):
                    entry_trade = symbol_trade_list[i]
                    exit_trade = symbol_trade_list[i + 1]
                    
                    if entry_trade.side != exit_trade.side:
                        # Calculate P&L
                        if entry_trade.side == OrderSide.BUY:
                            pnl = (exit_trade.fill_price - entry_trade.fill_price) * entry_trade.quantity
                        else:
                            pnl = (entry_trade.fill_price - exit_trade.fill_price) * entry_trade.quantity
                        
                        # Subtract commissions
                        pnl -= (entry_trade.commission + exit_trade.commission)
                        trade_pnls.append(float(pnl))
        
        if not trade_pnls:
            return {
                'total_trades': len(trades),
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Calculate metrics
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
        
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0
        
        return {
            'total_trades': len(trade_pnls),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }


class BacktestResults:
    """
    Container for backtest results and analysis.
    
    Provides methods for accessing results data and generating reports.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        performance_history: List[PortfolioSummary],
        completed_trades: List[FillEvent],
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal,
        benchmark: Optional[str] = None
    ):
        """
        Initialize backtest results.
        
        Args:
            portfolio: Final portfolio state
            performance_history: Historical portfolio snapshots
            completed_trades: List of completed trades
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
            benchmark: Benchmark symbol (optional)
        """
        self.portfolio = portfolio
        self.performance_history = performance_history
        self.completed_trades = completed_trades
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        
        # Calculate metrics
        self.analyzer = PerformanceAnalyzer()
        self.metrics = self._calculate_performance_metrics()
        
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics from results."""
        return self.analyzer.calculate_metrics(
            equity_curve=self.portfolio.equity_curve,
            trades=self.completed_trades
        )
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        df = pd.DataFrame(self.portfolio.equity_curve, columns=['timestamp', 'equity'])
        df['equity'] = df['equity'].astype(float)
        
        # Handle duplicate timestamps by grouping by timestamp and taking the last value
        df = df.groupby('timestamp').last().reset_index()
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        return df
    
    def get_drawdown_curve(self) -> pd.DataFrame:
        """Get drawdown curve as DataFrame."""
        df = pd.DataFrame(self.portfolio.drawdown_curve, columns=['timestamp', 'drawdown'])
        df.set_index('timestamp', inplace=True)
        df['drawdown'] = df['drawdown'].astype(float)
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.completed_trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.completed_trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': getattr(trade.side, 'value', trade.side),
                'quantity': trade.quantity,
                'price': float(trade.fill_price),
                'commission': float(trade.commission),
                'slippage': float(trade.slippage)
            })
        
        return pd.DataFrame(trades_data)
    
    def plot_performance(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive performance visualization.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve', 'Drawdown',
                'Monthly Returns Heatmap', 'Rolling Sharpe Ratio',
                'Trade Distribution', 'Key Metrics'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "table"}]
            ]
        )
        
        # Equity curve
        equity_df = self.get_equity_curve()
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        drawdown_df = self.get_drawdown_curve()
        fig.add_trace(
            go.Scatter(
                x=drawdown_df.index,
                y=drawdown_df['drawdown'] * 100,
                mode='lines',
                name='Drawdown %',
                fill='tonexty',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
        
        # Add monthly returns heatmap (simplified)
        self._add_monthly_returns_heatmap(fig, equity_df, row=2, col=1)
        
        # Rolling Sharpe ratio
        self._add_rolling_sharpe(fig, equity_df, row=2, col=2)
        
        # Trade distribution
        if self.completed_trades:
            self._add_trade_distribution(fig, row=3, col=1)
        
        # Key metrics table
        self._add_metrics_table(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=f'Backtest Performance Report ({self.start_date.date()} to {self.end_date.date()})',
            height=1200,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _add_monthly_returns_heatmap(self, fig: go.Figure, equity_df: pd.DataFrame, row: int, col: int) -> None:
        """Add monthly returns heatmap to figure."""
        # Calculate monthly returns
        monthly_equity = equity_df.resample('M').last()
        monthly_returns = monthly_equity['equity'].pct_change().dropna()
        
        if len(monthly_returns) == 0:
            return
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        heatmap_data = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).first().unstack()
        
        if not heatmap_data.empty:
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values * 100,
                    x=[f"Month {i}" for i in heatmap_data.columns],
                    y=heatmap_data.index,
                    colorscale='RdYlGn',
                    name='Monthly Returns %'
                ),
                row=row, col=col
            )
    
    def _add_rolling_sharpe(self, fig: go.Figure, equity_df: pd.DataFrame, row: int, col: int) -> None:
        """Add rolling Sharpe ratio to figure."""
        returns = equity_df['equity'].pct_change().dropna()
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: x.mean() / x.std() * math.sqrt(252) if x.std() > 0 else 0
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe (252d)',
                line=dict(color='green', width=2)
            ),
            row=row, col=col
        )
    
    def _add_trade_distribution(self, fig: go.Figure, row: int, col: int) -> None:
        """Add trade P&L distribution to figure."""
        trades_df = self.get_trades_df()
        if trades_df.empty:
            return
        
        # Simple P&L approximation (this would be more sophisticated in practice)
        pnl_values = []
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            # Very simplified P&L calculation
            for i in range(0, len(symbol_trades), 2):
                if i + 1 < len(symbol_trades):
                    pnl = (symbol_trades.iloc[i+1]['price'] - symbol_trades.iloc[i]['price']) * \
                          symbol_trades.iloc[i]['quantity']
                    pnl_values.append(pnl)
        
        if pnl_values:
            fig.add_trace(
                go.Histogram(
                    x=pnl_values,
                    nbinsx=30,
                    name='Trade P&L Distribution',
                    marker_color='lightblue'
                ),
                row=row, col=col
            )
    
    def _add_metrics_table(self, fig: go.Figure, row: int, col: int) -> None:
        """Add key metrics table to figure."""
        metrics_data = [
            ['Total Return', f"{self.metrics.total_return:.2%}"],
            ['Annualized Return', f"{self.metrics.annualized_return:.2%}"],
            ['Volatility', f"{self.metrics.volatility:.2%}"],
            ['Sharpe Ratio', f"{self.metrics.sharpe_ratio:.2f}"],
            ['Max Drawdown', f"{self.metrics.max_drawdown:.2%}"],
            ['Win Rate', f"{self.metrics.win_rate:.2%}"],
            ['Profit Factor', f"{self.metrics.profit_factor:.2f}"],
            ['Total Trades', str(self.metrics.total_trades)]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*metrics_data)))
            ),
            row=row, col=col
        )
    
    def generate_report(self, output_dir: str = "results") -> str:
        """
        Generate a comprehensive HTML report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_path / f"backtest_report_{timestamp}.html"
        
        # Create the performance plot
        fig = self.plot_performance()
        
        # Save the interactive plot
        fig.write_html(str(report_path))
        
        logger.info(f"Generated backtest report: {report_path}")
        return str(report_path)
    
    def print_summary(self) -> None:
        """Print a summary of the backtest results."""
        print("=" * 80)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.metrics.final_capital:,.2f}")
        print()
        
        print("PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Total Return: {self.metrics.total_return:.2%}")
        print(f"Annualized Return: {self.metrics.annualized_return:.2%}")
        print(f"Volatility: {self.metrics.volatility:.2%}")
        print(f"Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {self.metrics.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {self.metrics.calmar_ratio:.2f}")
        print()
        
        print("RISK METRICS")
        print("-" * 40)
        print(f"Maximum Drawdown: {self.metrics.max_drawdown:.2%}")
        print(f"Max DD Duration: {self.metrics.max_drawdown_duration} periods")
        print(f"VaR (95%): {self.metrics.var_95:.2%}")
        print(f"CVaR (95%): {self.metrics.cvar_95:.2%}")
        print()
        
        print("TRADING METRICS")
        print("-" * 40)
        print(f"Total Trades: {self.metrics.total_trades}")
        print(f"Win Rate: {self.metrics.win_rate:.2%}")
        print(f"Profit Factor: {self.metrics.profit_factor:.2f}")
        print(f"Average Win: ${self.metrics.avg_win:.2f}")
        print(f"Average Loss: ${self.metrics.avg_loss:.2f}")
        print("=" * 80)
