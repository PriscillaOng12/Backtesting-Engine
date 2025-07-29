"""
Comprehensive report generation for backtesting results.

This module generates detailed HTML and PDF reports including
performance metrics, charts, and analysis summaries.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import base64
from io import BytesIO

import pandas as pd
import numpy as np
from jinja2 import Template
import weasyprint  # For PDF generation
import plotly.graph_objects as go
import plotly.io as pio

from ..core.portfolio import Portfolio
from ..analysis.performance import BacktestResults, PerformanceMetrics
from ..analysis.walk_forward import WalkForwardResults
from ..analysis.monte_carlo import MonteCarloResults
from ..risk.metrics import RiskMetrics
from .charts import ChartGenerator


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive report generator for backtesting results.
    
    Generates professional HTML and PDF reports with embedded
    charts, performance metrics, and detailed analysis.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Parameters
        ----------
        template_dir : str, optional
            Directory containing custom templates
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.chart_generator = ChartGenerator(style="plotly", theme="white")
        
        # Default templates
        self.html_template = self._get_default_html_template()
        self.summary_template = self._get_summary_template()
        
        logger.info("Report generator initialized")
    
    def generate_full_report(self, 
                           portfolio: Portfolio,
                           performance_metrics: PerformanceMetrics,
                           risk_metrics: Optional[RiskMetrics] = None,
                           wf_results: Optional[WalkForwardResults] = None,
                           mc_results: Optional[MonteCarloResults] = None,
                           output_path: str = "backtest_report.html",
                           include_charts: bool = True) -> str:
        """
        Generate comprehensive backtest report.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio with trading history
        performance_metrics : PerformanceMetrics
            Performance analysis results
        risk_metrics : RiskMetrics, optional
            Risk analysis results
        wf_results : WalkForwardResults, optional
            Walk-forward analysis results
        mc_results : MonteCarloResults, optional
            Monte Carlo analysis results
        output_path : str
            Output file path
        include_charts : bool
            Whether to include charts in report
            
        Returns
        -------
        str
            Path to generated report
        """
        try:
            logger.info(f"Generating full report: {output_path}")
            
            # Prepare report data
            report_data = self._prepare_report_data(
                portfolio, performance_metrics, risk_metrics, 
                wf_results, mc_results
            )
            
            # Generate charts if requested
            charts_data = {}
            if include_charts:
                charts_data = self._generate_report_charts(
                    portfolio, wf_results, mc_results
                )
            
            # Render HTML report
            html_content = self._render_html_report(report_data, charts_data)
            
            # Save report
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def generate_summary_report(self, 
                              performance_metrics: PerformanceMetrics,
                              output_path: str = "summary_report.html") -> str:
        """
        Generate concise summary report.
        
        Parameters
        ----------
        performance_metrics : PerformanceMetrics
            Performance metrics
        output_path : str
            Output file path
            
        Returns
        -------
        str
            Path to generated report
        """
        try:
            # Prepare summary data
            summary_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': asdict(performance_metrics)
            }
            
            # Render summary
            html_content = self.summary_template.render(**summary_data)
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Summary report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise
    
    def generate_pdf_report(self, html_path: str, 
                          pdf_path: Optional[str] = None) -> str:
        """
        Convert HTML report to PDF.
        
        Parameters
        ----------
        html_path : str
            Path to HTML report
        pdf_path : str, optional
            Output PDF path
            
        Returns
        -------
        str
            Path to generated PDF
        """
        try:
            if pdf_path is None:
                pdf_path = html_path.replace('.html', '.pdf')
            
            # Read HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Generate PDF
            doc = weasyprint.HTML(string=html_content)
            doc.write_pdf(pdf_path)
            
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise
    
    def generate_comparison_report(self, 
                                 strategies_data: Dict[str, Dict[str, Any]],
                                 output_path: str = "comparison_report.html") -> str:
        """
        Generate strategy comparison report.
        
        Parameters
        ----------
        strategies_data : dict
            Dictionary of strategy results
        output_path : str
            Output file path
            
        Returns
        -------
        str
            Path to generated report
        """
        try:
            # Prepare comparison data
            comparison_data = self._prepare_comparison_data(strategies_data)
            
            # Generate comparison charts
            comparison_charts = self._generate_comparison_charts(strategies_data)
            
            # Use comparison template
            template = self._get_comparison_template()
            html_content = template.render(
                strategies=comparison_data,
                charts=comparison_charts,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Comparison report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            raise
    
    def _prepare_report_data(self, portfolio: Portfolio, 
                           performance_metrics: PerformanceMetrics,
                           risk_metrics: Optional[RiskMetrics],
                           wf_results: Optional[WalkForwardResults],
                           mc_results: Optional[MonteCarloResults]) -> Dict[str, Any]:
        """Prepare all data for the report."""
        
        # Basic portfolio information
        equity_curve = portfolio.get_equity_curve()
        
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio': {
                'initial_capital': float(portfolio.initial_capital),
                'final_equity': float(portfolio.calculate_total_equity()),
                'cash_position': float(portfolio.cash),
                'num_positions': len(portfolio.positions),
                'equity_curve_length': len(equity_curve)
            },
            'performance': asdict(performance_metrics),
            'risk': asdict(risk_metrics) if risk_metrics else None,
            'positions': []
        }
        
        # Position details
        for symbol, position in portfolio.positions.items():
            position_data = {
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_price': float(position.avg_price),
                'market_value': float(position.market_value),
                'unrealized_pnl': float(position.unrealized_pnl)
            }
            data['positions'].append(position_data)
        
        # Walk-forward analysis
        if wf_results:
            wf_summary = wf_results.get_performance_summary()
            data['walk_forward'] = {
                'num_periods': len(wf_results.periods),
                'avg_is_sharpe': wf_summary['in_sample_sharpe'].mean() if not wf_summary.empty else 0,
                'avg_oos_sharpe': wf_summary['out_sample_sharpe'].mean() if not wf_summary.empty else 0,
                'success_rate': wf_results.success_rate,
                'summary_stats': wf_results.summary_statistics
            }
        
        # Monte Carlo analysis
        if mc_results:
            data['monte_carlo'] = {
                'num_simulations': len(mc_results.simulation_results),
                'success_rate': mc_results.success_rate,
                'probability_of_loss': mc_results.probability_of_loss,
                'stability_ratio': mc_results.stability_ratio,
                'confidence_intervals': mc_results.confidence_intervals
            }
        
        return data
    
    def _generate_report_charts(self, portfolio: Portfolio,
                              wf_results: Optional[WalkForwardResults],
                              mc_results: Optional[MonteCarloResults]) -> Dict[str, str]:
        """Generate charts for the report."""
        charts = {}
        
        try:
            # Equity curve
            fig = self.chart_generator.create_equity_curve(portfolio)
            charts['equity_curve'] = self._fig_to_html(fig)
            
            # Drawdown chart
            fig = self.chart_generator.create_drawdown_chart(portfolio)
            charts['drawdown'] = self._fig_to_html(fig)
            
            # Returns distribution
            fig = self.chart_generator.create_returns_distribution(portfolio)
            charts['returns_distribution'] = self._fig_to_html(fig)
            
            # Rolling metrics
            fig = self.chart_generator.create_rolling_metrics(portfolio)
            charts['rolling_metrics'] = self._fig_to_html(fig)
            
            # Walk-forward analysis
            if wf_results:
                fig = self.chart_generator.create_walk_forward_results(wf_results)
                charts['walk_forward'] = self._fig_to_html(fig)
            
            # Monte Carlo analysis
            if mc_results:
                fig = self.chart_generator.create_monte_carlo_results(mc_results)
                charts['monte_carlo'] = self._fig_to_html(fig)
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
        
        return charts
    
    def _render_html_report(self, report_data: Dict[str, Any], 
                          charts_data: Dict[str, str]) -> str:
        """Render the HTML report."""
        return self.html_template.render(
            data=report_data,
            charts=charts_data
        )
    
    def _prepare_comparison_data(self, strategies_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for strategy comparison."""
        comparison_data = []
        
        for strategy_name, data in strategies_data.items():
            strategy_summary = {
                'name': strategy_name,
                'total_return': data.get('performance_metrics', {}).get('total_return', 0),
                'sharpe_ratio': data.get('performance_metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': data.get('performance_metrics', {}).get('max_drawdown', 0),
                'volatility': data.get('performance_metrics', {}).get('volatility', 0),
                'num_trades': data.get('performance_metrics', {}).get('total_trades', 0)
            }
            comparison_data.append(strategy_summary)
        
        return comparison_data
    
    def _generate_comparison_charts(self, strategies_data: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Generate comparison charts."""
        charts = {}
        
        try:
            # Prepare comparison data
            comparison_df = pd.DataFrame(self._prepare_comparison_data(strategies_data))
            
            if not comparison_df.empty:
                # Performance comparison bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Total Return',
                    x=comparison_df['name'],
                    y=comparison_df['total_return'] * 100,
                    yaxis='y1'
                ))
                
                fig.add_trace(go.Scatter(
                    name='Sharpe Ratio',
                    x=comparison_df['name'],
                    y=comparison_df['sharpe_ratio'],
                    yaxis='y2',
                    mode='markers+lines',
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title='Strategy Performance Comparison',
                    xaxis_title='Strategy',
                    yaxis=dict(title='Total Return (%)', side='left'),
                    yaxis2=dict(title='Sharpe Ratio', side='right', overlaying='y'),
                    template='plotly_white'
                )
                
                charts['performance_comparison'] = self._fig_to_html(fig)
                
                # Risk-return scatter
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=comparison_df['volatility'] * 100,
                    y=comparison_df['total_return'] * 100,
                    mode='markers+text',
                    text=comparison_df['name'],
                    textposition="top center",
                    marker=dict(size=12)
                ))
                
                fig.update_layout(
                    title='Risk-Return Profile',
                    xaxis_title='Volatility (%)',
                    yaxis_title='Total Return (%)',
                    template='plotly_white'
                )
                
                charts['risk_return'] = self._fig_to_html(fig)
            
        except Exception as e:
            logger.error(f"Error generating comparison charts: {e}")
        
        return charts
    
    def _fig_to_html(self, fig: go.Figure) -> str:
        """Convert plotly figure to HTML string."""
        return pio.to_html(fig, include_plotlyjs='inline', div_id=None, 
                          config={'displayModeBar': False})
    
    def _get_default_html_template(self) -> Template:
        """Get the default HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtesting Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            background: #fafbfc;
        }
        .section h2 {
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .table th, .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        .table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        .table tr:hover {
            background-color: #f8f9fa;
        }
        .summary-stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        .stat-item {
            text-align: center;
            margin: 10px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #666;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Professional Backtesting Report</h1>
            <div class="timestamp">Generated on {{ data.timestamp }}</div>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {% if data.performance.total_return > 0 %}positive{% else %}negative{% endif %}">
                        {{ "%.2f"|format(data.performance.total_return * 100) }}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {% if data.performance.sharpe_ratio > 1 %}positive{% elif data.performance.sharpe_ratio > 0 %}{% else %}negative{% endif %}">
                        {{ "%.3f"|format(data.performance.sharpe_ratio) }}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">
                        {{ "%.2f"|format(data.performance.max_drawdown * 100) }}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value">
                        {{ "%.2f"|format(data.performance.volatility * 100) }}%
                    </div>
                </div>
            </div>
        </div>

        <!-- Portfolio Overview -->
        <div class="section">
            <h2>Portfolio Overview</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">${{ "{:,.0f}"|format(data.portfolio.initial_capital) }}</div>
                    <div class="stat-label">Initial Capital</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${{ "{:,.0f}"|format(data.portfolio.final_equity) }}</div>
                    <div class="stat-label">Final Equity</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ data.portfolio.num_positions }}</div>
                    <div class="stat-label">Active Positions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ data.performance.total_trades }}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
            </div>
        </div>

        <!-- Performance Charts -->
        {% if charts %}
        <div class="section">
            <h2>Performance Charts</h2>
            
            {% if charts.equity_curve %}
            <div class="chart-container">
                <h3>Equity Curve</h3>
                {{ charts.equity_curve|safe }}
            </div>
            {% endif %}
            
            {% if charts.drawdown %}
            <div class="chart-container">
                <h3>Drawdown Analysis</h3>
                {{ charts.drawdown|safe }}
            </div>
            {% endif %}
            
            {% if charts.returns_distribution %}
            <div class="chart-container">
                <h3>Returns Distribution</h3>
                {{ charts.returns_distribution|safe }}
            </div>
            {% endif %}
            
            {% if charts.rolling_metrics %}
            <div class="chart-container">
                <h3>Rolling Performance Metrics</h3>
                {{ charts.rolling_metrics|safe }}
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Detailed Performance Metrics -->
        <div class="section">
            <h2>Detailed Performance Metrics</h2>
            <table class="table">
                <tr><td><strong>Total Return</strong></td><td>{{ "%.2f"|format(data.performance.total_return * 100) }}%</td></tr>
                <tr><td><strong>Annualized Return</strong></td><td>{{ "%.2f"|format(data.performance.annualized_return * 100) }}%</td></tr>
                <tr><td><strong>Volatility</strong></td><td>{{ "%.2f"|format(data.performance.volatility * 100) }}%</td></tr>
                <tr><td><strong>Sharpe Ratio</strong></td><td>{{ "%.3f"|format(data.performance.sharpe_ratio) }}</td></tr>
                <tr><td><strong>Sortino Ratio</strong></td><td>{{ "%.3f"|format(data.performance.sortino_ratio) }}</td></tr>
                <tr><td><strong>Calmar Ratio</strong></td><td>{{ "%.3f"|format(data.performance.calmar_ratio) }}</td></tr>
                <tr><td><strong>Maximum Drawdown</strong></td><td>{{ "%.2f"|format(data.performance.max_drawdown * 100) }}%</td></tr>
                <tr><td><strong>Win Rate</strong></td><td>{{ "%.2f"|format(data.performance.win_rate * 100) }}%</td></tr>
                <tr><td><strong>Profit Factor</strong></td><td>{{ "%.2f"|format(data.performance.profit_factor) }}</td></tr>
                <tr><td><strong>Total Trades</strong></td><td>{{ data.performance.total_trades }}</td></tr>
            </table>
        </div>

        <!-- Risk Metrics -->
        {% if data.risk %}
        <div class="section">
            <h2>Risk Analysis</h2>
            <table class="table">
                <tr><td><strong>VaR (95%)</strong></td><td>{{ "%.2f"|format(data.risk.var_95_daily * 100) }}%</td></tr>
                <tr><td><strong>CVaR (95%)</strong></td><td>{{ "%.2f"|format(data.risk.cvar_95_daily * 100) }}%</td></tr>
                <tr><td><strong>Skewness</strong></td><td>{{ "%.3f"|format(data.risk.skewness) }}</td></tr>
                <tr><td><strong>Kurtosis</strong></td><td>{{ "%.3f"|format(data.risk.kurtosis) }}</td></tr>
                <tr><td><strong>Max Drawdown Duration</strong></td><td>{{ data.risk.max_drawdown_duration }} days</td></tr>
            </table>
        </div>
        {% endif %}

        <!-- Current Positions -->
        {% if data.positions %}
        <div class="section">
            <h2>Current Positions</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Avg Price</th>
                        <th>Market Value</th>
                        <th>Unrealized P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {% for position in data.positions %}
                    <tr>
                        <td>{{ position.symbol }}</td>
                        <td>{{ position.quantity }}</td>
                        <td>${{ "%.2f"|format(position.avg_price) }}</td>
                        <td>${{ "{:,.2f}"|format(position.market_value) }}</td>
                        <td class="{% if position.unrealized_pnl > 0 %}positive{% else %}negative{% endif %}">
                            ${{ "{:,.2f}"|format(position.unrealized_pnl) }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Walk-Forward Analysis -->
        {% if data.walk_forward and charts.walk_forward %}
        <div class="section">
            <h2>Walk-Forward Analysis</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{{ data.walk_forward.num_periods }}</div>
                    <div class="stat-label">Test Periods</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.3f"|format(data.walk_forward.avg_is_sharpe) }}</div>
                    <div class="stat-label">Avg In-Sample Sharpe</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.3f"|format(data.walk_forward.avg_oos_sharpe) }}</div>
                    <div class="stat-label">Avg Out-of-Sample Sharpe</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(data.walk_forward.success_rate * 100) }}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
            <div class="chart-container">
                {{ charts.walk_forward|safe }}
            </div>
        </div>
        {% endif %}

        <!-- Monte Carlo Analysis -->
        {% if data.monte_carlo and charts.monte_carlo %}
        <div class="section">
            <h2>Monte Carlo Analysis</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{{ data.monte_carlo.num_simulations }}</div>
                    <div class="stat-label">Simulations</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(data.monte_carlo.probability_of_loss * 100) }}%</div>
                    <div class="stat-label">Probability of Loss</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(data.monte_carlo.stability_ratio * 100) }}%</div>
                    <div class="stat-label">Stability Ratio</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(data.monte_carlo.success_rate * 100) }}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
            <div class="chart-container">
                {{ charts.monte_carlo|safe }}
            </div>
        </div>
        {% endif %}

        <!-- Footer -->
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #666;">
            <p>Generated by Professional Backtesting Engine</p>
            <p><small>This report is for informational purposes only and does not constitute investment advice.</small></p>
        </div>
    </div>
</body>
</html>
        """
        
        return Template(template_str)
    
    def _get_summary_template(self) -> Template:
        """Get the summary template."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { margin: 10px 0; }
        .value { font-weight: bold; color: #007bff; }
    </style>
</head>
<body>
    <h1>Backtest Summary</h1>
    <p>Generated: {{ timestamp }}</p>
    
    <div class="metric">Total Return: <span class="value">{{ "%.2f"|format(metrics.total_return * 100) }}%</span></div>
    <div class="metric">Sharpe Ratio: <span class="value">{{ "%.3f"|format(metrics.sharpe_ratio) }}</span></div>
    <div class="metric">Max Drawdown: <span class="value">{{ "%.2f"|format(metrics.max_drawdown * 100) }}%</span></div>
    <div class="metric">Total Trades: <span class="value">{{ metrics.total_trades }}</span></div>
</body>
</html>
        """
        
        return Template(template_str)
    
    def _get_comparison_template(self) -> Template:
        """Get the comparison template."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .chart-container { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Strategy Comparison Report</h1>
    <p>Generated: {{ timestamp }}</p>
    
    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Volatility</th>
                <th>Trades</th>
            </tr>
        </thead>
        <tbody>
            {% for strategy in strategies %}
            <tr>
                <td>{{ strategy.name }}</td>
                <td>{{ "%.2f"|format(strategy.total_return * 100) }}%</td>
                <td>{{ "%.3f"|format(strategy.sharpe_ratio) }}</td>
                <td>{{ "%.2f"|format(strategy.max_drawdown * 100) }}%</td>
                <td>{{ "%.2f"|format(strategy.volatility * 100) }}%</td>
                <td>{{ strategy.num_trades }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    {% if charts.performance_comparison %}
    <div class="chart-container">
        <h2>Performance Comparison</h2>
        {{ charts.performance_comparison|safe }}
    </div>
    {% endif %}
    
    {% if charts.risk_return %}
    <div class="chart-container">
        <h2>Risk-Return Profile</h2>
        {{ charts.risk_return|safe }}
    </div>
    {% endif %}
</body>
</html>
        """
        
        return Template(template_str)
