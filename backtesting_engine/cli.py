"""
Command Line Interface for the Backtesting Engine.

This module provides a command-line interface for running backtests
and generating reports.
"""

import click
import logging
import yaml
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, Optional

from . import BacktestEngine
from .core.data_handler import DataConfig, create_data_handler
from .execution.broker import SimulatedBroker
from .execution.slippage import create_slippage_model
from .execution.commissions import create_commission_model
from .strategies.mean_reversion import MeanReversionStrategy


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_engine_from_config(config: Dict[str, Any]) -> BacktestEngine:
    """Create backtesting engine from configuration."""
    engine_config = config['engine']
    
    start_date = datetime.fromisoformat(engine_config['start_date'])
    end_date = datetime.fromisoformat(engine_config['end_date'])
    
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(engine_config['initial_capital'])),
        commission=Decimal(str(engine_config.get('commission', '0.001'))),
        margin_requirement=Decimal(str(engine_config.get('margin_requirement', '0.5'))),
        max_leverage=Decimal(str(engine_config.get('max_leverage', '2.0'))),
        benchmark=engine_config.get('benchmark')
    )
    
    return engine


def setup_data_handler(engine: BacktestEngine, config: Dict[str, Any]) -> None:
    """Set up data handler from configuration."""
    data_config_dict = config['data']
    
    start_date = datetime.fromisoformat(config['engine']['start_date'])
    end_date = datetime.fromisoformat(config['engine']['end_date'])
    
    data_config = DataConfig(
        source_type=data_config_dict['source'],
        path_or_connection=data_config_dict['path'],
        symbols=data_config_dict['symbols'],
        start_date=start_date,
        end_date=end_date,
        frequency=data_config_dict.get('frequency', 'daily'),
        validate_data=data_config_dict.get('validate_data', True),
        handle_missing=data_config_dict.get('handle_missing', 'forward_fill')
    )
    
    data_handler = create_data_handler(data_config)
    engine.add_data_handler(data_handler)


def setup_broker(engine: BacktestEngine, config: Dict[str, Any]) -> SimulatedBroker:
    """Set up broker from configuration."""
    execution_config = config.get('execution', {})
    
    # Create slippage model
    slippage_config = execution_config.get('slippage', {})
    slippage_type = slippage_config.get('model', 'linear')
    slippage_params = {k: Decimal(str(v)) if isinstance(v, (int, float)) else v 
                      for k, v in slippage_config.get('parameters', {}).items()}
    slippage_model = create_slippage_model(slippage_type, **slippage_params)
    
    # Create commission model
    commission_config = execution_config.get('commission', {})
    commission_type = commission_config.get('model', 'percentage')
    commission_params = {k: Decimal(str(v)) if isinstance(v, (int, float)) else v 
                        for k, v in commission_config.get('parameters', {}).items()}
    commission_model = create_commission_model(commission_type, **commission_params)
    
    # Create broker
    broker = SimulatedBroker(
        slippage_model=slippage_model,
        commission_model=commission_model,
        partial_fill_probability=execution_config.get('partial_fill_probability', 0.1),
        min_partial_fill_ratio=execution_config.get('min_partial_fill_ratio', 0.3),
        reject_probability=execution_config.get('reject_probability', 0.001)
    )
    
    engine.set_broker(broker)
    return broker


def setup_strategy(engine: BacktestEngine, config: Dict[str, Any]) -> None:
    """Set up strategy from configuration."""
    strategy_config = config['strategy']
    strategy_name = strategy_config['name']
    strategy_params = strategy_config.get('parameters', {})
    
    if strategy_name == 'MeanReversion':
        strategy = MeanReversionStrategy(
            strategy_id=f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbols=config['data']['symbols'],
            **strategy_params
        )
        engine.add_strategy(strategy.strategy_id, strategy)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


@click.group()
def cli():
    """Professional Backtesting Engine CLI."""
    pass


@cli.command()
@click.option('--config', '-c', required=True, help='Path to configuration file')
@click.option('--output', '-o', default='results', help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def run(config: str, output: str, verbose: bool):
    """Run a backtest with the specified configuration."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        click.echo(f"Loading configuration from {config}")
        config_data = load_config(config)
        
        # Create engine
        click.echo("Creating backtesting engine...")
        engine = create_engine_from_config(config_data)
        
        # Setup data handler
        click.echo("Setting up data handler...")
        setup_data_handler(engine, config_data)
        
        # Setup broker
        click.echo("Setting up execution system...")
        broker = setup_broker(engine, config_data)
        
        # Setup strategy
        click.echo("Setting up strategy...")
        setup_strategy(engine, config_data)
        
        # Run backtest
        click.echo("Running backtest...")
        with click.progressbar(length=100, label='Processing') as bar:
            # This is a simplified progress bar
            # In practice, you'd need to integrate this with the engine
            results = engine.run()
            bar.update(100)
        
        # Generate results
        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True)
        
        click.echo("Generating report...")
        report_path = results.generate_report(str(output_dir))
        
        # Print summary
        click.echo("\n" + "="*80)
        click.echo("BACKTEST COMPLETED")
        click.echo("="*80)
        results.print_summary()
        
        # Broker statistics
        broker_stats = broker.get_statistics()
        click.echo("\nEXECUTION STATISTICS")
        click.echo("-" * 40)
        click.echo(f"Total Orders: {broker_stats['total_orders']}")
        click.echo(f"Fill Rate: {broker_stats['fill_rate']:.2%}")
        click.echo(f"Average Slippage: {broker_stats['avg_slippage']:.4f}")
        
        click.echo(f"\nDetailed report saved to: {report_path}")
        click.echo("="*80)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--template', type=click.Choice(['basic', 'advanced']), default='basic')
@click.option('--output', '-o', default='config.yaml', help='Output configuration file')
def init_config(template: str, output: str):
    """Initialize a configuration file template."""
    
    if template == 'basic':
        config = {
            'engine': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 1000000,
                'commission': 0.001,
                'margin_requirement': 0.5,
                'max_leverage': 2.0
            },
            'data': {
                'source': 'csv',
                'path': 'data/',
                'symbols': ['AAPL', 'MSFT'],
                'frequency': 'daily',
                'validate_data': True,
                'handle_missing': 'forward_fill'
            },
            'strategy': {
                'name': 'MeanReversion',
                'parameters': {
                    'lookback_period': 20,
                    'std_dev_multiplier': 2.0,
                    'rsi_period': 14,
                    'position_size': 0.05,
                    'stop_loss': 0.05,
                    'take_profit': 0.10
                }
            },
            'execution': {
                'slippage': {
                    'model': 'linear',
                    'parameters': {
                        'base_rate': 0.001,
                        'size_impact': 0.01,
                        'volatility_impact': 0.005
                    }
                },
                'commission': {
                    'model': 'percentage',
                    'parameters': {
                        'commission_rate': 0.001,
                        'min_commission': 1.0
                    }
                },
                'partial_fill_probability': 0.1
            }
        }
    else:  # advanced
        config = {
            'engine': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 5000000,
                'commission': 0.0005,
                'margin_requirement': 0.25,
                'max_leverage': 4.0,
                'benchmark': 'SPY'
            },
            'data': {
                'source': 'api',
                'path': 'yfinance',
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'frequency': 'daily',
                'validate_data': True,
                'handle_missing': 'forward_fill'
            },
            'strategy': {
                'name': 'MeanReversion',
                'parameters': {
                    'lookback_period': 30,
                    'std_dev_multiplier': 2.5,
                    'rsi_period': 14,
                    'rsi_oversold': 25,
                    'rsi_overbought': 75,
                    'position_size': 0.03,
                    'stop_loss': 0.03,
                    'take_profit': 0.15
                }
            },
            'execution': {
                'slippage': {
                    'model': 'square_root',
                    'parameters': {
                        'impact_coefficient': 0.1,
                        'volatility_scaling': 1.5
                    }
                },
                'commission': {
                    'model': 'interactive_brokers',
                    'parameters': {
                        'account_type': 'pro'
                    }
                },
                'partial_fill_probability': 0.05,
                'reject_probability': 0.0005
            },
            'risk': {
                'max_position_size': 0.1,
                'max_portfolio_leverage': 3.0,
                'stop_loss_global': 0.02
            }
        }
    
    # Save configuration
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    click.echo(f"Configuration template saved to: {output}")
    click.echo(f"Template type: {template}")
    click.echo("Edit the configuration file before running your backtest.")


@cli.command()
@click.argument('report_path')
def view_report(report_path: str):
    """Open a backtest report in the browser."""
    import webbrowser
    from pathlib import Path
    
    report_file = Path(report_path)
    if not report_file.exists():
        click.echo(f"Report file not found: {report_path}", err=True)
        raise click.Abort()
    
    if report_file.suffix.lower() != '.html':
        click.echo("Report file must be an HTML file", err=True)
        raise click.Abort()
    
    # Open in browser
    webbrowser.open(f"file://{report_file.absolute()}")
    click.echo(f"Opened report in browser: {report_path}")


@cli.command()
@click.option('--config', '-c', required=True, help='Base configuration file')
@click.option('--param', '-p', multiple=True, help='Parameter to optimize (format: param_name:min:max:step)')
@click.option('--output', '-o', default='optimization_results', help='Output directory')
@click.option('--metric', default='sharpe_ratio', help='Optimization metric')
def optimize(config: str, param: tuple, output: str, metric: str):
    """Run parameter optimization (placeholder for future implementation)."""
    click.echo("Parameter optimization feature coming soon!")
    click.echo(f"Would optimize config: {config}")
    click.echo(f"Parameters: {param}")
    click.echo(f"Metric: {metric}")
    click.echo(f"Output: {output}")
    
    # This would implement a full parameter optimization routine
    # using techniques like grid search, random search, or Bayesian optimization


@cli.command()
def version():
    """Show version information."""
    from . import __version__
    click.echo(f"Backtesting Engine v{__version__}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()
