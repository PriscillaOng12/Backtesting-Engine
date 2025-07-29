"""
Simple backtesting example.

This script demonstrates how to use the backtesting engine with a basic
mean reversion strategy.
"""

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from backtesting_engine import BacktestEngine
from backtesting_engine.core.data_handler import DataConfig, CSVDataHandler
from backtesting_engine.execution.broker import SimulatedBroker
from backtesting_engine.execution.slippage import LinearSlippageModel
from backtesting_engine.execution.commissions import PercentageCommissionModel
from backtesting_engine.strategies.mean_reversion import MeanReversionStrategy


def create_sample_data():
    """Create sample market data for testing."""
    import pandas as pd
    import numpy as np
    
    # Generate sample data for demonstration
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    for symbol in symbols:
        # Generate realistic-looking price data
        np.random.seed(42 + hash(symbol) % 1000)  # Consistent random data
        
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100
        
        # Add some volatility clustering
        volatility = 0.02 + 0.01 * np.abs(np.random.normal(0, 1, len(dates)))
        prices += prices * volatility * np.random.normal(0, 1, len(dates))
        
        # Generate OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
        df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        df['volume'] = np.random.lognormal(15, 0.5, len(dates)).astype(int)
        df['adj_close'] = df['close']  # Assume no adjustments
        
        # Clean up any NaN values
        df = df.dropna()
        
        # Save to CSV
        file_path = data_dir / f'{symbol}.csv'
        df.to_csv(file_path)
        print(f"Created sample data: {file_path}")


def run_simple_backtest():
    """Run a simple backtest example."""
    
    # Create sample data if it doesn't exist
    if not Path('data').exists():
        print("Creating sample data...")
        create_sample_data()
    
    # Configuration
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_capital = Decimal('1000000')  # $1M
    
    print(f"Starting backtest: {start_date.date()} to {end_date.date()}")
    print(f"Initial capital: ${initial_capital:,}")
    
    # Initialize backtesting engine
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=Decimal('0.001'),  # 0.1% commission
        margin_requirement=Decimal('0.5'),  # 50% margin
        max_leverage=Decimal('2.0')  # 2x leverage
    )
    
    # Set up data handler
    data_config = DataConfig(
        source_type='csv',
        path_or_connection='data',
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date=start_date,
        end_date=end_date,
        frequency='daily'
    )
    
    data_handler = CSVDataHandler(data_config)
    engine.add_data_handler(data_handler)
    
    # Set up execution system
    slippage_model = LinearSlippageModel(
        base_rate=Decimal('0.001'),
        size_impact=Decimal('0.01'),
        volatility_impact=Decimal('0.005')
    )
    
    commission_model = PercentageCommissionModel(
        commission_rate=Decimal('0.001'),
        min_commission=Decimal('1.0')
    )
    
    broker = SimulatedBroker(
        slippage_model=slippage_model,
        commission_model=commission_model,
        partial_fill_probability=0.1
    )
    engine.set_broker(broker)
    
    # Set up strategy
    strategy = MeanReversionStrategy(
        strategy_id="mean_reversion_example",
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        lookback_period=20,
        std_dev_multiplier=2.0,
        rsi_period=14,
        position_size=0.05,  # 5% per position
        stop_loss=0.05,      # 5% stop loss
        take_profit=0.10     # 10% take profit
    )
    
    engine.add_strategy(strategy.strategy_id, strategy)
    
    print("Configuration complete. Running backtest...")
    
    # Run the backtest
    try:
        results = engine.run()
        
        print("\nBacktest completed successfully!")
        
        # Print summary
        results.print_summary()
        
        # Generate detailed report
        report_path = results.generate_report()
        print(f"\nDetailed report generated: {report_path}")
        
        # Get broker statistics
        broker_stats = broker.get_statistics()
        print("\nBROKER STATISTICS")
        print("-" * 40)
        print(f"Total Orders: {broker_stats['total_orders']}")
        print(f"Filled Orders: {broker_stats['filled_orders']}")
        print(f"Fill Rate: {broker_stats['fill_rate']:.2%}")
        print(f"Average Commission: ${broker_stats['avg_commission']:.2f}")
        print(f"Average Slippage: ${broker_stats['avg_slippage']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_parameter_study():
    """Run a simple parameter study."""
    print("\nRunning parameter study...")
    
    parameters_to_test = [
        {'lookback_period': 10, 'std_dev_multiplier': 1.5},
        {'lookback_period': 20, 'std_dev_multiplier': 2.0},
        {'lookback_period': 30, 'std_dev_multiplier': 2.5},
    ]
    
    results_summary = []
    
    for i, params in enumerate(parameters_to_test):
        print(f"\nTesting parameter set {i+1}: {params}")
        
        # This would be a full backtest for each parameter set
        # For brevity, we'll just show the structure
        try:
            # Configure and run backtest with these parameters
            # results = run_backtest_with_params(params)
            # results_summary.append({
            #     'parameters': params,
            #     'total_return': results.metrics.total_return,
            #     'sharpe_ratio': results.metrics.sharpe_ratio,
            #     'max_drawdown': results.metrics.max_drawdown
            # })
            print(f"Parameter set {i+1} would be tested here")
            
        except Exception as e:
            print(f"Parameter set {i+1} failed: {e}")
    
    print("\nParameter study structure demonstrated")


if __name__ == "__main__":
    print("=" * 80)
    print("BACKTESTING ENGINE EXAMPLE")
    print("=" * 80)
    
    # Run simple backtest
    results = run_simple_backtest()
    
    if results:
        # Demonstrate parameter study structure
        run_parameter_study()
        
        print("\n" + "=" * 80)
        print("Example completed successfully!")
        print("Check the generated report for detailed analysis.")
        print("=" * 80)
    else:
        print("\nExample failed. Check error messages above.")


# Additional utility functions
def load_real_data():
    """Example of how to load real market data."""
    print("\nTo use real market data:")
    print("1. Install yfinance: pip install yfinance")
    print("2. Use APIDataHandler with Yahoo Finance")
    print("3. Example:")
    print("""
    from backtesting_engine.core.data_handler import DataConfig, APIDataHandler
    
    data_config = DataConfig(
        source_type='api',
        path_or_connection='yfinance',
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        frequency='daily'
    )
    
    data_handler = APIDataHandler(data_config)
    """)


def advanced_configuration_example():
    """Show advanced configuration options."""
    print("\nAdvanced Configuration Examples:")
    print("""
    # Custom slippage model
    from backtesting_engine.execution.slippage import SquareRootSlippageModel
    slippage = SquareRootSlippageModel(
        impact_coefficient=Decimal('0.1'),
        volatility_scaling=Decimal('1.5')
    )
    
    # Custom commission model
    from backtesting_engine.execution.commissions import InteractiveBrokersCommissionModel
    commission = InteractiveBrokersCommissionModel(account_type="pro")
    
    # Risk management configuration
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal('1000000'),
        margin_requirement=Decimal('0.25'),  # 25% margin
        max_leverage=Decimal('4.0')          # 4x leverage
    )
    """)
