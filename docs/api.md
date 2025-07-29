# API Reference Documentation

## Overview

This document provides API documentation for this backtesting engine, including detailed examples, error handling patterns, and integration guides. The API design follows REST principles.

## Core API Components

### BacktestEngine

The main orchestrator for all backtesting operations.

```python
class BacktestEngine:
    """
    Main backtesting engine that coordinates data, strategies, and execution.
    
    Thread-safe for concurrent strategy execution.
    Designed for both programmatic use and CLI integration.
    """
    
    def __init__(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        initial_capital: Union[float, Decimal] = 1_000_000,
        commission_model: Union[str, CommissionModel] = 'percentage',
        slippage_model: Union[str, SlippageModel] = 'fixed',
        benchmark: Optional[str] = None,
        currency: str = 'USD',
        **kwargs
    ) -> None:
        """
        Initialize backtesting engine.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD or datetime)
            end_date: Backtest end date (YYYY-MM-DD or datetime)
            initial_capital: Starting portfolio value
            commission_model: Commission calculation method
                - 'percentage': Percentage of trade value
                - 'fixed': Fixed amount per trade
                - 'tiered': Volume-based tiered structure
                - Custom CommissionModel instance
            slippage_model: Market impact calculation method
                - 'fixed': Fixed percentage slippage
                - 'linear': Linear market impact model
                - 'sqrt': Square-root market impact
                - Custom SlippageModel instance
            benchmark: Benchmark symbol for comparison (e.g., 'SPY')
            currency: Base currency for calculations
            
        Kwargs:
            max_leverage: Maximum portfolio leverage (default: 2.0)
            margin_requirement: Margin requirement for positions (default: 0.5)
            risk_free_rate: Risk-free rate for calculations (default: 0.02)
            save_snapshots: Save portfolio snapshots (default: True)
            
        Example:
            >>> engine = BacktestEngine(
            ...     start_date='2020-01-01',
            ...     end_date='2023-12-31',
            ...     initial_capital=1_000_000,
            ...     commission_model='tiered',
            ...     slippage_model='sqrt',
            ...     benchmark='SPY'
            ... )
        """
```

#### Core Methods

```python
def add_data_handler(self, handler: BaseDataHandler) -> None:
    """
    Add data source to the engine.
    
    Args:
        handler: Data handler instance (CSV, Database, API, etc.)
        
    Raises:
        ValueError: If handler symbols overlap with existing handlers
        DataValidationError: If handler data fails validation
        
    Example:
        >>> from backtesting_engine.data import CSVDataHandler
        >>> handler = CSVDataHandler(
        ...     symbols=['AAPL', 'MSFT'],
        ...     data_directory='./data/',
        ...     frequency='daily'
        ... )
        >>> engine.add_data_handler(handler)
    """

def add_strategy(self, strategy_id: str, strategy: BaseStrategy) -> None:
    """
    Add trading strategy to the engine.
    
    Args:
        strategy_id: Unique identifier for the strategy
        strategy: Strategy instance implementing BaseStrategy interface
        
    Raises:
        ValueError: If strategy_id already exists
        TypeError: If strategy doesn't implement BaseStrategy
        
    Example:
        >>> from backtesting_engine.strategies import MeanReversionStrategy
        >>> strategy = MeanReversionStrategy(
        ...     symbols=['AAPL'],
        ...     lookback_period=20,
        ...     entry_threshold=2.0,
        ...     exit_threshold=0.5
        ... )
        >>> engine.add_strategy('mean_reversion_1', strategy)
    """

def set_risk_manager(self, risk_manager: RiskManager) -> None:
    """
    Set risk management system.
    
    Args:
        risk_manager: Risk manager instance with portfolio controls
        
    Example:
        >>> from backtesting_engine.risk import RiskManager, RiskLimits
        >>> limits = RiskLimits(
        ...     max_position_size=0.1,
        ...     max_portfolio_leverage=2.0,
        ...     max_drawdown=0.15,
        ...     max_var_95=0.05
        ... )
        >>> risk_manager = RiskManager(limits)
        >>> engine.set_risk_manager(risk_manager)
    """

def run(self, 
        progress_callback: Optional[Callable] = None,
        save_results: bool = True
       ) -> BacktestResults:
    """
    Execute the backtest.
    
    Args:
        progress_callback: Optional callback for progress updates
            Signature: callback(processed_events: int, total_events: int) -> None
        save_results: Whether to save results to disk
        
    Returns:
        BacktestResults: Comprehensive results object
        
    Raises:
        RuntimeError: If no strategies or data handlers configured
        DataError: If data validation fails
        MemoryError: If memory limits exceeded
        
    Example:
        >>> def progress_update(processed, total):
        ...     print(f"Progress: {processed/total:.1%}")
        ...
        >>> results = engine.run(progress_callback=progress_update)
        >>> print(f"Total Return: {results.total_return:.2%}")
        >>> print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    """
```

### Strategy Framework

#### BaseStrategy

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides framework for strategy development with:
    - Technical indicator calculation
    - Position sizing utilities
    - Risk management integration
    - Performance tracking
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            symbols: List of symbols this strategy trades
            parameters: Strategy-specific parameters
            
        Example:
            >>> class MyStrategy(BaseStrategy):
            ...     def __init__(self, **kwargs):
            ...         super().__init__(**kwargs)
            ...         self.lookback = self.parameters.get('lookback', 20)
        """
        
    @abstractmethod
    def generate_signals(
        self,
        market_data: MarketDataSnapshot,
        portfolio: Portfolio
    ) -> List[SignalEvent]:
        """
        Generate trading signals based on market data and portfolio state.
        
        This is the core method that defines strategy logic.
        
        Args:
            market_data: Current market data snapshot
            portfolio: Current portfolio state
            
        Returns:
            List of SignalEvent objects representing trading decisions
            
        Example:
            >>> def generate_signals(self, market_data, portfolio):
            ...     signals = []
            ...     
            ...     for symbol in self.symbols:
            ...         if symbol not in market_data.data:
            ...             continue
            ...             
            ...         current_price = market_data.data[symbol].close_price
            ...         sma = self.calculate_sma(symbol, 20)
            ...         
            ...         if current_price > sma * 1.02:  # 2% above SMA
            ...             signal = SignalEvent(
            ...                 timestamp=market_data.timestamp,
            ...                 symbol=symbol,
            ...                 signal_type=SignalType.BUY,
            ...                 strength=0.8,
            ...                 strategy_id=self.strategy_id
            ...             )
            ...             signals.append(signal)
            ...             
            ...     return signals
        """
        
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        portfolio: Portfolio,
        risk_level: float = 0.02
    ) -> int:
        """
        Calculate appropriate position size based on risk management.
        
        Uses volatility-adjusted position sizing by default.
        
        Args:
            symbol: Symbol to calculate size for
            signal_strength: Signal confidence (0.0 to 1.0)
            portfolio: Current portfolio state
            risk_level: Maximum risk per trade as fraction of portfolio
            
        Returns:
            Position size in shares
            
        Example:
            >>> position_size = self.calculate_position_size(
            ...     symbol='AAPL',
            ...     signal_strength=0.8,
            ...     portfolio=portfolio,
            ...     risk_level=0.02  # 2% of portfolio at risk
            ... )
        """
```

#### Built-in Strategy Examples

```python
class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band-based mean reversion strategy.
    
    Entry: When price moves beyond Bollinger Bands
    Exit: When price returns to moving average
    """
    
    def __init__(
        self,
        symbols: List[str],
        lookback_period: int = 20,
        num_std: float = 2.0,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            symbols: Symbols to trade
            lookback_period: Period for moving average and volatility
            num_std: Number of standard deviations for Bollinger Bands
            entry_threshold: Z-score threshold for entry signals
            exit_threshold: Z-score threshold for exit signals
            
        Example:
            >>> strategy = MeanReversionStrategy(
            ...     symbols=['AAPL', 'MSFT'],
            ...     lookback_period=20,
            ...     entry_threshold=2.5,
            ...     exit_threshold=0.5
            ... )
        """

class MomentumStrategy(BaseStrategy):
    """
    Multi-timeframe momentum strategy with trend confirmation.
    
    Uses combination of:
    - Price momentum (rate of change)
    - Volume confirmation
    - Trend strength indicators
    """
    
    def __init__(
        self,
        symbols: List[str],
        short_period: int = 10,
        long_period: int = 50,
        momentum_threshold: float = 0.05,
        volume_multiplier: float = 1.5,
        **kwargs
    ):
        """
        Initialize momentum strategy.
        
        Args:
            symbols: Symbols to trade
            short_period: Short-term momentum period
            long_period: Long-term trend period
            momentum_threshold: Minimum momentum for signal generation
            volume_multiplier: Volume confirmation threshold
            
        Example:
            >>> strategy = MomentumStrategy(
            ...     symbols=['QQQ', 'SPY'],
            ...     short_period=10,
            ...     long_period=50,
            ...     momentum_threshold=0.03
            ... )
        """

class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy using cointegration analysis.
    
    Identifies and trades mean-reverting spread relationships
    between correlated securities.
    """
    
    def __init__(
        self,
        symbol_pairs: List[Tuple[str, str]],
        lookback_period: int = 60,
        cointegration_threshold: float = 0.05,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        **kwargs
    ):
        """
        Initialize pairs trading strategy.
        
        Args:
            symbol_pairs: List of symbol pairs to analyze
            lookback_period: Period for cointegration analysis
            cointegration_threshold: P-value threshold for cointegration test
            entry_zscore: Z-score threshold for trade entry
            exit_zscore: Z-score threshold for trade exit
            
        Example:
            >>> strategy = PairsTradingStrategy(
            ...     symbol_pairs=[('AAPL', 'MSFT'), ('JPM', 'BAC')],
            ...     lookback_period=60,
            ...     entry_zscore=2.5
            ... )
        """
```

### Risk Management API

```python
class RiskManager:
    """
    Comprehensive risk management system.
    """
    
    def __init__(self, limits: RiskLimits):
        """
        Initialize risk manager with specified limits.
        
        Args:
            limits: RiskLimits object defining portfolio constraints
            
        Example:
            >>> limits = RiskLimits(
            ...     max_position_size=0.1,        # 10% max per position
            ...     max_portfolio_leverage=2.0,    # 2x max leverage
            ...     max_drawdown=0.15,             # 15% max drawdown
            ...     max_daily_var_95=0.03,         # 3% daily VaR limit
            ...     max_correlation=0.8,           # Max pairwise correlation
            ...     sector_limits={'Technology': 0.4}  # Sector concentration
            ... )
            >>> risk_manager = RiskManager(limits)
        """
        
    def validate_order(
        self,
        order: OrderEvent,
        portfolio: Portfolio,
        market_data: MarketDataSnapshot
    ) -> RiskCheckResult:
        """
        Validate order against risk limits.
        
        Args:
            order: Proposed order to validate
            portfolio: Current portfolio state
            market_data: Current market conditions
            
        Returns:
            RiskCheckResult with validation outcome and adjustments
            
        Example:
            >>> result = risk_manager.validate_order(order, portfolio, market_data)
            >>> if result.approved:
            ...     # Execute order
            ...     broker.submit_order(result.adjusted_order or order)
            >>> else:
            ...     print(f"Order rejected: {result.rejection_reason}")
        """
        
    def calculate_portfolio_risk(
        self,
        portfolio: Portfolio,
        market_data: MarketDataSnapshot,
        confidence_level: float = 0.95
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio: Portfolio to analyze
            market_data: Current market data
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            PortfolioRiskMetrics with detailed risk analysis
            
        Example:
            >>> risk_metrics = risk_manager.calculate_portfolio_risk(
            ...     portfolio, market_data, confidence_level=0.99
            ... )
            >>> print(f"Portfolio VaR (99%): {risk_metrics.var_99:.2%}")
            >>> print(f"Expected Shortfall: {risk_metrics.expected_shortfall:.2%}")
        """

class RiskLimits:
    """Risk limit configuration object."""
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        max_portfolio_leverage: float = 2.0,
        max_drawdown: float = 0.2,
        max_daily_var_95: float = 0.05,
        max_correlation: float = 0.8,
        sector_limits: Optional[Dict[str, float]] = None,
        position_concentration_limit: float = 0.15,
        overnight_position_limit: float = 0.8
    ):
        """
        Define portfolio risk limits.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_leverage: Maximum portfolio leverage ratio
            max_drawdown: Maximum allowed drawdown before position reduction
            max_daily_var_95: Maximum daily Value-at-Risk (95% confidence)
            max_correlation: Maximum correlation between any two positions
            sector_limits: Dictionary of sector exposure limits
            position_concentration_limit: Maximum exposure to single position
            overnight_position_limit: Maximum overnight position exposure
        """
```

### Data Handling API

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union

class BaseDataHandler(ABC):
    """
    Abstract base class for data handlers.
    
    Supports multiple data sources: CSV, databases, APIs, live feeds.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Initialize data handler.
        
        Args:
            symbols: List of symbols to load
            start_date: Data start date (None for all available)
            end_date: Data end date (None for all available)
        """
        
    @abstractmethod
    def get_data_iterator(self) -> Iterator[MarketEvent]:
        """
        Return iterator over market events in chronological order.
        
        Yields:
            MarketEvent: Individual market data points
            
        Example:
            >>> for market_event in data_handler.get_data_iterator():
            ...     print(f"{market_event.symbol}: ${market_event.close_price}")
        """
        
    def validate_data(self) -> DataValidationReport:
        """
        Validate data quality and completeness.
        
        Returns:
            DataValidationReport with validation results
            
        Example:
            >>> report = data_handler.validate_data()
            >>> if not report.is_valid:
            ...     print("Data issues found:")
            ...     for issue in report.issues:
            ...         print(f"  - {issue}")
        """

class CSVDataHandler(BaseDataHandler):
    """
    CSV file data handler with advanced parsing capabilities.
    """
    
    def __init__(
        self,
        symbols: List[str],
        data_directory: str,
        file_pattern: str = "{symbol}.csv",
        date_column: str = "date",
        price_columns: Optional[Dict[str, str]] = None,
        volume_column: str = "volume",
        **kwargs
    ):
        """
        Initialize CSV data handler.
        
        Args:
            symbols: Symbols to load
            data_directory: Directory containing CSV files
            file_pattern: File naming pattern (use {symbol} placeholder)
            date_column: Name of date column
            price_columns: Mapping of OHLC columns
            volume_column: Name of volume column
            
        Example:
            >>> handler = CSVDataHandler(
            ...     symbols=['AAPL', 'MSFT'],
            ...     data_directory='./market_data/',
            ...     file_pattern="{symbol}_daily.csv",
            ...     price_columns={
            ...         'open': 'Open',
            ...         'high': 'High', 
            ...         'low': 'Low',
            ...         'close': 'Close'
            ...     }
            ... )
        """

class DatabaseDataHandler(BaseDataHandler):
    """
    Database data handler with connection pooling and optimization.
    """
    
    def __init__(
        self,
        symbols: List[str],
        connection_string: str,
        table_name: str = "market_data",
        query_batch_size: int = 10000,
        **kwargs
    ):
        """
        Initialize database data handler.
        
        Args:
            symbols: Symbols to load
            connection_string: Database connection string
            table_name: Name of market data table
            query_batch_size: Number of records per query batch
            
        Example:
            >>> handler = DatabaseDataHandler(
            ...     symbols=['AAPL', 'MSFT'],
            ...     connection_string='postgresql://user:pass@localhost/marketdata',
            ...     table_name='daily_prices'
            ... )
        """
```

### Performance Analysis API

```python
class PerformanceAnalyzer:
    """
    Comprehensive performance analysis with statistical rigor.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        
    def analyze_results(
        self,
        backtest_results: BacktestResults,
        benchmark_returns: Optional[pd.Series] = None,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> PerformanceReport:
        """
        Generate comprehensive performance analysis.
        
        Args:
            backtest_results: Results from backtest execution
            benchmark_returns: Optional benchmark for comparison
            confidence_levels: Confidence levels for VaR/CVaR calculation
            
        Returns:
            PerformanceReport with detailed metrics and analysis
            
        Example:
            >>> analyzer = PerformanceAnalyzer(risk_free_rate=0.025)
            >>> report = analyzer.analyze_results(
            ...     backtest_results,
            ...     benchmark_returns=spy_returns
            ... )
            >>> 
            >>> print("Performance Summary:")
            >>> print(f"Total Return: {report.total_return:.2%}")
            >>> print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
            >>> print(f"Information Ratio: {report.information_ratio:.2f}")
            >>> print(f"Maximum Drawdown: {report.max_drawdown:.2%}")
        """
        
    def generate_html_report(
        self,
        performance_report: PerformanceReport,
        output_path: str,
        include_charts: bool = True
    ) -> str:
        """
        Generate interactive HTML performance report.
        
        Args:
            performance_report: Performance analysis results
            output_path: Path for output HTML file
            include_charts: Whether to include interactive charts
            
        Returns:
            Path to generated HTML file
            
        Example:
            >>> report_path = analyzer.generate_html_report(
            ...     performance_report,
            ...     'results/backtest_report.html',
            ...     include_charts=True
            ... )
            >>> print(f"Report generated: {report_path}")
        """
```

### Error Handling

```python
class BacktestingError(Exception):
    """Base exception for backtesting engine errors."""
    pass

class DataValidationError(BacktestingError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, invalid_records: List[Dict] = None):
        super().__init__(message)
        self.invalid_records = invalid_records or []

class InsufficientDataError(BacktestingError):
    """Raised when insufficient data for analysis."""
    
    def __init__(self, symbol: str, required_periods: int, available_periods: int):
        message = (f"Insufficient data for {symbol}: "
                  f"required {required_periods}, available {available_periods}")
        super().__init__(message)
        self.symbol = symbol
        self.required_periods = required_periods
        self.available_periods = available_periods

class RiskLimitExceededError(BacktestingError):
    """Raised when risk limits are exceeded."""
    
    def __init__(self, limit_type: str, current_value: float, limit_value: float):
        message = (f"{limit_type} limit exceeded: "
                  f"{current_value:.2%} > {limit_value:.2%}")
        super().__init__(message)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value

# Example error handling:
try:
    results = engine.run()
except InsufficientDataError as e:
    print(f"Data issue: {e}")
    print(f"Consider reducing lookback period or adding more data for {e.symbol}")
except RiskLimitExceededError as e:
    print(f"Risk management: {e}")
    print("Consider adjusting position sizes or risk limits")
except DataValidationError as e:
    print(f"Data validation failed: {e}")
    if e.invalid_records:
        print(f"Found {len(e.invalid_records)} invalid records")
```

### Configuration API

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

@dataclass
class BacktestConfig:
    """Configuration object for backtesting engine."""
    
    # Time range
    start_date: str
    end_date: str
    
    # Capital and risk
    initial_capital: float = 1_000_000
    max_leverage: float = 2.0
    
    # Execution
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    # Data
    data_source: str = 'csv'
    data_directory: str = './data/'
    
    # Strategy parameters
    strategy_config: Dict[str, Any] = None
    
    # Risk management
    risk_limits: Dict[str, float] = None
    
    # Performance
    benchmark: Optional[str] = None
    risk_free_rate: float = 0.02
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'BacktestConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

# Example configuration usage:
config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1_000_000,
    strategy_config={
        'type': 'MeanReversion',
        'symbols': ['AAPL', 'MSFT'],
        'lookback_period': 20,
        'entry_threshold': 2.0
    },
    risk_limits={
        'max_position_size': 0.1,
        'max_drawdown': 0.15
    }
)

# Save for reuse
config.to_yaml('configs/my_strategy.yaml')

# Load and use
loaded_config = BacktestConfig.from_yaml('configs/my_strategy.yaml')
engine = BacktestEngine(**loaded_config.__dict__)
```

## Integration Examples

### Command Line Interface

```bash
# Run backtest from configuration file
backtest run --config configs/mean_reversion.yaml --output results/

# Run parameter sweep
backtest sweep --config configs/optimization.yaml --parameters lookback:10,20,30 entry_threshold:1.5,2.0,2.5

# Generate performance report
backtest report --results results/backtest_20240101.pkl --benchmark SPY --output reports/

# Validate data integrity
backtest validate-data --directory data/ --symbols AAPL,MSFT,GOOGL
```

### Jupyter Notebook Integration

```python
# Notebook-friendly progress tracking
from IPython.display import clear_output
import matplotlib.pyplot as plt

def notebook_progress_callback(processed, total):
    clear_output(wait=True)
    progress = processed / total
    
    # Progress bar
    bar_length = 50
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"Progress: {bar} {progress:.1%} ({processed:,}/{total:,})")

# Run backtest with progress tracking
results = engine.run(progress_callback=notebook_progress_callback)

# Inline plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
results.plot_equity_curve(ax=axes[0,0])
results.plot_drawdown(ax=axes[0,1])
results.plot_returns_distribution(ax=axes[1,0])
results.plot_rolling_sharpe(ax=axes[1,1])
plt.tight_layout()
plt.show()
```
