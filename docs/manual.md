## Getting Started Guide

### Step 1: Clone from GitHub
```bash
# Clone the repository
git clone https://github.com/PriscillaOng12/backtesting-engine.git
cd backtesting-engine

# Create virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Step 2: Get Sample Data

```bash
# The repository includes sample data
ls data/  # Check available sample datasets

# Or download your own data (CSV format expected):
# Date,Open,High,Low,Close,Volume
# 2020-01-02,74.06,75.15,73.80,75.09,135480400
```

### Step 3: Choose How to Try This Backtesting Engine

#### **Option A: Interactive Tutorial (Jupyter Notebook)**

```bash
# Start Jupyter notebook
jupyter notebook

# Open the comprehensive tutorial
# Navigate to: examples/notebooks/tutorial.ipynb
```

The tutorial covers:
- Framework overview and architecture
- Loading and validating market data  
- Creating your first trading strategy
- Running backtests with different configurations
- Analyzing results and generating reports
- Advanced features and customization

#### **Option B: Quick Script Examples**

##### Example 1: Run Pre-built Strategy
```bash
# Run the simple example strategy
python examples/simple_strategy.py

# Expected output:
# ✅ Backtest completed successfully!
# 📈 Total Return: 18.5%
# 📊 Sharpe Ratio: 1.42
# 📉 Max Drawdown: -8.3%
```

##### Example 2: Create Your Own Strategy
```python
# Create file: my_strategy.py
from backtesting_engine import BacktestEngine
from backtesting_engine.strategies import MeanReversionStrategy
from backtesting_engine.data import CSVDataHandler

# 1. Setup data handler
data_handler = CSVDataHandler(
    symbols=['AAPL'],
    data_directory='examples/data/',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 2. Create strategy
strategy = MeanReversionStrategy(
    strategy_id='my_mean_reversion',
    symbols=['AAPL'],
    lookback_period=20,        # 20-day moving average
    entry_threshold=2.0,       # Enter when 2 std devs from mean
    exit_threshold=0.5,        # Exit when 0.5 std devs from mean
    position_size=0.1          # 10% of portfolio per position
)

# 3. Configure backtest engine
engine = BacktestEngine(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000,    # $100K starting capital
    commission_rate=0.001,     # 0.1% commission
    slippage_rate=0.0005      # 0.05% slippage
)

# 4. Add components and run
engine.add_data_handler(data_handler)
engine.add_strategy('strategy_1', strategy)

print("Running backtest...")
results = engine.run()

# 5. View results
print(f"Total Return: {results.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results.metrics['max_drawdown']:.2%}")

# 6. Generate detailed report
results.generate_html_report('my_backtest_results.html')
print("📊 Detailed report saved to: my_backtest_results.html")
```

##### Example 3: Multi-Strategy Setup
```python
# Create file: advanced_example.py
from backtesting_engine import BacktestEngine
from backtesting_engine.strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    PairsTradingStrategy
)
from backtesting_engine.risk import RiskManager, RiskLimits

# Setup with multiple strategies and risk management
engine = BacktestEngine(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1000000
)

# Add multiple strategies
strategies = {
    'mean_reversion': MeanReversionStrategy(['AAPL', 'MSFT']),
    'momentum': MomentumStrategy(['GOOGL', 'AMZN']),
    'pairs_trading': PairsTradingStrategy([('AAPL', 'MSFT')])
}

for name, strategy in strategies.items():
    engine.add_strategy(name, strategy)

# Add risk management
risk_limits = RiskLimits(
    max_position_size=0.1,     # Max 10% per position
    max_portfolio_leverage=2.0, # Max 2x leverage
    max_drawdown=0.15,         # Stop if 15% drawdown
    max_daily_var=0.03         # Max 3% daily VaR
)
engine.set_risk_manager(RiskManager(risk_limits))

# Run and analyze
results = engine.run()
print("Multi-strategy backtest completed!")
```

### Step 4: Understanding the Output

#### Console Output
```
🚀 Starting backtest...
📊 Loading data for AAPL (2020-01-01 to 2023-12-31)
✅ Loaded 1,008 trading days
🎯 Initializing Mean Reversion Strategy
⚡ Processing 1,008 market events...
✅ Backtest completed in 2.3 seconds

📈 PERFORMANCE SUMMARY:
├── Total Return: +18.5%
├── Annualized Return: +4.3%
├── Sharpe Ratio: 1.42
├── Maximum Drawdown: -8.3%
├── Win Rate: 64.2%
└── Total Trades: 127

💾 Results saved to: results/backtest_20240729_143022.pkl
📊 HTML report: results/backtest_report.html
```

#### HTML Report Contents
- 📈 **Equity Curve**: Portfolio value over time
- 📉 **Drawdown Chart**: Risk visualization
- 📊 **Returns Distribution**: Statistical analysis
- 🎯 **Trade Analysis**: Win/loss breakdown
- 📋 **Performance Metrics**: 20+ key statistics
- 🔍 **Monthly/Yearly Breakdown**: Detailed performance

### Step 5: Customize Strategy (Optional)

#### Customize Your Strategy
```python
# Create custom strategy by inheriting from BaseStrategy
from backtesting_engine.strategies import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_parameter = kwargs.get('my_parameter', 10)
    
    def generate_signals(self, market_data, portfolio):
        # Your custom trading logic here
        signals = []
        
        for symbol in self.symbols:
            # Example: Simple price-based signal
            current_price = market_data.get_latest_price(symbol)
            sma = self.calculate_sma(symbol, self.my_parameter)
            
            if current_price > sma * 1.02:  # Buy signal
                signal = self.create_signal(symbol, 'BUY', strength=0.8)
                signals.append(signal)
                
        return signals
```
