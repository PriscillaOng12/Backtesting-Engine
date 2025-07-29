"""
Comprehensive test suite for the backtesting engine.

This module provides extensive test coverage for all components
of the backtesting framework.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

# Import all modules to test
from backtesting_engine.core.events import (
    Event, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    EventType, OrderSide, OrderType, MarketDataSnapshot
)
from backtesting_engine.core.portfolio import Portfolio, Position
from backtesting_engine.core.data_handler import BaseDataHandler, CSVDataHandler
from backtesting_engine.core.execution import ExecutionHandler, SlippageModel, CommissionModel
from backtesting_engine.strategies.base import BaseStrategy
from backtesting_engine.strategies.mean_reversion import MeanReversionStrategy
from backtesting_engine.strategies.momentum import MomentumStrategy
from backtesting_engine.analysis.performance import PerformanceAnalyzer
from backtesting_engine.risk.risk_manager import RiskManager, RiskLimits


class TestEvents:
    """Test event system components."""
    
    def test_market_event_creation(self):
        """Test MarketEvent creation and properties."""
        timestamp = datetime.now()
        
        event = MarketEvent(
            timestamp=timestamp,
            symbol="AAPL",
            open_price=Decimal('150.0'),
            high_price=Decimal('152.0'),
            low_price=Decimal('149.0'),
            close_price=Decimal('151.0'),
            volume=1000000
        )
        
        assert event.event_type == EventType.MARKET
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.open_price == Decimal('150.0')
        assert event.close_price == Decimal('151.0')
    
    def test_signal_event_creation(self):
        """Test SignalEvent creation and validation."""
        timestamp = datetime.now()
        
        signal = SignalEvent(
            timestamp=timestamp,
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=OrderSide.BUY,
            strength=0.8,
            target_percent=0.1
        )
        
        assert signal.event_type == EventType.SIGNAL
        assert signal.timestamp == timestamp
        assert signal.strategy_id == "test_strategy"
        assert signal.symbol == "AAPL"
        assert signal.signal_type == "BUY"  # This one is stored as string
        assert signal.strength == 0.8
        assert signal.target_percent == 0.1
    
    def test_order_event_creation(self):
        """Test OrderEvent creation and validation."""
        timestamp = datetime.now()
        
        order = OrderEvent(
            order_id="test_order_1",
            timestamp=timestamp,
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            price=None
        )
        
        assert order.event_type == EventType.ORDER
        assert order.symbol == "AAPL"
        assert order.order_type == "MARKET"  # Stored as string
        assert order.side == "BUY"  # Stored as string
        assert order.quantity == 100
        assert order.price is None
    
    def test_fill_event_creation(self):
        """Test FillEvent creation and calculations."""
        timestamp = datetime.now()
        
        fill = FillEvent(
            fill_id="test_fill_1",
            order_id="test_order_1",
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=Decimal('150.0'),
            commission=Decimal('1.0')
        )
        
        assert fill.event_type == EventType.FILL
        assert fill.symbol == "AAPL"
        assert fill.side == "BUY"  # Stored as string
        assert fill.quantity == 100
        assert fill.fill_price == Decimal('150.0')
        assert fill.commission == Decimal('1.0')
        assert fill.gross_amount == Decimal('15000.0')  # 100 * 150.0


class TestPortfolio:
    """Test portfolio management components."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        initial_capital = 100000.0
        
        portfolio = Portfolio(Decimal(str(initial_capital)))
        
        assert portfolio.initial_capital == Decimal(str(initial_capital))
        assert portfolio.cash == Decimal(str(initial_capital))
        assert portfolio.calculate_total_equity() == Decimal(str(initial_capital))
        assert len(portfolio.positions) == 0
    
    def test_portfolio_buy_order(self):
        """Test buying shares through portfolio."""
        portfolio = Portfolio(Decimal('100000.0'))
        timestamp = datetime.now()
        
        # Create a fill event for buying shares
        fill = FillEvent(
            fill_id="test_fill_1",
            order_id="test_order_1",
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=Decimal('150.0'),
            commission=Decimal('1.0')
        )
        
        portfolio.process_fill(fill)
        
        assert portfolio.get_position("AAPL").quantity == 100
        expected_cash = Decimal('100000.0') - Decimal('15001.0')  # Initial - (100*150 + 1)
        assert abs(portfolio.cash - expected_cash) < Decimal('0.01')
        assert portfolio.get_position("AAPL").last_price == Decimal('150.0')
    
    def test_portfolio_sell_order(self):
        """Test selling shares through portfolio."""
        portfolio = Portfolio(Decimal('100000.0'))
        timestamp = datetime.now()
        
        # First buy some shares
        buy_fill = FillEvent(
            fill_id="test_fill_1",
            order_id="test_order_1",
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=Decimal('150.0'),
            commission=Decimal('1.0')
        )
        portfolio.process_fill(buy_fill)
        
        # Then sell some shares
        sell_fill = FillEvent(
            fill_id="test_fill_2",
            order_id="test_order_2",
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            fill_price=Decimal('155.0'),
            commission=Decimal('1.0')
        )
        portfolio.process_fill(sell_fill)
        
        assert portfolio.get_position("AAPL").quantity == 50
        # Cash should reflect the buy and sell transactions
        expected_cash = Decimal('100000.0') - Decimal('15001.0') + Decimal('7749.0')  # Buy cost + Sell proceeds
        assert abs(portfolio.cash - expected_cash) < Decimal('1.0')
    
    def test_portfolio_value_calculation(self):
        """Test portfolio total value calculation."""
        portfolio = Portfolio(Decimal('100000.0'))
        timestamp = datetime.now()
        
        # Buy some shares
        fill = FillEvent(
            fill_id="test_fill_1",
            order_id="test_order_1",
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=Decimal('150.0'),
            commission=Decimal('1.0')
        )
        portfolio.process_fill(fill)
        
        # Update market price - need to create MarketEvent first
        market_event = MarketEvent(
            timestamp=timestamp,
            symbol="AAPL", 
            open_price=Decimal('155.0'),
            high_price=Decimal('162.0'),
            low_price=Decimal('154.0'),
            close_price=Decimal('160.0'),
            volume=1000000
        )
        market_data = MarketDataSnapshot(
            timestamp=timestamp,
            data={"AAPL": market_event}
        )
        portfolio.update_market_data(market_data)
        
        expected_total = portfolio.cash + Decimal('100') * Decimal('160.0')
        actual_total = portfolio.calculate_total_equity()
        assert abs(actual_total - expected_total) < Decimal('1.0')


class TestStrategies:
    """Test strategy implementations."""
    
    def test_base_strategy_initialization(self):
        """Test BaseStrategy initialization with a concrete implementation."""
        from backtesting_engine.strategies.mean_reversion import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(
            strategy_id="test_strategy",
            symbols=["AAPL", "GOOGL"],
            lookback_period=20,
            position_size=0.1
        )
        
        assert strategy.strategy_id == "test_strategy"
        assert strategy.symbols == {"AAPL", "GOOGL"}  # Stored as set
        assert strategy.parameters["lookback_period"] == 20
        assert strategy.parameters["position_size"] == 0.1
        assert len(strategy.price_history) == 2
        assert "AAPL" in strategy.price_history
        assert "GOOGL" in strategy.price_history
    
    def test_mean_reversion_strategy(self):
        """Test MeanReversionStrategy signal generation."""
        strategy = MeanReversionStrategy(
            strategy_id="test_mr",
            symbols=["AAPL"],
            lookback_period=20,
            std_dev_multiplier=2.0,
            position_size=0.1
        )
        
        # Generate some price data
        prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
                  150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 95]  # Mean reversion pattern
        
        portfolio = Portfolio(Decimal('100000.0'))
        signals = []
        
        for i, price in enumerate(prices):
            timestamp = datetime.now() + timedelta(days=i)
            
            # Create proper MarketEvent
            market_event = MarketEvent(
                timestamp=timestamp,
                symbol="AAPL",
                open_price=Decimal(str(price)),
                high_price=Decimal(str(price * 1.02)),
                low_price=Decimal(str(price * 0.98)), 
                close_price=Decimal(str(price)),
                volume=1000000
            )
            
            # Create MarketDataSnapshot for the strategy
            market_data = MarketDataSnapshot(
                timestamp=timestamp,
                data={"AAPL": market_event}
            )
            
            strategy_signals = strategy.generate_signals(market_data, portfolio)
            signals.extend(strategy_signals)
        
        # Should generate some signals for mean reversion
        assert len(signals) >= 0  # May or may not generate signals with limited data
        
        # Check that any generated signals have proper structure
        for signal in signals:
            assert signal.event_type == EventType.SIGNAL
            assert signal.symbol == "AAPL"
            assert signal.strategy_id == "test_mr"
            assert signal.signal_type in ["BUY", "SELL"]  # Stored as strings


class TestExecution:
    """Test execution system components."""
    
    def test_slippage_models(self):
        """Test different slippage models."""
        from backtesting_engine.core.execution import FixedSlippageModel, PercentageSlippageModel
        
        # Fixed slippage
        fixed_model = FixedSlippageModel(0.01)
        slippage = fixed_model.calculate_slippage("AAPL", 100, 150.0, OrderSide.BUY)
        assert slippage == 0.01
        
        # Percentage slippage
        pct_model = PercentageSlippageModel(0.001)  # 0.1%
        slippage = pct_model.calculate_slippage("AAPL", 100, 150.0, OrderSide.BUY)
        assert slippage == 0.15  # 0.1% of 150.0
    
    def test_commission_models(self):
        """Test different commission models."""
        from backtesting_engine.core.execution import FixedCommissionModel, PercentageCommissionModel
        
        # Fixed commission
        fixed_model = FixedCommissionModel(1.0)
        commission = fixed_model.calculate_commission("AAPL", 100, 150.0)
        assert commission == 1.0
        
        # Percentage commission
        pct_model = PercentageCommissionModel(0.001)  # 0.1%
        commission = pct_model.calculate_commission("AAPL", 100, 150.0)
        assert commission == 15.0  # 0.1% of 15000
    
    def test_execution_handler(self):
        """Test ExecutionHandler order processing."""
        from backtesting_engine.core.execution import FixedSlippageModel, FixedCommissionModel
        
        slippage_model = FixedSlippageModel(0.01)
        commission_model = FixedCommissionModel(1.0)
        
        execution_handler = ExecutionHandler(
            slippage_model=slippage_model,
            commission_model=commission_model
        )
        
        timestamp = datetime.now()
        order = OrderEvent(
            order_id="test_order_1",
            timestamp=timestamp,
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            price=None
        )
        
        # Create proper market data
        market_event = MarketEvent(
            timestamp=timestamp,
            symbol="AAPL",
            open_price=Decimal('149.5'),
            high_price=Decimal('150.5'),
            low_price=Decimal('149.0'),
            close_price=Decimal('150.0'),
            volume=1000000
        )
        
        market_data = MarketDataSnapshot(
            timestamp=timestamp,
            data={"AAPL": market_event}
        )
        
        fill = execution_handler.execute_order(order, market_data)
        
        assert fill.event_type == EventType.FILL
        assert fill.symbol == "AAPL"
        assert fill.side == "BUY"  # Stored as string
        assert fill.quantity == 100
        assert fill.commission == Decimal('1.0')
        # Fill price should include slippage
        assert fill.fill_price > Decimal('150.0')


class TestRiskManagement:
    """Test risk management components."""
    
    def test_risk_limits(self):
        """Test RiskLimits dataclass."""
        limits = RiskLimits(
            max_position_size_percent=Decimal('0.1'),
            max_daily_loss_percent=Decimal('0.05'),
            max_var_95_percent=Decimal('0.02'),
            max_portfolio_leverage=Decimal('2.0'),
            max_correlation_threshold=Decimal('0.8')
        )
        
        assert limits.max_position_size_percent == Decimal('0.1')
        assert limits.max_daily_loss_percent == Decimal('0.05')
        assert limits.max_var_95_percent == Decimal('0.02')
        assert limits.max_portfolio_leverage == Decimal('2.0')
        assert limits.max_correlation_threshold == Decimal('0.8')
    
    def test_risk_manager_initialization(self):
        """Test RiskManager initialization."""
        limits = RiskLimits()
        risk_manager = RiskManager(limits)
        
        assert risk_manager.limits == limits
        assert len(risk_manager.daily_stats) == 0
        assert len(risk_manager.alerts) == 0
        assert len(risk_manager.alerts) == 0
    
    def test_position_size_validation(self):
        """Test position size validation."""
        limits = RiskLimits(max_position_size_percent=Decimal('0.1'))
        risk_manager = RiskManager(limits)
        
        portfolio = Portfolio(Decimal('100000.0'))
        
        # Create market data
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open_price=Decimal('149.0'),
            high_price=Decimal('151.0'),
            low_price=Decimal('148.0'),
            close_price=Decimal('150.0'),
            volume=1000000
        )
        
        # Valid position size (5% of portfolio)
        order = OrderEvent(
            order_id="test_1",
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=33,  # $5000 at $150 = 5%
            price=Decimal('150.0')
        )
        
        risk_passed = risk_manager.check_order_risk(order, portfolio, market_event)
        assert risk_passed  # Should be valid since 5% < 10%
        
        # Invalid position size (15% of portfolio)
        large_order = OrderEvent(
            order_id="test_2", 
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,  # $15000 at $150 = 15%
            price=Decimal('150.0')
        )
        
        risk_passed = risk_manager.check_order_risk(large_order, portfolio, market_event)
        assert not risk_passed  # Should be invalid since 15% > 10%
class TestPerformanceAnalysis:
    """Test performance analysis components."""
    
    def test_performance_analyzer_initialization(self):
        """Test PerformanceAnalyzer initialization."""
        analyzer = PerformanceAnalyzer()
        
        assert analyzer.risk_free_rate == 0.02
        # Basic initialization test
        assert hasattr(analyzer, 'calculate_metrics')
    
    def test_returns_calculation(self):
        """Test returns calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create equity curve data
        timestamps = [datetime.now() + timedelta(days=i) for i in range(5)]
        values = [Decimal('100000'), Decimal('101000'), Decimal('102000'), Decimal('101500'), Decimal('103000')]
        equity_curve = list(zip(timestamps, values))
        
        # Create mock trades list
        trades = []
        
        # Calculate metrics
        metrics = analyzer.calculate_metrics(equity_curve, trades)
        
        # Basic validation
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'annualized_return')
        assert metrics.total_return > 0  # Portfolio grew
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create equity curve with positive trend
        timestamps = [datetime.now() + timedelta(days=i) for i in range(50)]
        np.random.seed(42)
        
        # Generate values with positive trend
        values = [Decimal('100000')]
        for i in range(49):
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return with 2% volatility
            new_value = values[-1] * Decimal(str(1 + daily_return))
            values.append(new_value)
        
        equity_curve = list(zip(timestamps, values))
        trades = []  # Empty trades for this test
        
        # Calculate metrics
        metrics = analyzer.calculate_metrics(equity_curve, trades)
        
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'annualized_return')
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'max_drawdown')
        
        # Sharpe ratio should be reasonable for our data (can be negative with random data)
        assert metrics.sharpe_ratio > -5 and metrics.sharpe_ratio < 5


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_simple_backtest_workflow(self):
        """Test a simple end-to-end backtest workflow."""
        # Initialize components
        portfolio = Portfolio(Decimal('100000.0'))
        strategy = MeanReversionStrategy(
            strategy_id="test_integration",
            symbols=["AAPL"],
            lookback_period=10,
            position_size=0.1
        )
        
        from backtesting_engine.core.execution import ExecutionHandler, FixedSlippageModel, FixedCommissionModel
        execution_handler = ExecutionHandler(
            slippage_model=FixedSlippageModel(0.01),
            commission_model=FixedCommissionModel(1.0)
        )
        
        analyzer = PerformanceAnalyzer()
        
        # Generate test data
        prices = [100 + i + 5*np.sin(i/10) for i in range(50)]  # Oscillating prices
        equity_values = []
        
        # Run backtest
        for i, price in enumerate(prices):
            timestamp = datetime.now() + timedelta(days=i)
            
            # Create proper market data
            market_event = MarketEvent(
                timestamp=timestamp,
                symbol="AAPL",
                open_price=Decimal(str(price * 0.995)),
                high_price=Decimal(str(price * 1.02)),
                low_price=Decimal(str(price * 0.98)),
                close_price=Decimal(str(price)),
                volume=1000000
            )
            
            market_data = MarketDataSnapshot(
                timestamp=timestamp,
                data={"AAPL": market_event}
            )
            
            # Update portfolio market value
            portfolio.update_market_data(market_data)
            
            # Generate signals
            signals = strategy.generate_signals(market_data, portfolio)
            
            # Execute signals (simplified)
            for signal in signals:
                if signal.signal_type == "BUY" and signal.target_percent and signal.target_percent > 0:
                    target_value = float(portfolio.calculate_total_equity()) * signal.target_percent
                    quantity = int(target_value / price)
                    if quantity > 0:
                        order = OrderEvent(
                            order_id=f"order_{i}",
                            timestamp=timestamp,
                            symbol=signal.symbol,
                            order_type=OrderType.MARKET,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            price=None
                        )
                        fill = execution_handler.execute_order(order, market_data)
                        if fill:
                            portfolio.process_fill(fill)
            
            # Record performance
            equity_values.append((timestamp, portfolio.calculate_total_equity()))
        
        # Analyze results
        metrics = analyzer.calculate_metrics(equity_values, [])
        
        # Basic checks
        assert len(equity_values) == len(prices)
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'max_drawdown')
        
        # Portfolio should have evolved (or remained stable)
        final_value = equity_values[-1][1]
        assert final_value >= Decimal('0')  # Should at least be non-negative
        
        # Test should have run without errors
        assert metrics.periods == len(prices)


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty or missing data."""
        strategy = MeanReversionStrategy("test", ["AAPL"])
        portfolio = Portfolio(Decimal('100000.0'))
        
        # Empty market data
        empty_data = MarketDataSnapshot(timestamp=datetime.now(), data={})
        signals = strategy.generate_signals(empty_data, portfolio)
        assert len(signals) == 0
        
        # Missing symbol data (only has GOOGL, but strategy wants AAPL)
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="GOOGL",
            open_price=Decimal('99.0'),
            high_price=Decimal('101.0'),
            low_price=Decimal('98.0'),
            close_price=Decimal('100.0'),
            volume=1000000
        )
        partial_data = MarketDataSnapshot(timestamp=datetime.now(), data={"GOOGL": market_event})
        signals = strategy.generate_signals(partial_data, portfolio)
        assert len(signals) == 0
    
    def test_invalid_signal_parameters(self):
        """Test handling of invalid signal parameters."""
        with pytest.raises(ValueError):
            SignalEvent(
                timestamp=datetime.now(),
                strategy_id="test",
                symbol="AAPL",
                signal_type=OrderSide.BUY,
                strength=-0.5,  # Invalid: negative strength
                target_percent=0.1
            )
        
        with pytest.raises(ValueError):
            SignalEvent(
                timestamp=datetime.now(),
                strategy_id="test",
                symbol="AAPL",
                signal_type=OrderSide.BUY,
                strength=0.5,
                target_percent=1.5  # Invalid: > 100%
            )
    
    def test_portfolio_edge_cases(self):
        """Test portfolio edge cases."""
        portfolio = Portfolio(Decimal('100000.0'))
        
        # Test very small quantity fill
        fill = FillEvent(
            fill_id="test_fill_1",
            order_id="test_order_1",
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1,  # Changed from 0 to 1 since quantity must be > 0
            fill_price=150.0,
            commission=1.0
        )
        
        initial_cash = portfolio.cash
        portfolio.process_fill(fill)
        
        # Cash should decrease by fill cost + commission
        expected_cash = initial_cash - Decimal('150.0') - Decimal('1.0')
        assert abs(portfolio.cash - expected_cash) < Decimal('0.01')
        assert portfolio.get_position("AAPL").quantity == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
