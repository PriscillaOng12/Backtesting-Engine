"""
Basic tests for the backtesting engine.

These tests verify the core functionality of the backtesting engine.
"""

import pytest
from datetime import datetime
from decimal import Decimal
import tempfile
import os

from backtesting_engine.core.events import (
    MarketEvent, SignalEvent, OrderEvent, FillEvent,
    OrderType, OrderSide, EventType
)
from backtesting_engine.core.portfolio import Portfolio, Position
from backtesting_engine.execution.commissions import PercentageCommissionModel
from backtesting_engine.execution.slippage import FixedSlippageModel


class TestEvents:
    """Test event system."""
    
    def test_market_event_creation(self):
        """Test creating a market event."""
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open_price=Decimal('150.00'),
            high_price=Decimal('152.00'),
            low_price=Decimal('149.00'),
            close_price=Decimal('151.00'),
            volume=1000000
        )
        
        assert event.event_type == EventType.MARKET
        assert event.symbol == "AAPL"
        assert event.close_price == Decimal('151.00')
    
    def test_signal_event_creation(self):
        """Test creating a signal event."""
        event = SignalEvent(
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=OrderSide.BUY,
            strength=0.8,
            target_percent=0.05
        )
        
        assert event.event_type == EventType.SIGNAL
        assert event.signal_type == OrderSide.BUY
        assert event.strength == 0.8
    
    def test_order_event_creation(self):
        """Test creating an order event."""
        event = OrderEvent(
            timestamp=datetime.now(),
            order_id="order_123",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100
        )
        
        assert event.event_type == EventType.ORDER
        assert event.order_type == OrderType.MARKET
        assert event.quantity == 100
    
    def test_fill_event_creation(self):
        """Test creating a fill event."""
        event = FillEvent(
            timestamp=datetime.now(),
            fill_id="fill_123",
            order_id="order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=Decimal('151.00'),
            commission=Decimal('0.15')
        )
        
        assert event.event_type == EventType.FILL
        assert event.gross_amount == Decimal('15100.00')
        assert event.net_amount == Decimal('-15100.15')  # Buy order


class TestPortfolio:
    """Test portfolio management."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(
            initial_capital=Decimal('1000000'),
            margin_requirement=Decimal('0.5'),
            max_leverage=Decimal('2.0')
        )
        
        assert portfolio.cash == Decimal('1000000')
        assert portfolio.calculate_total_equity() == Decimal('1000000')
        assert len(portfolio.positions) == 0
    
    def test_position_creation(self):
        """Test position creation and updates."""
        position = Position(symbol="AAPL")
        
        # Test initial state
        assert position.quantity == 0
        assert position.is_flat
        assert not position.is_long
        assert not position.is_short
        
        # Test fill processing
        fill = FillEvent(
            timestamp=datetime.now(),
            fill_id="fill_1",
            order_id="order_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=Decimal('150.00'),
            commission=Decimal('1.00')
        )
        
        realized_pnl = position.add_fill(fill)
        
        assert position.quantity == 100
        assert position.is_long
        assert position.avg_price == Decimal('150.01')  # Including commission
        assert realized_pnl == Decimal('0')  # No realized P&L on first trade
    
    def test_portfolio_fill_processing(self):
        """Test portfolio processing fills."""
        portfolio = Portfolio(initial_capital=Decimal('1000000'))
        
        fill = FillEvent(
            timestamp=datetime.now(),
            fill_id="fill_1",
            order_id="order_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=Decimal('150.00'),
            commission=Decimal('1.00')
        )
        
        initial_cash = portfolio.cash
        portfolio.process_fill(fill)
        
        # Check cash reduction
        expected_cash = initial_cash - (Decimal('15000.00') + Decimal('1.00'))
        assert portfolio.cash == expected_cash
        
        # Check position creation
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100


class TestCommissions:
    """Test commission models."""
    
    def test_percentage_commission(self):
        """Test percentage-based commission calculation."""
        model = PercentageCommissionModel(
            commission_rate=Decimal('0.001'),
            min_commission=Decimal('1.00')
        )
        
        # Create dummy order
        order = OrderEvent(
            timestamp=datetime.now(),
            order_id="test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100
        )
        
        # Test commission calculation
        commission = model.calculate_commission(
            order, Decimal('150.00'), 100
        )
        
        # 100 shares * $150 * 0.001 = $15.00
        expected = Decimal('15.00')
        assert commission == expected
        
        # Test minimum commission
        small_commission = model.calculate_commission(
            order, Decimal('1.00'), 100
        )
        assert small_commission == Decimal('1.00')  # Minimum applies


class TestSlippage:
    """Test slippage models."""
    
    def test_fixed_slippage(self):
        """Test fixed slippage model."""
        model = FixedSlippageModel(
            slippage_amount=Decimal('0.01'),
            is_percentage=True
        )
        
        # Create test data
        order = OrderEvent(
            timestamp=datetime.now(),
            order_id="test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100
        )
        
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open_price=Decimal('150.00'),
            high_price=Decimal('152.00'),
            low_price=Decimal('149.00'),
            close_price=Decimal('151.00'),
            volume=1000000
        )
        
        from backtesting_engine.core.events import MarketDataSnapshot
        snapshot = MarketDataSnapshot(
            timestamp=datetime.now(),
            data={"AAPL": market_event}
        )
        
        slippage = model.calculate_slippage(order, snapshot)
        
        # 1% of $151.00 = $1.51 (positive for buy orders)
        expected = Decimal('151.00') * Decimal('0.01')
        assert slippage == expected


def test_integration_basic():
    """Basic integration test of core components."""
    # This would be a more comprehensive test
    # combining multiple components together
    
    portfolio = Portfolio(initial_capital=Decimal('1000000'))
    
    # Create and process a complete trade cycle
    buy_fill = FillEvent(
        timestamp=datetime.now(),
        fill_id="fill_1",
        order_id="order_1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        fill_price=Decimal('150.00'),
        commission=Decimal('1.00')
    )
    
    portfolio.process_fill(buy_fill)
    
    # Update market price
    market_event = MarketEvent(
        timestamp=datetime.now(),
        symbol="AAPL",
        open_price=Decimal('155.00'),
        high_price=Decimal('157.00'),
        low_price=Decimal('154.00'),
        close_price=Decimal('156.00'),
        volume=1000000
    )
    
    from backtesting_engine.core.events import MarketDataSnapshot
    snapshot = MarketDataSnapshot(
        timestamp=datetime.now(),
        data={"AAPL": market_event}
    )
    
    portfolio.update_market_data(snapshot)
    
    # Check unrealized P&L
    position = portfolio.positions["AAPL"]
    expected_pnl = (Decimal('156.00') - Decimal('150.01')) * 100
    assert abs(position.unrealized_pnl - expected_pnl) < Decimal('0.01')


if __name__ == "__main__":
    # Run tests manually if not using pytest
    test_events = TestEvents()
    test_events.test_market_event_creation()
    test_events.test_signal_event_creation()
    test_events.test_order_event_creation()
    test_events.test_fill_event_creation()
    
    test_portfolio = TestPortfolio()
    test_portfolio.test_portfolio_initialization()
    test_portfolio.test_position_creation()
    test_portfolio.test_portfolio_fill_processing()
    
    test_commissions = TestCommissions()
    test_commissions.test_percentage_commission()
    
    test_slippage = TestSlippage()
    test_slippage.test_fixed_slippage()
    
    test_integration_basic()
    
    print("All tests passed!")
