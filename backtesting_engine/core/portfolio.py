"""
Portfolio management system for the backtesting engine.

This module handles position tracking, P&L calculation, margin requirements,
and portfolio-level risk controls during backtesting.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging
from dataclasses import dataclass, field

from .events import FillEvent, OrderSide, MarketDataSnapshot


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents a position in a single security.
    
    Attributes:
        symbol: Trading symbol
        quantity: Current position size (positive for long, negative for short)
        avg_price: Average price of the position
        last_price: Most recent market price
        unrealized_pnl: Current unrealized P&L
        realized_pnl: Cumulative realized P&L
        market_value: Current market value of position
        cost_basis: Total cost basis of position
    """
    symbol: str
    quantity: int = 0
    avg_price: Decimal = Decimal('0')
    last_price: Decimal = Decimal('0')
    cost_basis: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of the position."""
        return Decimal(str(self.quantity)) * self.last_price
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.quantity == 0:
            return Decimal('0')
        return self.market_value - self.cost_basis
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == 0
    
    def update_market_price(self, price: Decimal) -> None:
        """Update the last market price."""
        self.last_price = price
    
    def add_fill(self, fill: FillEvent) -> Decimal:
        """
        Add a fill to the position and return the realized P&L.
        
        Args:
            fill: Fill event to process
            
        Returns:
            Realized P&L from this fill
        """
        if fill.symbol != self.symbol:
            raise ValueError(f"Fill symbol {fill.symbol} does not match position symbol {self.symbol}")
        
        fill_quantity = fill.quantity if fill.side == "BUY" else -fill.quantity
        fill_cost = fill.gross_amount + fill.commission
        
        # Track trade times
        if self.first_trade_time is None:
            self.first_trade_time = fill.timestamp
        self.last_trade_time = fill.timestamp
        
        # Calculate realized P&L for position reductions
        realized_pnl = Decimal('0')
        
        if self.quantity != 0 and ((self.quantity > 0 > fill_quantity) or (self.quantity < 0 < fill_quantity)):
            # Position reduction - calculate realized P&L
            reduction_quantity = min(abs(self.quantity), abs(fill_quantity))
            avg_cost_per_share = self.cost_basis / abs(self.quantity)
            
            if fill.side == "SELL":
                # Selling long position
                realized_pnl = (fill.fill_price - self.avg_price) * Decimal(str(reduction_quantity)) - fill.commission
            else:
                # Covering short position
                realized_pnl = (self.avg_price - fill.fill_price) * Decimal(str(reduction_quantity)) - fill.commission
            
            self.realized_pnl += realized_pnl
            
            # Update cost basis
            remaining_ratio = Decimal(str((abs(self.quantity) - reduction_quantity) / abs(self.quantity)))
            self.cost_basis *= remaining_ratio
        
        # Update position
        new_quantity = self.quantity + fill_quantity
        
        if new_quantity == 0:
            # Position closed
            self.quantity = 0
            self.avg_price = Decimal('0')
            self.cost_basis = Decimal('0')
        elif self.quantity == 0 or (self.quantity > 0) == (new_quantity > 0):
            # New position or adding to existing position
            total_cost = self.cost_basis + abs(fill_cost)
            self.quantity = new_quantity
            self.cost_basis = total_cost
            self.avg_price = total_cost / abs(self.quantity)
        else:
            # Position reversal
            self.quantity = new_quantity
            self.cost_basis = abs(fill_cost)
            self.avg_price = abs(fill_cost) / abs(self.quantity)
        
        self.update_market_price(fill.fill_price)
        
        return realized_pnl


@dataclass
class PortfolioSummary:
    """Summary statistics for the portfolio."""
    timestamp: datetime
    total_value: Decimal
    cash: Decimal
    total_pnl: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    positions_count: int
    long_positions: int
    short_positions: int
    leverage: Decimal
    margin_used: Decimal
    margin_available: Decimal


class Portfolio:
    """
    Portfolio manager that tracks positions, cash, and P&L.
    
    This class maintains the state of all positions and calculates
    portfolio-level metrics for risk management and performance analysis.
    """
    
    def __init__(
        self,
        initial_capital: Decimal,
        margin_requirement: Decimal = Decimal('0.5'),
        max_leverage: Decimal = Decimal('2.0')
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Starting cash amount
            margin_requirement: Margin requirement as decimal (0.5 = 50%)
            max_leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.margin_requirement = margin_requirement
        self.max_leverage = max_leverage
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.historical_positions: List[Tuple[datetime, Dict[str, Position]]] = []
        
        # P&L tracking
        self.realized_pnl = Decimal('0')
        self.total_commission = Decimal('0')
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, Decimal]] = [(datetime.now(), initial_capital)]
        self.drawdown_curve: List[Tuple[datetime, Decimal]] = [(datetime.now(), Decimal('0'))]
        self.max_equity = initial_capital
        
        # Risk tracking
        self.margin_calls: List[Tuple[datetime, str]] = []
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol, creating if it doesn't exist."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_market_data(self, market_data: MarketDataSnapshot) -> None:
        """Update positions with latest market prices."""
        for symbol, position in self.positions.items():
            if symbol in market_data.data:
                price = market_data.get_price(symbol, 'close')
                if price:
                    position.update_market_price(price)
        
        # Update equity curve
        current_equity = self.calculate_total_equity()
        self.equity_curve.append((market_data.timestamp, current_equity))
        
        # Update drawdown
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_drawdown = (self.max_equity - current_equity) / self.max_equity
        self.drawdown_curve.append((market_data.timestamp, current_drawdown))
    
    def process_fill(self, fill: FillEvent) -> None:
        """
        Process a fill event and update portfolio state.
        
        Args:
            fill: Fill event to process
        """
        logger.debug(f"Processing fill: {fill.symbol} {fill.side} {fill.quantity} @ ${fill.fill_price}")
        
        # Get or create position
        position = self.get_position(fill.symbol)
        
        # Update position and get realized P&L
        realized_pnl = position.add_fill(fill)
        self.realized_pnl += realized_pnl
        self.total_commission += fill.commission
        
        # Update cash based on fill
        if fill.side == "BUY":
            self.cash -= (fill.gross_amount + fill.commission)
        else:
            self.cash += (fill.gross_amount - fill.commission)
        
        # Remove flat positions
        if position.is_flat:
            del self.positions[fill.symbol]
        
        logger.debug(f"Position after fill - {fill.symbol}: {position.quantity} shares, "
                    f"avg price: ${position.avg_price}, realized P&L: ${realized_pnl}")
    
    def calculate_total_equity(self) -> Decimal:
        """Calculate total portfolio equity."""
        market_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + market_value
    
    def calculate_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def calculate_total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.calculate_unrealized_pnl()
    
    def calculate_leverage(self) -> Decimal:
        """Calculate current leverage ratio."""
        total_equity = self.calculate_total_equity()
        if total_equity <= 0:
            return Decimal('0')
        
        gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        return gross_exposure / total_equity
    
    def calculate_margin_used(self) -> Decimal:
        """Calculate margin currently used."""
        return sum(
            abs(pos.market_value) * self.margin_requirement 
            for pos in self.positions.values()
        )
    
    def calculate_margin_available(self) -> Decimal:
        """Calculate available margin."""
        return self.cash - self.calculate_margin_used()
    
    def calculate_buying_power(self) -> Decimal:
        """Calculate current buying power."""
        return self.calculate_margin_available() / self.margin_requirement
    
    def can_trade(self, symbol: str, side: OrderSide, quantity: int, price: Decimal) -> Tuple[bool, str]:
        """
        Check if a trade can be executed given current portfolio state.
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Tuple of (can_trade, reason)
        """
        trade_value = Decimal(str(quantity)) * price
        
        if side == OrderSide.BUY:
            # Check if we have enough buying power
            required_margin = trade_value * self.margin_requirement
            if required_margin > self.calculate_margin_available():
                return False, "Insufficient margin for trade"
            
            # Check leverage limits
            new_leverage = self.calculate_leverage() + (trade_value / self.calculate_total_equity())
            if new_leverage > self.max_leverage:
                return False, f"Trade would exceed maximum leverage of {self.max_leverage}"
        
        else:  # SELL
            position = self.positions.get(symbol)
            if position is None or position.quantity < quantity:
                # Short selling - check margin requirements
                required_margin = trade_value * self.margin_requirement
                if required_margin > self.calculate_margin_available():
                    return False, "Insufficient margin for short sale"
        
        return True, "Trade approved"
    
    def get_position_size(self, symbol: str) -> int:
        """Get current position size for a symbol."""
        position = self.positions.get(symbol)
        return position.quantity if position else 0
    
    def get_symbols(self) -> Set[str]:
        """Get all symbols with current positions."""
        return set(self.positions.keys())
    
    def get_long_positions(self) -> Dict[str, Position]:
        """Get all long positions."""
        return {symbol: pos for symbol, pos in self.positions.items() if pos.is_long}
    
    def get_short_positions(self) -> Dict[str, Position]:
        """Get all short positions."""
        return {symbol: pos for symbol, pos in self.positions.items() if pos.is_short}
    
    def generate_summary(self, timestamp: datetime) -> PortfolioSummary:
        """Generate a portfolio summary at a given time."""
        total_equity = self.calculate_total_equity()
        unrealized_pnl = self.calculate_unrealized_pnl()
        
        long_positions = sum(1 for pos in self.positions.values() if pos.is_long)
        short_positions = sum(1 for pos in self.positions.values() if pos.is_short)
        
        return PortfolioSummary(
            timestamp=timestamp,
            total_value=total_equity,
            cash=self.cash,
            total_pnl=self.calculate_total_pnl(),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            positions_count=len(self.positions),
            long_positions=long_positions,
            short_positions=short_positions,
            leverage=self.calculate_leverage(),
            margin_used=self.calculate_margin_used(),
            margin_available=self.calculate_margin_available()
        )
    
    def save_snapshot(self, timestamp: datetime) -> None:
        """Save a snapshot of current positions."""
        snapshot = {symbol: Position(**pos.__dict__) for symbol, pos in self.positions.items()}
        self.historical_positions.append((timestamp, snapshot))
    
    def __repr__(self) -> str:
        """String representation of portfolio."""
        total_equity = self.calculate_total_equity()
        total_pnl = self.calculate_total_pnl()
        return (f"Portfolio(equity=${total_equity:,.2f}, "
                f"pnl=${total_pnl:,.2f}, "
                f"positions={len(self.positions)})")
