"""
Simulated broker for order execution.

This module provides a realistic broker simulation that handles order execution,
slippage, commissions, and partial fills.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
import logging
from enum import Enum

from ..core.events import (
    OrderEvent, FillEvent, MarketDataSnapshot, 
    OrderType, OrderSide, OrderStatus
)
from .slippage import SlippageModel, FixedSlippageModel
from .commissions import CommissionModel, PercentageCommissionModel


logger = logging.getLogger(__name__)


class OrderManager:
    """Manages pending orders and order lifecycle."""
    
    def __init__(self):
        """Initialize order manager."""
        self.pending_orders: Dict[str, OrderEvent] = {}
        self.order_status: Dict[str, OrderStatus] = {}
        self.partial_fills: Dict[str, int] = {}  # Tracks partially filled quantities
        
    def add_order(self, order: OrderEvent) -> None:
        """Add a new order to pending orders."""
        self.pending_orders[order.order_id] = order
        self.order_status[order.order_id] = OrderStatus.PENDING
        self.partial_fills[order.order_id] = 0
        
        logger.debug(f"Added pending order: {order.order_id} {order.symbol} "
                    f"{getattr(order.side, 'value', order.side)} {order.quantity}")
    
    def remove_order(self, order_id: str) -> Optional[OrderEvent]:
        """Remove an order from pending orders."""
        order = self.pending_orders.pop(order_id, None)
        if order:
            self.order_status.pop(order_id, None)
            self.partial_fills.pop(order_id, None)
            logger.debug(f"Removed order: {order_id}")
        return order
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[OrderEvent]:
        """Get pending orders, optionally filtered by symbol."""
        orders = list(self.pending_orders.values())
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        return orders
    
    def update_order_status(self, order_id: str, status: OrderStatus) -> None:
        """Update order status."""
        if order_id in self.order_status:
            self.order_status[order_id] = status
    
    def add_partial_fill(self, order_id: str, filled_quantity: int) -> int:
        """Add partial fill and return remaining quantity."""
        if order_id not in self.pending_orders:
            return 0
        
        order = self.pending_orders[order_id]
        self.partial_fills[order_id] += filled_quantity
        
        remaining = order.quantity - self.partial_fills[order_id]
        
        if remaining <= 0:
            self.update_order_status(order_id, OrderStatus.FILLED)
        else:
            self.update_order_status(order_id, OrderStatus.PARTIALLY_FILLED)
        
        return remaining


class SimulatedBroker:
    """
    Simulated broker for realistic order execution.
    
    Handles various order types, slippage, commissions, and partial fills
    to simulate real-world trading conditions.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        partial_fill_probability: float = 0.1,
        min_partial_fill_ratio: float = 0.3,
        latency_ms: int = 0,
        reject_probability: float = 0.001
    ):
        """
        Initialize simulated broker.
        
        Args:
            slippage_model: Model for calculating slippage
            commission_model: Model for calculating commissions
            partial_fill_probability: Probability of partial fills
            min_partial_fill_ratio: Minimum ratio for partial fills
            latency_ms: Simulated execution latency in milliseconds
            reject_probability: Probability of order rejection
        """
        self.slippage_model = slippage_model or FixedSlippageModel(Decimal('0.001'))
        self.commission_model = commission_model or PercentageCommissionModel(Decimal('0.001'))
        self.partial_fill_probability = partial_fill_probability
        self.min_partial_fill_ratio = min_partial_fill_ratio
        self.latency_ms = latency_ms
        self.reject_probability = reject_probability
        
        # Order management
        self.order_manager = OrderManager()
        
        # Statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.total_commission = Decimal('0')
        self.total_slippage = Decimal('0')
        
        # Engine reference
        self.engine = None
        
        logger.info(f"SimulatedBroker initialized with slippage: {slippage_model.__class__.__name__}, "
                   f"commission: {commission_model.__class__.__name__}")
    
    def set_engine(self, engine) -> None:
        """Set reference to backtesting engine."""
        self.engine = engine
    
    def execute_order(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> List[FillEvent]:
        """
        Execute an order and return fill events.
        
        Args:
            order: Order to execute
            market_data: Current market data
            
        Returns:
            List of fill events
        """
        self.total_orders += 1
        
        # Add to pending orders
        self.order_manager.add_order(order)
        
        # Check if order should be rejected
        if self._should_reject_order(order, market_data):
            self.rejected_orders += 1
            self.order_manager.update_order_status(order.order_id, OrderStatus.REJECTED)
            logger.warning(f"Order rejected: {order.order_id}")
            return []
        
        # Execute based on order type
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, market_data)
        elif order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, market_data)
        elif order.order_type == OrderType.STOP:
            return self._execute_stop_order(order, market_data)
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._execute_stop_limit_order(order, market_data)
        else:
            logger.error(f"Unsupported order type: {order.order_type}")
            return []
    
    def _should_reject_order(self, order: OrderEvent, market_data: MarketDataSnapshot) -> bool:
        """Determine if an order should be rejected."""
        import random
        
        # Random rejection (simulates broker issues, insufficient margin, etc.)
        if random.random() < self.reject_probability:
            return True
        
        # Check if market data is available
        if order.symbol not in market_data.data:
            logger.warning(f"No market data for {order.symbol}")
            return True
        
        # Check for unrealistic prices in limit orders
        if order.order_type == OrderType.LIMIT and order.price:
            current_price = market_data.get_price(order.symbol)
            if current_price is None:
                return True
            
            # Reject if limit price is too far from current price
            price_diff = abs(order.price - current_price) / current_price
            if price_diff > 0.1:  # 10% away from current price
                return True
        
        return False
    
    def _execute_market_order(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> List[FillEvent]:
        """Execute a market order."""
        if order.symbol not in market_data.data:
            return []
        
        current_price = market_data.get_price(order.symbol)
        if current_price is None:
            return []
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(order, market_data)
        fill_price = current_price + slippage
        
        # Ensure positive price
        if fill_price <= 0:
            fill_price = current_price
            slippage = Decimal('0')
        
        # Determine fill quantity (check for partial fills)
        fill_quantity = self._determine_fill_quantity(order, market_data)
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, fill_price, fill_quantity)
        
        # Create fill event
        fill = FillEvent(
            timestamp=market_data.timestamp,
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage
        )
        
        # Update order manager
        remaining_quantity = self.order_manager.add_partial_fill(order.order_id, fill_quantity)
        
        if remaining_quantity <= 0:
            self.order_manager.remove_order(order.order_id)
            self.filled_orders += 1
        
        # Update statistics
        self.total_commission += commission
        self.total_slippage += abs(slippage)
        
        logger.debug(f"Market order filled: {order.symbol} {getattr(order.side, 'value', order.side)} "
                    f"{fill_quantity} @ ${fill_price} (slippage: ${slippage})")
        
        return [fill]
    
    def _execute_limit_order(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> List[FillEvent]:
        """Execute a limit order."""
        if order.symbol not in market_data.data or order.price is None:
            return []
        
        market_event = market_data.data[order.symbol]
        
        # Check if limit order can be filled
        can_fill = False
        fill_price = order.price
        
        if order.side == OrderSide.BUY:
            # Buy limit order fills if market price is at or below limit price
            if market_event.low_price <= order.price:
                can_fill = True
                # Fill at the better of limit price or current price
                fill_price = min(order.price, market_event.close_price)
        else:
            # Sell limit order fills if market price is at or above limit price
            if market_event.high_price >= order.price:
                can_fill = True
                # Fill at the better of limit price or current price
                fill_price = max(order.price, market_event.close_price)
        
        if not can_fill:
            # Order remains pending
            logger.debug(f"Limit order not filled: {order.symbol} {getattr(order.side, 'value', order.side)} "
                        f"@ ${order.price}, current: ${market_event.close_price}")
            return []
        
        # Determine fill quantity
        fill_quantity = self._determine_fill_quantity(order, market_data)
        
        # Calculate commission (no slippage for limit orders)
        commission = self.commission_model.calculate_commission(order, fill_price, fill_quantity)
        
        # Create fill event
        fill = FillEvent(
            timestamp=market_data.timestamp,
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=Decimal('0')  # No slippage for limit orders
        )
        
        # Update order manager
        remaining_quantity = self.order_manager.add_partial_fill(order.order_id, fill_quantity)
        
        if remaining_quantity <= 0:
            self.order_manager.remove_order(order.order_id)
            self.filled_orders += 1
        
        # Update statistics
        self.total_commission += commission
        
        logger.debug(f"Limit order filled: {order.symbol} {getattr(order.side, 'value', order.side)} "
                    f"{fill_quantity} @ ${fill_price}")
        
        return [fill]
    
    def _execute_stop_order(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> List[FillEvent]:
        """Execute a stop order."""
        if order.symbol not in market_data.data or order.stop_price is None:
            return []
        
        market_event = market_data.data[order.symbol]
        
        # Check if stop order is triggered
        triggered = False
        
        if order.side == OrderSide.BUY:
            # Buy stop order triggers if price goes above stop price
            if market_event.high_price >= order.stop_price:
                triggered = True
        else:
            # Sell stop order triggers if price goes below stop price
            if market_event.low_price <= order.stop_price:
                triggered = True
        
        if not triggered:
            # Order remains pending
            return []
        
        # Convert to market order and execute
        market_order = OrderEvent(
            timestamp=market_data.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            order_type=OrderType.MARKET,
            side=order.side,
            quantity=order.quantity,
            strategy_id=order.strategy_id
        )
        
        return self._execute_market_order(market_order, market_data)
    
    def _execute_stop_limit_order(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> List[FillEvent]:
        """Execute a stop-limit order."""
        if (order.symbol not in market_data.data or 
            order.stop_price is None or 
            order.price is None):
            return []
        
        market_event = market_data.data[order.symbol]
        
        # Check if stop is triggered (same logic as stop order)
        triggered = False
        
        if order.side == OrderSide.BUY:
            if market_event.high_price >= order.stop_price:
                triggered = True
        else:
            if market_event.low_price <= order.stop_price:
                triggered = True
        
        if not triggered:
            return []
        
        # Convert to limit order and execute
        limit_order = OrderEvent(
            timestamp=market_data.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            order_type=OrderType.LIMIT,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            strategy_id=order.strategy_id
        )
        
        return self._execute_limit_order(limit_order, market_data)
    
    def _determine_fill_quantity(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> int:
        """Determine how much of an order to fill (handles partial fills)."""
        import random
        
        # Check for partial fill
        if random.random() < self.partial_fill_probability:
            # Calculate partial fill quantity
            min_fill = int(order.quantity * self.min_partial_fill_ratio)
            max_fill = order.quantity
            fill_quantity = random.randint(min_fill, max_fill)
            return max(1, fill_quantity)  # At least 1 share
        
        # Full fill
        return order.quantity
    
    def process_pending_orders(self, market_data: MarketDataSnapshot) -> List[FillEvent]:
        """Process all pending orders with new market data."""
        fills = []
        
        # Get all pending orders
        pending_orders = self.order_manager.get_pending_orders()
        
        for order in pending_orders:
            if order.symbol in market_data.data:
                order_fills = self.execute_order(order, market_data)
                fills.extend(order_fills)
        
        return fills
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        order = self.order_manager.remove_order(order_id)
        if order:
            self.order_manager.update_order_status(order_id, OrderStatus.CANCELLED)
            logger.info(f"Order cancelled: {order_id}")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get the status of an order."""
        return self.order_manager.order_status.get(order_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get broker execution statistics."""
        fill_rate = (self.filled_orders / self.total_orders) if self.total_orders > 0 else 0
        rejection_rate = (self.rejected_orders / self.total_orders) if self.total_orders > 0 else 0
        
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
            'fill_rate': fill_rate,
            'rejection_rate': rejection_rate,
            'total_commission': float(self.total_commission),
            'total_slippage': float(self.total_slippage),
            'avg_commission': float(self.total_commission / max(1, self.filled_orders)),
            'avg_slippage': float(self.total_slippage / max(1, self.filled_orders)),
            'pending_orders': len(self.order_manager.pending_orders)
        }
    
    def reset(self) -> None:
        """Reset broker state for new backtest."""
        self.order_manager = OrderManager()
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.total_commission = Decimal('0')
        self.total_slippage = Decimal('0')
        
        logger.info("Broker state reset")
    
    def __repr__(self) -> str:
        """String representation of broker."""
        return (f"SimulatedBroker(orders={self.total_orders}, "
                f"fills={self.filled_orders}, "
                f"pending={len(self.order_manager.pending_orders)})")
