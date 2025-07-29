"""
Execution system for realistic order simulation.

This module provides realistic execution simulation with various
slippage and commission models that mirror real trading costs.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal

from .events import OrderEvent, FillEvent, MarketEvent, OrderSide, OrderType, MarketDataSnapshot


logger = logging.getLogger(__name__)


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(self, symbol: str, quantity: int, price: float, side: OrderSide) -> float:
        """
        Calculate slippage for an order.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        quantity : int
            Order quantity
        price : float
            Order price
        side : OrderSide
            Order side (buy/sell)
            
        Returns
        -------
        float
            Slippage amount
        """
        pass


class FixedSlippageModel(SlippageModel):
    """Fixed slippage model with constant slippage per share."""
    
    def __init__(self, slippage_per_share: float = 0.01):
        """
        Initialize fixed slippage model.
        
        Parameters
        ----------
        slippage_per_share : float
            Fixed slippage amount per share
        """
        self.slippage_per_share = slippage_per_share
    
    def calculate_slippage(self, symbol: str, quantity: int, price: float, side: OrderSide) -> float:
        """Calculate fixed slippage."""
        return self.slippage_per_share


class PercentageSlippageModel(SlippageModel):
    """Percentage-based slippage model."""
    
    def __init__(self, slippage_rate: float = 0.001):
        """
        Initialize percentage slippage model.
        
        Parameters
        ----------
        slippage_rate : float
            Slippage as percentage of price (e.g., 0.001 = 0.1%)
        """
        self.slippage_rate = slippage_rate
    
    def calculate_slippage(self, symbol: str, quantity: int, price: float, side: OrderSide) -> float:
        """Calculate percentage-based slippage."""
        return price * self.slippage_rate


class VolumeBasedSlippageModel(SlippageModel):
    """Volume-based slippage model with market impact."""
    
    def __init__(self, base_slippage: float = 0.001, impact_coefficient: float = 0.0001):
        """
        Initialize volume-based slippage model.
        
        Parameters
        ----------
        base_slippage : float
            Base slippage rate
        impact_coefficient : float
            Market impact coefficient
        """
        self.base_slippage = base_slippage
        self.impact_coefficient = impact_coefficient
    
    def calculate_slippage(self, symbol: str, quantity: int, price: float, side: OrderSide) -> float:
        """Calculate volume-based slippage with market impact."""
        base_slip = price * self.base_slippage
        impact = price * self.impact_coefficient * np.sqrt(quantity / 1000)
        return base_slip + impact


class CommissionModel(ABC):
    """Abstract base class for commission models."""
    
    @abstractmethod
    def calculate_commission(self, symbol: str, quantity: int, price: float) -> float:
        """
        Calculate commission for a trade.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        quantity : int
            Trade quantity
        price : float
            Trade price
            
        Returns
        -------
        float
            Commission amount
        """
        pass


class FixedCommissionModel(CommissionModel):
    """Fixed commission per trade."""
    
    def __init__(self, commission_per_trade: float = 1.0):
        """
        Initialize fixed commission model.
        
        Parameters
        ----------
        commission_per_trade : float
            Fixed commission amount per trade
        """
        self.commission_per_trade = commission_per_trade
    
    def calculate_commission(self, symbol: str, quantity: int, price: float) -> float:
        """Calculate fixed commission."""
        return self.commission_per_trade


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission model."""
    
    def __init__(self, commission_rate: float = 0.001):
        """
        Initialize percentage commission model.
        
        Parameters
        ----------
        commission_rate : float
            Commission as percentage of trade value
        """
        self.commission_rate = commission_rate
    
    def calculate_commission(self, symbol: str, quantity: int, price: float) -> float:
        """Calculate percentage-based commission."""
        return quantity * price * self.commission_rate


class TieredCommissionModel(CommissionModel):
    """Tiered commission model based on trade value."""
    
    def __init__(self, tiers: List[tuple] = None):
        """
        Initialize tiered commission model.
        
        Parameters
        ----------
        tiers : List[tuple]
            List of (threshold, rate) tuples for commission tiers
        """
        if tiers is None:
            # Default tiered structure
            tiers = [
                (10000, 0.005),   # 0.5% for trades under $10k
                (50000, 0.003),   # 0.3% for trades $10k-$50k
                (100000, 0.002),  # 0.2% for trades $50k-$100k
                (float('inf'), 0.001)  # 0.1% for trades over $100k
            ]
        self.tiers = sorted(tiers, key=lambda x: x[0])
    
    def calculate_commission(self, symbol: str, quantity: int, price: float) -> float:
        """Calculate tiered commission."""
        trade_value = quantity * price
        
        for threshold, rate in self.tiers:
            if trade_value <= threshold:
                return trade_value * rate
        
        return trade_value * self.tiers[-1][1]


class ExecutionHandler:
    """
    Handles order execution with realistic slippage and commission simulation.
    """
    
    def __init__(self, 
                 slippage_model: SlippageModel = None,
                 commission_model: CommissionModel = None,
                 market_impact: bool = True,
                 partial_fills: bool = False):
        """
        Initialize execution handler.
        
        Parameters
        ----------
        slippage_model : SlippageModel
            Model for calculating slippage
        commission_model : CommissionModel
            Model for calculating commissions
        market_impact : bool
            Whether to simulate market impact
        partial_fills : bool
            Whether to allow partial fills
        """
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.commission_model = commission_model or FixedCommissionModel()
        self.market_impact = market_impact
        self.partial_fills = partial_fills
        
        logger.info(f"Execution handler initialized with {type(self.slippage_model).__name__} "
                   f"and {type(self.commission_model).__name__}")
    
    def execute_order(self, order: OrderEvent, market_data: MarketDataSnapshot) -> Optional[FillEvent]:
        """
        Execute an order with realistic slippage and commissions.
        
        Parameters
        ----------
        order : OrderEvent
            Order to execute
        market_data : MarketEvent
            Current market data
            
        Returns
        -------
        FillEvent or None
            Fill event if execution successful, None otherwise
        """
        try:
            if order.symbol not in market_data.data:
                logger.warning(f"No market data available for {order.symbol}")
                return None
            
            symbol_data = market_data.data[order.symbol]
            
            # Determine execution price based on order type
            execution_price = self._get_execution_price(order, symbol_data)
            if execution_price is None:
                return None
            
            # Calculate slippage
            slippage = self.slippage_model.calculate_slippage(
                order.symbol, order.quantity, execution_price, order.side
            )
            
            # Apply slippage to execution price
            if order.side == "BUY":
                fill_price = execution_price + slippage
            else:
                fill_price = execution_price - slippage
            
            # Ensure positive price
            fill_price = max(fill_price, 0.01)
            
            # Calculate commission
            commission = self.commission_model.calculate_commission(
                order.symbol, order.quantity, fill_price
            )
            
            # Handle partial fills if enabled
            fill_quantity = order.quantity
            if self.partial_fills:
                fill_quantity = self._calculate_partial_fill(order, symbol_data)
            
            # Create fill event
            fill = FillEvent(
                fill_id=f"fill_{order.order_id}_{datetime.now().timestamp()}",
                order_id=order.order_id,
                timestamp=order.timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                fill_price=Decimal(str(fill_price)),
                commission=Decimal(str(commission)),
                slippage=Decimal(str(slippage))
            )
            
            logger.debug(f"Order executed: {order.symbol} {order.side} {fill_quantity} @ {fill_price:.4f}")
            
            return fill
            
        except Exception as e:
            logger.error(f"Error executing order {order.symbol}: {e}")
            return None
    
    def _get_execution_price(self, order: OrderEvent, symbol_data: MarketEvent) -> Optional[float]:
        """Get execution price based on order type and market data."""
        try:
            if order.order_type == "MARKET":
                # Market orders execute at current price
                if order.side == "BUY":
                    # Buy at close price (in real system would use ask)
                    return float(symbol_data.close_price)
                else:
                    # Sell at close price (in real system would use bid)
                    return float(symbol_data.close_price)
            
            elif order.order_type == "LIMIT":
                # Limit orders only execute if price is favorable
                current_price = float(symbol_data.close_price)
                
                if order.side == "BUY" and order.price >= current_price:
                    return min(float(order.price), current_price)
                elif order.side == "SELL" and order.price <= current_price:
                    return max(float(order.price), current_price)
                else:
                    return None  # Limit order not filled
            
            elif order.order_type == "STOP":
                # Stop orders become market orders when triggered
                current_price = float(symbol_data.close_price)
                
                if order.side == "BUY" and current_price >= order.price:
                    return current_price
                elif order.side == "SELL" and current_price <= order.price:
                    return current_price
                else:
                    return None  # Stop not triggered
            
            return float(symbol_data.close_price)
            
        except Exception as e:
            logger.error(f"Error determining execution price: {e}")
            return None
    
    def _calculate_partial_fill(self, order: OrderEvent, symbol_data: MarketEvent) -> int:
        """Calculate partial fill quantity based on volume."""
        try:
            volume = symbol_data.volume  # Get volume from MarketEvent
            
            # Assume we can fill up to 10% of daily volume
            max_fill = int(volume * 0.1)
            
            return min(order.quantity, max_fill) if max_fill > 0 else order.quantity
            
        except Exception as e:
            logger.error(f"Error calculating partial fill: {e}")
            return order.quantity
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'slippage_model': type(self.slippage_model).__name__,
            'commission_model': type(self.commission_model).__name__,
            'market_impact': self.market_impact,
            'partial_fills': self.partial_fills
        }
