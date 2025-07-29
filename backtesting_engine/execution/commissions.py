"""
Commission models for realistic cost simulation.

This module provides various commission structures commonly used
by brokers and trading platforms.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Any, Optional
import logging

from ..core.events import OrderEvent, FillEvent, OrderSide


logger = logging.getLogger(__name__)


class CommissionModel(ABC):
    """Abstract base class for commission models."""
    
    @abstractmethod
    def calculate_commission(self, order: OrderEvent, fill_price: Decimal, fill_quantity: int) -> Decimal:
        """
        Calculate commission for a trade.
        
        Args:
            order: Original order
            fill_price: Actual fill price
            fill_quantity: Quantity filled
            
        Returns:
            Commission amount
        """
        pass


class FixedCommissionModel(CommissionModel):
    """
    Fixed commission per trade.
    
    Common for discount brokers offering flat-rate pricing.
    """
    
    def __init__(self, commission_per_trade: Decimal):
        """
        Initialize fixed commission model.
        
        Args:
            commission_per_trade: Fixed commission amount per trade
        """
        self.commission_per_trade = commission_per_trade
        logger.info(f"FixedCommissionModel initialized: ${commission_per_trade} per trade")
    
    def calculate_commission(self, order: OrderEvent, fill_price: Decimal, fill_quantity: int) -> Decimal:
        """Calculate fixed commission."""
        return self.commission_per_trade


class PercentageCommissionModel(CommissionModel):
    """
    Percentage-based commission model.
    
    Commission is calculated as a percentage of trade value.
    """
    
    def __init__(self, commission_rate: Decimal, min_commission: Decimal = Decimal('0')):
        """
        Initialize percentage commission model.
        
        Args:
            commission_rate: Commission rate as decimal (e.g., 0.001 for 0.1%)
            min_commission: Minimum commission per trade
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        logger.info(f"PercentageCommissionModel initialized: {commission_rate*100:.3f}% "
                   f"(min: ${min_commission})")
    
    def calculate_commission(self, order: OrderEvent, fill_price: Decimal, fill_quantity: int) -> Decimal:
        """Calculate percentage-based commission."""
        trade_value = fill_price * fill_quantity
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)


class TieredCommissionModel(CommissionModel):
    """
    Tiered commission model based on trade volume or account size.
    
    Different commission rates apply based on predefined tiers.
    """
    
    def __init__(self, tiers: Dict[Decimal, Decimal], default_rate: Decimal):
        """
        Initialize tiered commission model.
        
        Args:
            tiers: Dictionary mapping volume thresholds to commission rates
            default_rate: Default rate for volumes below minimum tier
        """
        self.tiers = sorted(tiers.items())  # Sort by volume threshold
        self.default_rate = default_rate
        logger.info(f"TieredCommissionModel initialized with {len(tiers)} tiers")
    
    def calculate_commission(self, order: OrderEvent, fill_price: Decimal, fill_quantity: int) -> Decimal:
        """Calculate tiered commission based on trade value."""
        trade_value = fill_price * fill_quantity
        
        # Find applicable tier
        commission_rate = self.default_rate
        for threshold, rate in self.tiers:
            if trade_value >= threshold:
                commission_rate = rate
            else:
                break
        
        return trade_value * commission_rate


class PerShareCommissionModel(CommissionModel):
    """
    Per-share commission model.
    
    Common for institutional brokers and active traders.
    """
    
    def __init__(
        self, 
        commission_per_share: Decimal, 
        min_commission: Decimal = Decimal('0'),
        max_commission: Optional[Decimal] = None
    ):
        """
        Initialize per-share commission model.
        
        Args:
            commission_per_share: Commission per share
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade (optional)
        """
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission = max_commission
        logger.info(f"PerShareCommissionModel initialized: ${commission_per_share} per share")
    
    def calculate_commission(self, order: OrderEvent, fill_price: Decimal, fill_quantity: int) -> Decimal:
        """Calculate per-share commission."""
        commission = self.commission_per_share * fill_quantity
        commission = max(commission, self.min_commission)
        
        if self.max_commission is not None:
            commission = min(commission, self.max_commission)
        
        return commission


class InteractiveBrokersCommissionModel(CommissionModel):
    """
    Interactive Brokers commission model.
    
    Replicates IB's tiered per-share pricing structure.
    """
    
    def __init__(self, account_type: str = "pro"):
        """
        Initialize IB commission model.
        
        Args:
            account_type: "pro" or "lite" account type
        """
        self.account_type = account_type
        
        if account_type == "pro":
            # IB Pro tiered pricing (simplified)
            self.base_rate = Decimal('0.005')  # $0.005 per share
            self.min_commission = Decimal('1.00')
            self.max_rate = Decimal('0.01')  # 1% of trade value
        else:
            # IB Lite (commission-free for US stocks)
            self.base_rate = Decimal('0')
            self.min_commission = Decimal('0')
            self.max_rate = Decimal('0')
        
        logger.info(f"InteractiveBrokersCommissionModel initialized: {account_type} account")
    
    def calculate_commission(self, order: OrderEvent, fill_price: Decimal, fill_quantity: int) -> Decimal:
        """Calculate IB-style commission."""
        if self.account_type == "lite":
            return Decimal('0')
        
        trade_value = fill_price * fill_quantity
        
        # Per-share calculation
        commission = self.base_rate * fill_quantity
        
        # Apply minimum
        commission = max(commission, self.min_commission)
        
        # Apply maximum (percentage of trade value)
        max_commission = trade_value * self.max_rate
        commission = min(commission, max_commission)
        
        return commission


class FTXCommissionModel(CommissionModel):
    """
    FTX-style commission model with maker/taker fees.
    
    Different rates for market orders (taker) vs limit orders (maker).
    """
    
    def __init__(
        self, 
        maker_rate: Decimal = Decimal('0.0002'),
        taker_rate: Decimal = Decimal('0.0007'),
        volume_tier: int = 0
    ):
        """
        Initialize FTX commission model.
        
        Args:
            maker_rate: Maker fee rate
            taker_rate: Taker fee rate  
            volume_tier: Volume tier for discounted rates
        """
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
        self.volume_tier = volume_tier
        
        # Volume-based discounts (simplified)
        discount_factor = Decimal('1') - (Decimal('0.05') * volume_tier)
        self.maker_rate *= discount_factor
        self.taker_rate *= discount_factor
        
        logger.info(f"FTXCommissionModel initialized: "
                   f"maker={maker_rate*100:.4f}%, taker={taker_rate*100:.4f}%")
    
    def calculate_commission(self, order: OrderEvent, fill_price: Decimal, fill_quantity: int) -> Decimal:
        """Calculate maker/taker commission."""
        trade_value = fill_price * fill_quantity
        
        # For simplicity, assume market orders are taker, limit orders are maker
        if getattr(order.order_type, 'value', order.order_type) == "MARKET":
            commission_rate = self.taker_rate
        else:
            commission_rate = self.maker_rate
        
        return trade_value * commission_rate


def create_commission_model(model_type: str, **kwargs) -> CommissionModel:
    """
    Factory function to create commission models.
    
    Args:
        model_type: Type of commission model
        **kwargs: Model-specific parameters
        
    Returns:
        Commission model instance
    """
    model_map = {
        'fixed': FixedCommissionModel,
        'percentage': PercentageCommissionModel,
        'tiered': TieredCommissionModel,
        'per_share': PerShareCommissionModel,
        'interactive_brokers': InteractiveBrokersCommissionModel,
        'ftx': FTXCommissionModel
    }
    
    if model_type not in model_map:
        available_types = list(model_map.keys())
        raise ValueError(f"Unknown commission model type: {model_type}. "
                        f"Available types: {available_types}")
    
    return model_map[model_type](**kwargs)
