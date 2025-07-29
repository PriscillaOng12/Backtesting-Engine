"""
Slippage models for realistic order execution simulation.

This module provides various slippage models that simulate the market impact
and execution costs of trading in real markets.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional, Dict, Any
import logging
import math

from ..core.events import OrderEvent, MarketDataSnapshot, OrderType, OrderSide


logger = logging.getLogger(__name__)


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> Decimal:
        """
        Calculate slippage for an order.
        
        Args:
            order: Order to execute
            market_data: Current market data
            
        Returns:
            Slippage amount (positive values increase cost)
        """
        pass


class NoSlippageModel(SlippageModel):
    """No slippage model for testing or perfect execution scenarios."""
    
    def calculate_slippage(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> Decimal:
        """Return zero slippage."""
        return Decimal('0')


class FixedSlippageModel(SlippageModel):
    """
    Fixed slippage model.
    
    Applies a constant slippage amount or percentage to all trades.
    """
    
    def __init__(self, slippage_amount: Decimal, is_percentage: bool = True):
        """
        Initialize fixed slippage model.
        
        Args:
            slippage_amount: Fixed slippage amount
            is_percentage: Whether slippage_amount is a percentage or absolute amount
        """
        self.slippage_amount = slippage_amount
        self.is_percentage = is_percentage
        
        if is_percentage:
            logger.info(f"FixedSlippageModel initialized: {slippage_amount*100:.3f}%")
        else:
            logger.info(f"FixedSlippageModel initialized: ${slippage_amount}")
    
    def calculate_slippage(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> Decimal:
        """Calculate fixed slippage."""
        if order.symbol not in market_data.data:
            return Decimal('0')
        
        current_price = market_data.get_price(order.symbol, 'close')
        if current_price is None:
            return Decimal('0')
        
        if self.is_percentage:
            slippage = current_price * self.slippage_amount
        else:
            slippage = self.slippage_amount
        
        # Apply slippage direction based on order side
        if order.side == OrderSide.BUY:
            return slippage  # Positive slippage increases buy price
        else:
            return -slippage  # Negative slippage decreases sell price


class LinearSlippageModel(SlippageModel):
    """
    Linear slippage model based on order size and volatility.
    
    Slippage = base_rate + size_impact * (order_size / avg_volume) + volatility_impact * volatility
    """
    
    def __init__(
        self,
        base_rate: Decimal = Decimal('0.001'),
        size_impact: Decimal = Decimal('0.01'),
        volatility_impact: Decimal = Decimal('0.005'),
        lookback_period: int = 20
    ):
        """
        Initialize linear slippage model.
        
        Args:
            base_rate: Base slippage rate
            size_impact: Impact coefficient for order size
            volatility_impact: Impact coefficient for volatility
            lookback_period: Period for calculating average volume and volatility
        """
        self.base_rate = base_rate
        self.size_impact = size_impact
        self.volatility_impact = volatility_impact
        self.lookback_period = lookback_period
        
        # Cache for historical data
        self.price_history: Dict[str, list] = {}
        self.volume_history: Dict[str, list] = {}
        
        logger.info(f"LinearSlippageModel initialized: base={base_rate*100:.3f}%, "
                   f"size_impact={size_impact:.3f}, volatility_impact={volatility_impact:.3f}")
    
    def _update_history(self, symbol: str, market_data: MarketDataSnapshot) -> None:
        """Update price and volume history for volatility calculations."""
        if symbol not in market_data.data:
            return
        
        market_event = market_data.data[symbol]
        
        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
        
        # Add current data
        self.price_history[symbol].append(float(market_event.close_price))
        self.volume_history[symbol].append(market_event.volume)
        
        # Limit history size
        max_history = self.lookback_period * 2
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
    
    def _calculate_volatility(self, symbol: str) -> Decimal:
        """Calculate historical volatility."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return Decimal('0.02')  # Default volatility
        
        prices = self.price_history[symbol][-self.lookback_period:]
        if len(prices) < 2:
            return Decimal('0.02')
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        # Calculate standard deviation of returns
        if len(returns) < 2:
            return Decimal('0.02')
        
        mean_return = sum(returns) / len(returns)
        variance = sum((ret - mean_return) ** 2 for ret in returns) / (len(returns) - 1)
        volatility = math.sqrt(variance)
        
        return Decimal(str(volatility))
    
    def _calculate_avg_volume(self, symbol: str) -> int:
        """Calculate average volume."""
        if symbol not in self.volume_history or not self.volume_history[symbol]:
            return 1000000  # Default volume
        
        volumes = self.volume_history[symbol][-self.lookback_period:]
        return int(sum(volumes) / len(volumes))
    
    def calculate_slippage(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> Decimal:
        """Calculate linear slippage."""
        if order.symbol not in market_data.data:
            return Decimal('0')
        
        # Update historical data
        self._update_history(order.symbol, market_data)
        
        current_price = market_data.get_price(order.symbol, 'close')
        if current_price is None:
            return Decimal('0')
        
        # Calculate components
        volatility = self._calculate_volatility(order.symbol)
        avg_volume = self._calculate_avg_volume(order.symbol)
        
        # Size impact
        size_ratio = Decimal(str(order.quantity)) / Decimal(str(avg_volume))
        size_component = self.size_impact * size_ratio
        
        # Volatility impact
        volatility_component = self.volatility_impact * volatility
        
        # Total slippage rate
        slippage_rate = self.base_rate + size_component + volatility_component
        
        # Calculate slippage amount
        slippage = current_price * slippage_rate
        
        # Apply direction
        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


class SquareRootSlippageModel(SlippageModel):
    """
    Square-root market impact model.
    
    Based on academic research showing market impact scales with square root of order size.
    Slippage = impact_coefficient * sqrt(order_size / avg_volume) * volatility
    """
    
    def __init__(
        self,
        impact_coefficient: Decimal = Decimal('0.1'),
        volatility_scaling: Decimal = Decimal('1.0'),
        lookback_period: int = 20
    ):
        """
        Initialize square-root slippage model.
        
        Args:
            impact_coefficient: Market impact coefficient
            volatility_scaling: Volatility scaling factor
            lookback_period: Period for calculating statistics
        """
        self.impact_coefficient = impact_coefficient
        self.volatility_scaling = volatility_scaling
        self.lookback_period = lookback_period
        
        # Reuse linear model's history tracking
        self.linear_model = LinearSlippageModel(lookback_period=lookback_period)
        
        logger.info(f"SquareRootSlippageModel initialized: "
                   f"impact_coeff={impact_coefficient:.3f}, "
                   f"volatility_scaling={volatility_scaling:.3f}")
    
    def calculate_slippage(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> Decimal:
        """Calculate square-root slippage."""
        if order.symbol not in market_data.data:
            return Decimal('0')
        
        # Update historical data
        self.linear_model._update_history(order.symbol, market_data)
        
        current_price = market_data.get_price(order.symbol, 'close')
        if current_price is None:
            return Decimal('0')
        
        # Calculate components
        volatility = self.linear_model._calculate_volatility(order.symbol)
        avg_volume = self.linear_model._calculate_avg_volume(order.symbol)
        
        # Square-root impact
        size_ratio = float(order.quantity) / float(avg_volume)
        sqrt_impact = math.sqrt(max(0.0, size_ratio))
        
        # Market impact
        impact_rate = self.impact_coefficient * Decimal(str(sqrt_impact))
        volatility_adjusted_impact = impact_rate * volatility * self.volatility_scaling
        
        # Calculate slippage amount
        slippage = current_price * volatility_adjusted_impact
        
        # Apply direction
        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


class VolumeBasedSlippageModel(SlippageModel):
    """
    Volume-based slippage model.
    
    Slippage varies based on the order size relative to typical trading volume.
    Larger orders relative to average volume incur higher slippage.
    """
    
    def __init__(
        self,
        base_slippage: Decimal = Decimal('0.0005'),
        volume_threshold_1: Decimal = Decimal('0.01'),  # 1% of avg volume
        volume_threshold_2: Decimal = Decimal('0.05'),  # 5% of avg volume
        slippage_tier_1: Decimal = Decimal('0.001'),
        slippage_tier_2: Decimal = Decimal('0.003'),
        slippage_tier_3: Decimal = Decimal('0.01'),
        lookback_period: int = 20
    ):
        """
        Initialize volume-based slippage model.
        
        Args:
            base_slippage: Base slippage for small orders
            volume_threshold_1: First volume threshold
            volume_threshold_2: Second volume threshold  
            slippage_tier_1: Slippage for tier 1 orders
            slippage_tier_2: Slippage for tier 2 orders
            slippage_tier_3: Slippage for tier 3 orders
            lookback_period: Period for calculating average volume
        """
        self.base_slippage = base_slippage
        self.volume_threshold_1 = volume_threshold_1
        self.volume_threshold_2 = volume_threshold_2
        self.slippage_tier_1 = slippage_tier_1
        self.slippage_tier_2 = slippage_tier_2
        self.slippage_tier_3 = slippage_tier_3
        self.lookback_period = lookback_period
        
        # Reuse linear model's history tracking
        self.linear_model = LinearSlippageModel(lookback_period=lookback_period)
        
        logger.info(f"VolumeBasedSlippageModel initialized with tiered structure")
    
    def calculate_slippage(
        self, 
        order: OrderEvent, 
        market_data: MarketDataSnapshot
    ) -> Decimal:
        """Calculate volume-based slippage."""
        if order.symbol not in market_data.data:
            return Decimal('0')
        
        # Update historical data
        self.linear_model._update_history(order.symbol, market_data)
        
        current_price = market_data.get_price(order.symbol, 'close')
        if current_price is None:
            return Decimal('0')
        
        # Calculate volume ratio
        avg_volume = self.linear_model._calculate_avg_volume(order.symbol)
        volume_ratio = Decimal(str(order.quantity)) / Decimal(str(avg_volume))
        
        # Determine slippage tier
        if volume_ratio <= self.volume_threshold_1:
            slippage_rate = self.base_slippage
        elif volume_ratio <= self.volume_threshold_2:
            slippage_rate = self.slippage_tier_1
        elif volume_ratio <= self.volume_threshold_2 * 2:
            slippage_rate = self.slippage_tier_2
        else:
            slippage_rate = self.slippage_tier_3
        
        # Calculate slippage amount
        slippage = current_price * slippage_rate
        
        # Apply direction
        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


def create_slippage_model(model_type: str, **kwargs) -> SlippageModel:
    """
    Factory function to create slippage models.
    
    Args:
        model_type: Type of slippage model
        **kwargs: Model-specific parameters
        
    Returns:
        Slippage model instance
    """
    model_map = {
        'none': NoSlippageModel,
        'fixed': FixedSlippageModel,
        'linear': LinearSlippageModel,
        'square_root': SquareRootSlippageModel,
        'volume_based': VolumeBasedSlippageModel
    }
    
    if model_type not in model_map:
        available_types = list(model_map.keys())
        raise ValueError(f"Unknown slippage model type: {model_type}. "
                        f"Available types: {available_types}")
    
    return model_map[model_type](**kwargs)
