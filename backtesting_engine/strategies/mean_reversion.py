"""
Mean Reversion Strategy using Bollinger Bands.

This strategy implements a classic mean reversion approach using Bollinger Bands
to identify overbought and oversold conditions.
"""

from typing import List, Optional
from decimal import Decimal

from ..strategies.base import BaseStrategy
from ..core.events import SignalEvent, MarketDataSnapshot, OrderSide
from ..core.portfolio import Portfolio


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band mean reversion strategy.
    
    Strategy Logic:
    - Buy when price touches lower Bollinger Band
    - Sell when price touches upper Bollinger Band or middle band
    - Use RSI as confirmation filter
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        lookback_period: int = 20,
        std_dev_multiplier: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        position_size: float = 0.05,  # 5% of portfolio per position
        stop_loss: float = 0.05,  # 5% stop loss
        take_profit: float = 0.10  # 10% take profit
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbols: List of symbols to trade
            lookback_period: Period for Bollinger Bands calculation
            std_dev_multiplier: Standard deviation multiplier for bands
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI level considered oversold
            rsi_overbought: RSI level considered overbought
            position_size: Position size as fraction of portfolio
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        parameters = {
            'lookback_period': lookback_period,
            'std_dev_multiplier': std_dev_multiplier,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        # Strategy-specific state
        self.entry_prices = {}  # Track entry prices for stop loss/take profit
        
    def generate_signals(
        self, 
        market_data: MarketDataSnapshot, 
        portfolio: Portfolio
    ) -> List[SignalEvent]:
        """Generate trading signals based on Bollinger Bands and RSI."""
        signals = []
        
        # Update indicators first
        self.update_indicators(market_data)
        
        for symbol in self.symbols:
            if symbol not in market_data.data:
                continue
            
            signal = self._generate_symbol_signal(symbol, market_data, portfolio)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _generate_symbol_signal(
        self,
        symbol: str,
        market_data: MarketDataSnapshot,
        portfolio: Portfolio
    ) -> Optional[SignalEvent]:
        """Generate signal for a specific symbol."""
        # Get current price
        current_price = market_data.get_price(symbol)
        if current_price is None:
            return None
        
        # Calculate indicators
        bb_bands = self.calculate_bollinger_bands(
            symbol, 
            self.parameters['lookback_period'],
            self.parameters['std_dev_multiplier']
        )
        
        rsi = self.calculate_rsi(symbol, self.parameters['rsi_period'])
        
        if bb_bands is None or rsi is None:
            return None
        
        current_position = self.get_position_size(symbol, portfolio)
        current_price_float = float(current_price)
        
        # Check for exit signals first
        if current_position != 0:
            exit_signal = self._check_exit_conditions(
                symbol, current_price_float, current_position, bb_bands, rsi
            )
            if exit_signal:
                return exit_signal
        
        # Check for entry signals
        if current_position == 0:
            entry_signal = self._check_entry_conditions(
                symbol, current_price_float, bb_bands, rsi, market_data, portfolio
            )
            if entry_signal:
                # Record entry price
                self.entry_prices[symbol] = current_price_float
                return entry_signal
        
        return None
    
    def _check_entry_conditions(
        self,
        symbol: str,
        current_price: float,
        bb_bands: dict,
        rsi: float,
        market_data: MarketDataSnapshot,
        portfolio: Portfolio
    ) -> Optional[SignalEvent]:
        """Check for entry signal conditions."""
        lower_band = bb_bands['lower']
        upper_band = bb_bands['upper']
        
        rsi_oversold = self.parameters['rsi_oversold']
        rsi_overbought = self.parameters['rsi_overbought']
        
        # Long entry: Price near lower band and RSI oversold
        if (current_price <= lower_band * 1.01 and  # Within 1% of lower band
            rsi <= rsi_oversold):
            
            signal_strength = min(1.0, (rsi_oversold - rsi) / 20.0 + 0.5)
            
            return SignalEvent(
                timestamp=market_data.timestamp,
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=OrderSide.BUY,
                strength=signal_strength,
                target_percent=self.parameters['position_size'],
                metadata={
                    'entry_reason': 'bollinger_band_oversold',
                    'rsi': rsi,
                    'bb_lower': lower_band,
                    'current_price': current_price
                }
            )
        
        # Short entry: Price near upper band and RSI overbought
        elif (current_price >= upper_band * 0.99 and  # Within 1% of upper band
              rsi >= rsi_overbought):
            
            signal_strength = min(1.0, (rsi - rsi_overbought) / 20.0 + 0.5)
            
            return SignalEvent(
                timestamp=market_data.timestamp,
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=OrderSide.SELL,
                strength=signal_strength,
                target_percent=self.parameters['position_size'],
                metadata={
                    'entry_reason': 'bollinger_band_overbought',
                    'rsi': rsi,
                    'bb_upper': upper_band,
                    'current_price': current_price
                }
            )
        
        return None
    
    def _check_exit_conditions(
        self,
        symbol: str,
        current_price: float,
        current_position: int,
        bb_bands: dict,
        rsi: float
    ) -> Optional[SignalEvent]:
        """Check for exit signal conditions."""
        middle_band = bb_bands['middle']
        entry_price = self.entry_prices.get(symbol, current_price)
        
        # Calculate P&L
        if current_position > 0:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Stop loss check
        if pnl_pct <= -self.parameters['stop_loss']:
            return self._create_exit_signal(
                symbol, current_position, 'stop_loss', 
                {'pnl_pct': pnl_pct, 'current_price': current_price}
            )
        
        # Take profit check
        if pnl_pct >= self.parameters['take_profit']:
            return self._create_exit_signal(
                symbol, current_position, 'take_profit',
                {'pnl_pct': pnl_pct, 'current_price': current_price}
            )
        
        # Mean reversion exit: price crosses middle band
        if current_position > 0 and current_price >= middle_band:
            return self._create_exit_signal(
                symbol, current_position, 'mean_reversion',
                {'middle_band': middle_band, 'current_price': current_price}
            )
        
        elif current_position < 0 and current_price <= middle_band:
            return self._create_exit_signal(
                symbol, current_position, 'mean_reversion',
                {'middle_band': middle_band, 'current_price': current_price}
            )
        
        return None
    
    def _create_exit_signal(
        self,
        symbol: str,
        current_position: int,
        exit_reason: str,
        metadata: dict
    ) -> SignalEvent:
        """Create an exit signal."""
        # Determine signal type (opposite of current position)
        if current_position > 0:
            signal_type = OrderSide.SELL
        else:
            signal_type = OrderSide.BUY
        
        # Clean up entry price tracking
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]
        
        return SignalEvent(
            timestamp=self.engine.get_current_time(),
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            strength=1.0,  # Full strength for exits
            target_percent=0.0,  # Close position
            metadata={
                'exit_reason': exit_reason,
                **metadata
            }
        )
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.entry_prices = {}


# Example usage and configuration
if __name__ == "__main__":
    # This would typically be in a separate script
    from datetime import datetime
    from backtesting_engine import BacktestEngine
    from backtesting_engine.core.data_handler import DataConfig, CSVDataHandler
    from backtesting_engine.execution.broker import SimulatedBroker
    from backtesting_engine.execution.slippage import LinearSlippageModel
    from backtesting_engine.execution.commissions import PercentageCommissionModel
    
    # Create strategy
    strategy = MeanReversionStrategy(
        strategy_id="mean_reversion_001",
        symbols=["AAPL", "MSFT", "GOOGL"],
        lookback_period=20,
        std_dev_multiplier=2.0,
        position_size=0.05
    )
    
    print(f"Created strategy: {strategy}")
    print(f"Strategy parameters: {strategy.parameters}")
