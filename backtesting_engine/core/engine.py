"""
Main backtesting engine that orchestrates the entire backtesting process.

This module contains the core BacktestEngine class that coordinates data handling,
strategy execution, order processing, and performance tracking.
"""

import heapq
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
import uuid

from .events import (
    Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    MarketDataSnapshot, OrderSide, OrderType
)
from .portfolio import Portfolio, PortfolioSummary
from .data_handler import DataManager, BaseDataHandler


logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the entire backtesting process.
    
    This class implements an event-driven architecture where market data, signals,
    orders, and fills are processed chronologically through an event queue system.
    """
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal = Decimal('1000000'),
        commission: Decimal = Decimal('0.001'),
        margin_requirement: Decimal = Decimal('0.5'),
        max_leverage: Decimal = Decimal('2.0'),
        benchmark: Optional[str] = None
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            commission: Commission rate (as decimal)
            margin_requirement: Margin requirement (as decimal)
            max_leverage: Maximum leverage allowed
            benchmark: Benchmark symbol for comparison
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission
        self.benchmark = benchmark
        
        # Core components
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            margin_requirement=margin_requirement,
            max_leverage=max_leverage
        )
        self.data_manager = DataManager()
        
        # Event system
        self.event_queue: List[Event] = []
        self.current_time: Optional[datetime] = None
        self.current_snapshot: Optional[MarketDataSnapshot] = None
        
        # Strategy management
        self.strategies: Dict[str, Any] = {}  # Will be properly typed when strategies are created
        
        # Execution system
        self.broker = None  # Will be set when broker is created
        
        # Performance tracking
        self.completed_trades: List[FillEvent] = []
        self.performance_history: List[PortfolioSummary] = []
        
        # State tracking
        self.is_running = False
        self.events_processed = 0
        self.total_events = 0
        
        # Configuration
        self.config = {
            'log_level': logging.INFO,
            'save_snapshots': True,
            'snapshot_frequency': 'daily',
            'max_events_per_cycle': 1000
        }
        
        logger.info(f"BacktestEngine initialized: {start_date} to {end_date}, "
                   f"capital=${initial_capital:,.2f}")
    
    def add_data_handler(self, handler: BaseDataHandler) -> None:
        """Add a data handler to the engine."""
        self.data_manager.add_handler(handler)
        logger.info(f"Added data handler for symbols: {handler.symbols}")
    
    def add_strategy(self, strategy_id: str, strategy: Any) -> None:
        """Add a trading strategy to the engine."""
        self.strategies[strategy_id] = strategy
        strategy.set_engine(self)  # Assume strategies have this method
        logger.info(f"Added strategy: {strategy_id}")
    
    def set_broker(self, broker: Any) -> None:
        """Set the execution broker."""
        self.broker = broker
        broker.set_engine(self)
        logger.info(f"Set broker: {broker.__class__.__name__}")
    
    def _add_event(self, event: Event) -> None:
        """Add an event to the priority queue."""
        heapq.heappush(self.event_queue, event)
    
    def _get_next_event(self) -> Optional[Event]:
        """Get the next event from the priority queue."""
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        return None
    
    def _process_market_event(self, event: MarketEvent) -> None:
        """Process a market data event."""
        # Update current time
        self.current_time = event.timestamp
        
        # Create or update market data snapshot
        if (self.current_snapshot is None or 
            self.current_snapshot.timestamp != event.timestamp):
            
            # Save previous snapshot if exists
            if self.current_snapshot is not None:
                self._finalize_snapshot()
            
            # Create new snapshot
            self.current_snapshot = MarketDataSnapshot(
                timestamp=event.timestamp,
                data={event.symbol: event}
            )
        else:
            # Add to current snapshot
            self.current_snapshot.data[event.symbol] = event
        
        logger.debug(f"Processed market event: {event.symbol} @ {event.timestamp}")
    
    def _finalize_snapshot(self) -> None:
        """Finalize the current market data snapshot and generate signals."""
        if self.current_snapshot is None:
            return
        
        # Update portfolio with latest market data
        self.portfolio.update_market_data(self.current_snapshot)
        
        # Generate strategy signals
        for strategy_id, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(self.current_snapshot, self.portfolio)
                for signal in signals:
                    self._add_event(signal)
                    logger.debug(f"Generated signal: {signal.symbol} {getattr(signal.signal_type, 'value', signal.signal_type)}")
            except Exception as e:
                logger.error(f"Error generating signals for {strategy_id}: {e}")
        
        # Save performance snapshot if configured
        if self.config['save_snapshots']:
            summary = self.portfolio.generate_summary(self.current_snapshot.timestamp)
            self.performance_history.append(summary)
    
    def _process_signal_event(self, event: SignalEvent) -> None:
        """Process a trading signal event."""
        try:
            # Convert signal to order
            order = self._signal_to_order(event)
            if order:
                self._add_event(order)
                logger.debug(f"Generated order from signal: {order.symbol} {getattr(order.side, 'value', order.side)} {order.quantity}")
        except Exception as e:
            logger.error(f"Error processing signal event: {e}")
    
    def _signal_to_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """Convert a signal event to an order event."""
        if not self.current_snapshot or signal.symbol not in self.current_snapshot.data:
            logger.warning(f"No market data for signal symbol: {signal.symbol}")
            return None
        
        current_price = self.current_snapshot.get_price(signal.symbol)
        if current_price is None:
            return None
        
        # Calculate target position
        current_position = self.portfolio.get_position_size(signal.symbol)
        
        if signal.target_percent is not None:
            # Position sizing based on target percentage
            portfolio_value = self.portfolio.calculate_total_equity()
            target_value = portfolio_value * Decimal(str(signal.target_percent))
            target_quantity = int(target_value / current_price)
        else:
            # Simple buy/sell signal
            if signal.signal_type == OrderSide.BUY:
                # Calculate position size (simple implementation)
                portfolio_value = self.portfolio.calculate_total_equity()
                position_value = portfolio_value * Decimal('0.05')  # 5% position size
                target_quantity = int(position_value / current_price)
            else:
                target_quantity = -abs(current_position)  # Close position
        
        # Calculate order quantity
        order_quantity = target_quantity - current_position
        
        if order_quantity == 0:
            return None
        
        # Determine order side
        order_side = OrderSide.BUY if order_quantity > 0 else OrderSide.SELL
        order_quantity = abs(order_quantity)
        
        # Check if trade is allowed
        can_trade, reason = self.portfolio.can_trade(
            signal.symbol, order_side, order_quantity, current_price
        )
        
        if not can_trade:
            logger.warning(f"Trade rejected for {signal.symbol}: {reason}")
            return None
        
        # Create order
        order = OrderEvent(
            timestamp=signal.timestamp,
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            order_type=OrderType.MARKET,  # Simple market orders for now
            side=order_side,
            quantity=order_quantity,
            strategy_id=signal.strategy_id
        )
        
        return order
    
    def _process_order_event(self, event: OrderEvent) -> None:
        """Process an order event."""
        if self.broker is None:
            logger.error("No broker set for order execution")
            return
        
        try:
            # Execute order through broker
            fills = self.broker.execute_order(event, self.current_snapshot)
            
            # Add fill events to queue
            for fill in fills:
                self._add_event(fill)
                logger.debug(f"Order executed: {fill.symbol} {getattr(fill.side, 'value', fill.side)} "
                           f"{fill.quantity} @ ${fill.fill_price}")
        except Exception as e:
            logger.error(f"Error executing order: {e}")
    
    def _process_fill_event(self, event: FillEvent) -> None:
        """Process a fill event."""
        try:
            # Update portfolio
            self.portfolio.process_fill(event)
            
            # Track completed trade
            self.completed_trades.append(event)
            
            logger.debug(f"Fill processed: {event.symbol} {getattr(event.side, 'value', event.side)} "
                        f"{event.quantity} @ ${event.fill_price}")
        except Exception as e:
            logger.error(f"Error processing fill event: {e}")
    
    def run(self):
        """
        Run the backtest.
        
        Returns:
            BacktestResults object containing performance metrics and analysis
        """
        logger.info("Starting backtest...")
        self.is_running = True
        
        try:
            # Prepare data
            logger.info("Preparing data stream...")
            self.data_manager.prepare_event_stream()
            self.total_events = len(self.data_manager.all_events)
            
            # Main event loop
            logger.info(f"Processing {self.total_events} events...")
            
            # Add initial market events to queue
            while self.data_manager.has_more_data():
                snapshot = self.data_manager.get_next_snapshot()
                if snapshot is None:
                    break
                
                # Add market events for this timestamp
                for symbol, market_event in snapshot.data.items():
                    self._add_event(market_event)
                
                # Process all events at this timestamp
                self._process_timestamp_events(snapshot.timestamp)
                
                self.events_processed += len(snapshot.data)
                
                # Progress logging
                if self.events_processed % 1000 == 0:
                    progress = (self.events_processed / self.total_events) * 100
                    logger.info(f"Progress: {progress:.1f}% ({self.events_processed}/{self.total_events})")
            
            # Finalize last snapshot
            if self.current_snapshot is not None:
                self._finalize_snapshot()
            
            logger.info("Backtest completed successfully")
            
            # Generate results
            return self._generate_results()
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False
    
    def _process_timestamp_events(self, timestamp: datetime) -> None:
        """Process all events for a given timestamp."""
        events_at_timestamp = []
        
        # Collect all events at this timestamp
        while self.event_queue and self.event_queue[0].timestamp == timestamp:
            events_at_timestamp.append(self._get_next_event())
        
        # Sort by priority
        events_at_timestamp.sort(key=lambda x: x.priority)
        
        # Process events in order
        for event in events_at_timestamp:
            if event.event_type == EventType.MARKET:
                self._process_market_event(event)
            elif event.event_type == EventType.SIGNAL:
                self._process_signal_event(event)
            elif event.event_type == EventType.ORDER:
                self._process_order_event(event)
            elif event.event_type == EventType.FILL:
                self._process_fill_event(event)
        
        # Finalize market data snapshot after processing all events
        if any(e.event_type == EventType.MARKET for e in events_at_timestamp):
            self._finalize_snapshot()
    
    def _generate_results(self):
        """Generate backtest results."""
        # Import here to avoid circular imports
        try:
            from ..analysis.performance import BacktestResults
            return BacktestResults(
                portfolio=self.portfolio,
                performance_history=self.performance_history,
                completed_trades=self.completed_trades,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                benchmark=self.benchmark
            )
        except ImportError:
            # Fallback if analysis module not available
            class SimpleResults:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                def print_summary(self):
                    print("Backtest completed - detailed analysis requires full dependencies")
            
            return SimpleResults(
                portfolio=self.portfolio,
                performance_history=self.performance_history,
                completed_trades=self.completed_trades,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                benchmark=self.benchmark
            )
    
    def get_current_time(self) -> Optional[datetime]:
        """Get the current backtest time."""
        return self.current_time
    
    def get_current_snapshot(self) -> Optional[MarketDataSnapshot]:
        """Get the current market data snapshot."""
        return self.current_snapshot
    
    def get_portfolio(self) -> Portfolio:
        """Get the current portfolio state."""
        return self.portfolio
    
    def stop(self) -> None:
        """Stop the backtest."""
        self.is_running = False
        logger.info("Backtest stopped by user")
    
    def __repr__(self) -> str:
        """String representation of the engine."""
        return (f"BacktestEngine({self.start_date} to {self.end_date}, "
                f"capital=${self.initial_capital:,.2f}, "
                f"strategies={len(self.strategies)})")
