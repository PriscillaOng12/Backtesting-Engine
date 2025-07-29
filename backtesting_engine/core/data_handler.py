"""
Data handling system for the backtesting engine.

This module provides data ingestion, validation, and management capabilities
for multiple data sources including CSV files, databases, and APIs.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Set
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .events import MarketEvent, MarketDataSnapshot


logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data sources."""
    source_type: str  # 'csv', 'database', 'api'
    path_or_connection: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    frequency: str = 'daily'  # 'tick', '1min', '5min', 'hourly', 'daily'
    timezone: str = 'UTC'
    validate_data: bool = True
    handle_missing: str = 'forward_fill'  # 'forward_fill', 'drop', 'interpolate'


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> List[str]:
        """
        Validate OHLCV data for consistency.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of validation errors
        """
        errors = []
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return errors
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                errors.append(f"Found non-positive prices in {col}")
        
        # Check for negative volume
        if (df['volume'] < 0).any():
            errors.append("Found negative volume")
        
        # Check OHLC relationships
        invalid_high = df['high'] < df[['open', 'low', 'close']].max(axis=1)
        if invalid_high.any():
            errors.append("High price is less than max(open, low, close)")
        
        invalid_low = df['low'] > df[['open', 'high', 'close']].min(axis=1)
        if invalid_low.any():
            errors.append("Low price is greater than min(open, high, close)")
        
        # Check for missing data
        if df.isnull().any().any():
            errors.append("Found missing values in data")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            errors.append("Found duplicate timestamps")
        
        return errors
    
    @staticmethod
    def clean_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Clean and handle missing data.
        
        Args:
            df: Input DataFrame
            method: Method for handling missing data
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        if method == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill')
        elif method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear')
        
        # Remove any remaining NaN values
        df_clean = df_clean.dropna()
        
        return df_clean


class BaseDataHandler(ABC):
    """Abstract base class for data handlers."""
    
    def __init__(self, config: DataConfig):
        """Initialize the data handler."""
        self.config = config
        self.symbols = set(config.symbols)
        self.current_data: Dict[str, pd.DataFrame] = {}
        self.data_generators: Dict[str, Iterator[MarketEvent]] = {}
        self.validator = DataValidator()
        
    @abstractmethod
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from the source."""
        pass
    
    def validate_and_clean_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean the loaded data."""
        cleaned_data = {}
        
        for symbol, df in data.items():
            logger.info(f"Validating data for {symbol}")
            
            if self.config.validate_data:
                errors = self.validator.validate_ohlcv(df)
                if errors:
                    logger.warning(f"Data validation errors for {symbol}: {errors}")
            
            # Clean the data
            cleaned_df = self.validator.clean_data(df, self.config.handle_missing)
            cleaned_data[symbol] = cleaned_df
            
            logger.info(f"Loaded {len(cleaned_df)} records for {symbol}")
        
        return cleaned_data
    
    def prepare_data(self) -> None:
        """Load, validate, and prepare data for backtesting."""
        logger.info("Loading data...")
        raw_data = self.load_data()
        
        logger.info("Validating and cleaning data...")
        self.current_data = self.validate_and_clean_data(raw_data)
        
        # Create data generators
        self.data_generators = {}
        for symbol, df in self.current_data.items():
            self.data_generators[symbol] = self._create_data_generator(symbol, df)
        
        logger.info(f"Data preparation complete for {len(self.symbols)} symbols")
    
    def _create_data_generator(self, symbol: str, df: pd.DataFrame) -> Iterator[MarketEvent]:
        """Create a generator for market events from DataFrame."""
        for timestamp, row in df.iterrows():
            yield MarketEvent(
                timestamp=timestamp,
                symbol=symbol,
                open_price=Decimal(str(row['open'])),
                high_price=Decimal(str(row['high'])),
                low_price=Decimal(str(row['low'])),
                close_price=Decimal(str(row['close'])),
                volume=int(row['volume']),
                adj_close=Decimal(str(row.get('adj_close', row['close']))),
                dividend=Decimal(str(row.get('dividend', 0))),
                split_ratio=Decimal(str(row.get('split_ratio', 1)))
            )
    
    def get_next_events(self) -> List[MarketEvent]:
        """Get the next market events across all symbols."""
        events = []
        exhausted_symbols = []
        
        for symbol, generator in self.data_generators.items():
            try:
                event = next(generator)
                events.append(event)
            except StopIteration:
                # No more data for this symbol
                exhausted_symbols.append(symbol)
                continue
        
        # Remove exhausted generators
        for symbol in exhausted_symbols:
            del self.data_generators[symbol]
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        return events
    
    def has_data(self) -> bool:
        """Check if there is more data available."""
        return bool(self.data_generators)
    
    def get_latest_data(self, symbol: str) -> Optional[pd.Series]:
        """Get the latest data point for a symbol."""
        if symbol in self.current_data and not self.current_data[symbol].empty:
            return self.current_data[symbol].iloc[-1]
        return None
    
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol within date range."""
        if symbol not in self.current_data:
            return None
        
        df = self.current_data[symbol]
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df


class CSVDataHandler(BaseDataHandler):
    """Data handler for CSV files."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files."""
        data = {}
        path = Path(self.config.path_or_connection)
        
        if path.is_file():
            # Single file with multiple symbols
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Assume symbol column exists or file name is the symbol
            if 'symbol' in df.columns:
                for symbol in self.symbols:
                    symbol_data = df[df['symbol'] == symbol].copy()
                    if not symbol_data.empty:
                        symbol_data = symbol_data.drop('symbol', axis=1)
                        data[symbol] = symbol_data
            else:
                # Assume single symbol file
                symbol = self.symbols[0] if self.symbols else path.stem
                data[symbol] = df
        
        elif path.is_dir():
            # Multiple files, one per symbol
            for symbol in self.symbols:
                file_path = path / f"{symbol}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    data[symbol] = df
                else:
                    logger.warning(f"Data file not found for symbol: {symbol}")
        
        else:
            raise FileNotFoundError(f"Data path not found: {path}")
        
        # Filter by date range
        for symbol, df in data.items():
            mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
            data[symbol] = df[mask]
        
        return data


class DatabaseDataHandler(BaseDataHandler):
    """Data handler for database sources."""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        try:
            from sqlalchemy import create_engine
            self.engine = create_engine(config.path_or_connection)
        except ImportError:
            raise ImportError("SQLAlchemy is required for database data handling")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from database."""
        data = {}
        
        for symbol in self.symbols:
            query = f"""
            SELECT timestamp, open, high, low, close, volume, adj_close
            FROM market_data 
            WHERE symbol = '{symbol}'
            AND timestamp >= '{self.config.start_date}'
            AND timestamp <= '{self.config.end_date}'
            ORDER BY timestamp
            """
            
            try:
                df = pd.read_sql(query, self.engine, index_col='timestamp', parse_dates=['timestamp'])
                if not df.empty:
                    data[symbol] = df
                else:
                    logger.warning(f"No data found for symbol: {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        return data


class APIDataHandler(BaseDataHandler):
    """Data handler for API sources like Yahoo Finance."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from API."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for API data handling")
        
        data = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=self._get_yf_interval()
                )
                
                if not df.empty:
                    # Standardize column names
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    df = df.rename(columns={
                        'adj_close': 'adj_close',
                        'dividends': 'dividend',
                        'stock_splits': 'split_ratio'
                    })
                    data[symbol] = df
                else:
                    logger.warning(f"No data found for symbol: {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        return data
    
    def _get_yf_interval(self) -> str:
        """Convert frequency to yfinance interval."""
        frequency_map = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            'hourly': '1h',
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        return frequency_map.get(self.config.frequency, '1d')


class DataManager:
    """Manages multiple data sources and provides unified access."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.handlers: List[BaseDataHandler] = []
        self.current_snapshot: Optional[MarketDataSnapshot] = None
        self.all_events: List[MarketEvent] = []
        self.event_index = 0
        
    def add_handler(self, handler: BaseDataHandler) -> None:
        """Add a data handler."""
        handler.prepare_data()
        self.handlers.append(handler)
        logger.info(f"Added data handler: {handler.__class__.__name__}")
    
    def prepare_event_stream(self) -> None:
        """Prepare the chronological event stream from all handlers."""
        all_events = []
        
        for handler in self.handlers:
            while handler.has_data():
                events = handler.get_next_events()
                if not events:
                    break
                all_events.extend(events)
        
        # Sort all events chronologically
        self.all_events = sorted(all_events, key=lambda x: x.timestamp)
        self.event_index = 0
        
        logger.info(f"Prepared event stream with {len(self.all_events)} events")
    
    def get_next_snapshot(self) -> Optional[MarketDataSnapshot]:
        """Get the next market data snapshot."""
        if self.event_index >= len(self.all_events):
            return None
        
        current_time = self.all_events[self.event_index].timestamp
        snapshot_data = {}
        
        # Collect all events at the current timestamp
        while (self.event_index < len(self.all_events) and 
               self.all_events[self.event_index].timestamp == current_time):
            
            event = self.all_events[self.event_index]
            snapshot_data[event.symbol] = event
            self.event_index += 1
        
        self.current_snapshot = MarketDataSnapshot(
            timestamp=current_time,
            data=snapshot_data
        )
        
        return self.current_snapshot
    
    def has_more_data(self) -> bool:
        """Check if there is more data in the stream."""
        return self.event_index < len(self.all_events)
    
    def get_symbols(self) -> Set[str]:
        """Get all available symbols."""
        symbols = set()
        for handler in self.handlers:
            symbols.update(handler.symbols)
        return symbols
    
    def get_historical_data(
        self, 
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        for handler in self.handlers:
            if symbol in handler.symbols:
                return handler.get_historical_data(symbol, start_date, end_date)
        return None


def create_data_handler(config: DataConfig) -> BaseDataHandler:
    """Factory function to create appropriate data handler."""
    if config.source_type == 'csv':
        return CSVDataHandler(config)
    elif config.source_type == 'database':
        return DatabaseDataHandler(config)
    elif config.source_type == 'api':
        return APIDataHandler(config)
    else:
        raise ValueError(f"Unsupported data source type: {config.source_type}")
