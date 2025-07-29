"""
Professional Backtesting Engine

A comprehensive, production-ready backtesting framework for quantitative trading strategies.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Trading Team"

from .core.engine import BacktestEngine
from .core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .core.portfolio import Portfolio
from .strategies.base import BaseStrategy

__all__ = [
    "BacktestEngine",
    "MarketEvent",
    "SignalEvent", 
    "OrderEvent",
    "FillEvent",
    "Portfolio",
    "BaseStrategy",
]
