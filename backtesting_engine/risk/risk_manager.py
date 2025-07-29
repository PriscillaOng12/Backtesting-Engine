"""
Risk management system for portfolio-level controls.

This module implements real-time risk monitoring, position limits,
and portfolio-level risk controls during backtesting.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..core.events import MarketEvent, OrderEvent, FillEvent, OrderSide
from ..core.portfolio import Portfolio


logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    
    # Portfolio-level limits
    max_portfolio_leverage: Optional[Decimal] = None
    max_drawdown_percent: Optional[Decimal] = None
    max_position_size_percent: Optional[Decimal] = None
    max_sector_exposure_percent: Optional[Decimal] = None
    
    # Daily limits
    max_daily_loss_percent: Optional[Decimal] = None
    max_daily_trades: Optional[int] = None
    max_daily_volume_percent: Optional[Decimal] = None
    
    # Risk metrics limits
    max_var_95_percent: Optional[Decimal] = None
    max_correlation_threshold: Optional[Decimal] = None
    
    # Asset-specific limits
    min_liquidity_ratio: Optional[Decimal] = None
    max_single_position_var: Optional[Decimal] = None


@dataclass
class RiskAlert:
    """Risk alert notification."""
    
    timestamp: datetime
    level: str  # 'WARNING', 'CRITICAL'
    category: str  # 'EXPOSURE', 'DRAWDOWN', 'CORRELATION', etc.
    message: str
    current_value: Any
    limit_value: Any
    symbol: Optional[str] = None


class RiskManager:
    """
    Real-time risk management system.
    
    Monitors portfolio risk during backtesting and can reject
    orders that violate risk limits.
    """
    
    def __init__(self, limits: RiskLimits):
        """
        Initialize risk manager.
        
        Parameters
        ----------
        limits : RiskLimits
            Risk limits configuration
        """
        self.limits = limits
        self.alerts: List[RiskAlert] = []
        self.daily_stats: Dict[str, Any] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Track daily activity
        self.current_date: Optional[datetime] = None
        self.daily_pnl: Decimal = Decimal('0')
        self.daily_trades: int = 0
        self.daily_volume: Decimal = Decimal('0')
        
        logger.info(f"Risk manager initialized with limits: {limits}")
    
    def check_order_risk(self, order: OrderEvent, portfolio: Portfolio, 
                        market_data: MarketEvent) -> bool:
        """
        Check if an order violates risk limits.
        
        Parameters
        ----------
        order : OrderEvent
            Order to check
        portfolio : Portfolio
            Current portfolio state
        market_data : MarketEvent
            Current market data
            
        Returns
        -------
        bool
            True if order is allowed, False if rejected
        """
        try:
            # Update daily tracking
            self._update_daily_tracking(market_data.timestamp)
            
            # Check various risk limits
            checks = [
                self._check_position_size_limit(order, portfolio, market_data),
                self._check_portfolio_leverage_limit(order, portfolio, market_data),
                self._check_daily_limits(order, portfolio),
                self._check_drawdown_limit(portfolio),
                self._check_sector_exposure_limit(order, portfolio),
                self._check_correlation_limit(order, portfolio)
            ]
            
            # All checks must pass
            return all(checks)
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False  # Reject on error
    
    def process_fill(self, fill: FillEvent, portfolio: Portfolio) -> None:
        """
        Process a fill event for risk tracking.
        
        Parameters
        ----------
        fill : FillEvent
            Fill event
        portfolio : Portfolio
            Portfolio state after fill
        """
        try:
            # Update daily statistics
            self.daily_trades += 1
            self.daily_volume += abs(fill.quantity * fill.fill_price)
            
            # Calculate impact on daily P&L
            if fill.side == "BUY":
                pnl_impact = -fill.quantity * fill.fill_price
            else:
                pnl_impact = fill.quantity * fill.fill_price
            
            self.daily_pnl += pnl_impact
            
            logger.debug(f"Processed fill for risk tracking: {fill.symbol} "
                        f"{fill.quantity}@{fill.fill_price}")
            
        except Exception as e:
            logger.error(f"Error processing fill for risk: {e}")
    
    def check_portfolio_risk(self, portfolio: Portfolio, 
                           market_data: Dict[str, Any]) -> List[RiskAlert]:
        """
        Comprehensive portfolio risk check.
        
        Parameters
        ----------
        portfolio : Portfolio
            Current portfolio state
        market_data : dict
            Current market data for all symbols
            
        Returns
        -------
        List[RiskAlert]
            List of risk alerts
        """
        alerts = []
        
        try:
            # Check drawdown
            drawdown_alert = self._check_current_drawdown(portfolio)
            if drawdown_alert:
                alerts.append(drawdown_alert)
            
            # Check VaR
            var_alert = self._check_var_limit(portfolio, market_data)
            if var_alert:
                alerts.append(var_alert)
            
            # Check correlations
            correlation_alerts = self._check_portfolio_correlations(portfolio, market_data)
            alerts.extend(correlation_alerts)
            
            # Store alerts
            self.alerts.extend(alerts)
            
            # Log critical alerts
            for alert in alerts:
                if alert.level == 'CRITICAL':
                    logger.warning(f"CRITICAL RISK ALERT: {alert.message}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error in portfolio risk check: {e}")
            return []
    
    def _update_daily_tracking(self, timestamp: datetime) -> None:
        """Update daily tracking variables."""
        current_date = timestamp.date()
        
        if self.current_date != current_date:
            # New day - reset counters
            self.current_date = current_date
            self.daily_pnl = Decimal('0')
            self.daily_trades = 0
            self.daily_volume = Decimal('0')
    
    def _check_position_size_limit(self, order: OrderEvent, 
                                 portfolio: Portfolio,
                                 market_data: MarketEvent) -> bool:
        """Check position size limit."""
        if self.limits.max_position_size_percent is None:
            return True
        
        # Calculate what position size would be after order
        current_position = portfolio.get_position_size(order.symbol)
        new_position = current_position
        
        if order.side == "BUY":
            new_position += order.quantity
        else:
            new_position -= order.quantity
        
        # Get current price from market data
        if order.symbol == market_data.symbol:
            current_price = float(market_data.close_price)
        else:
            return True  # Can't check without price for this symbol
        
        position_value = Decimal(str(abs(new_position * current_price)))
        portfolio_value = portfolio.calculate_total_equity()
        
        if portfolio_value <= 0:
            return False
        
        position_percent = float(position_value / portfolio_value)
        max_percent = float(self.limits.max_position_size_percent)
        
        if position_percent > max_percent:
            self._create_alert(
                'WARNING', 'EXPOSURE',
                f"Position size would exceed limit: {position_percent:.2%} > {max_percent:.2%}",
                position_percent, max_percent, order.symbol
            )
            return False
        
        return True
    
    def _check_portfolio_leverage_limit(self, order: OrderEvent, 
                                      portfolio: Portfolio, 
                                      market_data: MarketEvent) -> bool:
        """Check portfolio leverage limit."""
        if self.limits.max_portfolio_leverage is None:
            return True
        
        # Calculate leverage after order
        total_equity = portfolio.calculate_total_equity()
        if total_equity <= 0:
            return False
        
        # Calculate gross exposure - for now, use positions as-is since we don't have current prices for all symbols
        gross_exposure = Decimal('0')
        for symbol, position in portfolio.positions.items():
            # Use position's last known price
            gross_exposure += abs(position.quantity * position.last_price)
        
        # Add this order's exposure
        if order.symbol == market_data.symbol:
            order_price = market_data.close_price
            gross_exposure += abs(order.quantity * order_price)
        
        leverage = gross_exposure / total_equity
        max_leverage = float(self.limits.max_portfolio_leverage)
        
        if leverage > max_leverage:
            self._create_alert(
                'CRITICAL', 'LEVERAGE',
                f"Portfolio leverage would exceed limit: {leverage:.2f}x > {max_leverage:.2f}x",
                leverage, max_leverage
            )
            return False
        
        return True
    
    def _check_daily_limits(self, order: OrderEvent, portfolio: Portfolio) -> bool:
        """Check daily trading limits."""
        # Check daily trade count
        if (self.limits.max_daily_trades is not None and 
            self.daily_trades >= self.limits.max_daily_trades):
            self._create_alert(
                'WARNING', 'DAILY_LIMIT',
                f"Daily trade limit reached: {self.daily_trades}",
                self.daily_trades, self.limits.max_daily_trades
            )
            return False
        
        # Check daily loss
        if self.limits.max_daily_loss_percent is not None:
            portfolio_value = portfolio.calculate_total_equity()
            if portfolio_value > 0:
                daily_loss_percent = float(self.daily_pnl / portfolio_value)
                max_loss = -float(self.limits.max_daily_loss_percent)
                
                if daily_loss_percent < max_loss:
                    self._create_alert(
                        'CRITICAL', 'DAILY_LOSS',
                        f"Daily loss limit exceeded: {daily_loss_percent:.2%} < {max_loss:.2%}",
                        daily_loss_percent, max_loss
                    )
                    return False
        
        return True
    
    def _check_drawdown_limit(self, portfolio: Portfolio) -> bool:
        """Check maximum drawdown limit."""
        if self.limits.max_drawdown_percent is None:
            return True
        
        # Calculate current drawdown
        equity_curve = portfolio.get_equity_curve()
        if len(equity_curve) < 2:
            return True
        
        peak = max(equity_curve)
        current = equity_curve[-1]
        drawdown = (current - peak) / peak if peak > 0 else 0
        
        max_dd = -float(self.limits.max_drawdown_percent)
        
        if drawdown < max_dd:
            self._create_alert(
                'CRITICAL', 'DRAWDOWN',
                f"Drawdown limit exceeded: {drawdown:.2%} < {max_dd:.2%}",
                drawdown, max_dd
            )
            return False
        
        return True
    
    def _check_sector_exposure_limit(self, order: OrderEvent, 
                                   portfolio: Portfolio) -> bool:
        """Check sector exposure limits (simplified)."""
        # This would require sector mapping data
        # For now, return True
        return True
    
    def _check_correlation_limit(self, order: OrderEvent, 
                               portfolio: Portfolio) -> bool:
        """Check correlation limits."""
        if self.limits.max_correlation_threshold is None:
            return True
        
        # This would require correlation calculation
        # For now, return True
        return True
    
    def _check_current_drawdown(self, portfolio: Portfolio) -> Optional[RiskAlert]:
        """Check current drawdown against limit."""
        if self.limits.max_drawdown_percent is None:
            return None
        
        equity_curve = portfolio.get_equity_curve()
        if len(equity_curve) < 2:
            return None
        
        peak = max(equity_curve)
        current = equity_curve[-1]
        drawdown = (current - peak) / peak if peak > 0 else 0
        
        max_dd = -float(self.limits.max_drawdown_percent)
        
        if drawdown < max_dd * 0.8:  # Alert at 80% of limit
            return RiskAlert(
                timestamp=datetime.now(),
                level='WARNING' if drawdown > max_dd else 'CRITICAL',
                category='DRAWDOWN',
                message=f"Approaching drawdown limit: {drawdown:.2%}",
                current_value=drawdown,
                limit_value=max_dd
            )
        
        return None
    
    def _check_var_limit(self, portfolio: Portfolio, 
                        market_data: Dict[str, Any]) -> Optional[RiskAlert]:
        """Check VaR limit."""
        # Simplified VaR calculation
        return None
    
    def _check_portfolio_correlations(self, portfolio: Portfolio, 
                                    market_data: Dict[str, Any]) -> List[RiskAlert]:
        """Check portfolio correlation structure."""
        # This would calculate correlation matrix and check for concentration
        return []
    
    def _create_alert(self, level: str, category: str, message: str, 
                     current_value: Any, limit_value: Any, 
                     symbol: Optional[str] = None) -> None:
        """Create a risk alert."""
        alert = RiskAlert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            current_value=current_value,
            limit_value=limit_value,
            symbol=symbol
        )
        self.alerts.append(alert)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary statistics."""
        return {
            'total_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a.level == 'CRITICAL']),
            'warning_alerts': len([a for a in self.alerts if a.level == 'WARNING']),
            'daily_trades': self.daily_trades,
            'daily_pnl': float(self.daily_pnl),
            'recent_alerts': self.alerts[-10:] if self.alerts else []
        }
