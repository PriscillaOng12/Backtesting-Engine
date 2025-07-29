"""
Machine learning-based trading strategy.

This module implements ML-driven strategies using various algorithms
and feature engineering techniques for market prediction.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn or XGBoost not available. ML strategy will have limited functionality.")

from .base import BaseStrategy
from ..core.events import SignalEvent, OrderSide, MarketEvent
from ..core.portfolio import Portfolio


logger = logging.getLogger(__name__)


class MLTradingStrategy(BaseStrategy):
    """
    Machine Learning-based trading strategy.
    
    Uses various ML algorithms to predict price movements and generate
    trading signals based on technical and fundamental features.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 symbols: List[str],
                 model_type: str = 'random_forest',
                 prediction_horizon: int = 5,
                 feature_lookback: int = 20,
                 min_training_samples: int = 100,
                 retrain_frequency: int = 20,
                 prediction_threshold: float = 0.6,
                 position_size: float = 0.1,
                 feature_set: str = 'technical'):
        """
        Initialize ML trading strategy.
        
        Parameters
        ----------
        strategy_id : str
            Unique strategy identifier
        symbols : List[str]
            List of symbols to trade
        model_type : str
            ML model type ('random_forest', 'gradient_boosting', 'xgboost', 'logistic')
        prediction_horizon : int
            Number of periods ahead to predict
        feature_lookback : int
            Number of periods for feature calculation
        min_training_samples : int
            Minimum samples required for training
        retrain_frequency : int
            Frequency of model retraining (trading days)
        prediction_threshold : float
            Minimum prediction probability for signal generation
        position_size : float
            Position size as fraction of portfolio
        feature_set : str
            Feature set to use ('technical', 'price_action', 'comprehensive')
        """
        parameters = {
            'model_type': model_type,
            'prediction_horizon': prediction_horizon,
            'feature_lookback': feature_lookback,
            'min_training_samples': min_training_samples,
            'retrain_frequency': retrain_frequency,
            'prediction_threshold': prediction_threshold,
            'position_size': position_size,
            'feature_set': feature_set
        }
        
        super().__init__(strategy_id, symbols, parameters)
        
        # Strategy parameters
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.feature_lookback = feature_lookback
        self.min_training_samples = min_training_samples
        self.retrain_frequency = retrain_frequency
        self.prediction_threshold = prediction_threshold
        self.position_size = position_size
        self.feature_set = feature_set
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.last_training_date: Optional[datetime] = None
        self.training_counter = 0
        
        # Feature data storage
        self.feature_history: Dict[str, List[Dict[str, float]]] = {symbol: [] for symbol in symbols}
        self.target_history: Dict[str, List[int]] = {symbol: [] for symbol in symbols}
        
        # Model performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Check if ML libraries are available
        if not SKLEARN_AVAILABLE:
            logger.warning("ML libraries not available. Strategy will not generate signals.")
        
        logger.info(f"ML trading strategy initialized: {self}")
    
    def generate_signals(self, market_data: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """
        Generate ML-based trading signals.
        
        Parameters
        ----------
        market_data : MarketEvent
            Current market data
        portfolio : Portfolio
            Current portfolio state
            
        Returns
        -------
        List[SignalEvent]
            List of generated signals
        """
        if not SKLEARN_AVAILABLE:
            return []
        
        signals = []
        
        try:
            # Update indicators and features
            self.update_indicators(market_data)
            
            # Check if we need to retrain models
            if self._should_retrain_models(market_data.timestamp):
                self._train_models()
                self.last_training_date = market_data.timestamp
            
            # Generate predictions and signals for each symbol
            for symbol in self.symbols:
                if symbol in market_data.data:
                    symbol_signals = self._generate_symbol_signals(symbol, market_data, portfolio)
                    signals.extend(symbol_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return []
    
    def _should_retrain_models(self, current_date: datetime) -> bool:
        """Check if models should be retrained."""
        if self.last_training_date is None:
            return True
        
        days_since_training = (current_date - self.last_training_date).days
        return days_since_training >= self.retrain_frequency
    
    def _extract_features(self, symbol: str, current_data: Dict[str, float]) -> Dict[str, float]:
        """Extract features for ML model."""
        features = {}
        
        try:
            # Get price history
            prices = self.price_history[symbol]
            volumes = self.volume_history.get(symbol, [])
            
            if len(prices) < self.feature_lookback:
                return features
            
            recent_prices = np.array(prices[-self.feature_lookback:])
            recent_volumes = np.array(volumes[-self.feature_lookback:]) if volumes else np.ones_like(recent_prices)
            
            # Price-based features
            features.update(self._extract_price_features(recent_prices))
            
            # Technical indicator features
            if self.feature_set in ['technical', 'comprehensive']:
                features.update(self._extract_technical_features(recent_prices))
            
            # Volume features
            if self.feature_set == 'comprehensive' and len(recent_volumes) > 1:
                features.update(self._extract_volume_features(recent_prices, recent_volumes))
            
            # Price action features
            if self.feature_set in ['price_action', 'comprehensive']:
                features.update(self._extract_price_action_features(recent_prices))
            
            # Market microstructure features (if OHLC data available)
            if 'high' in current_data and 'low' in current_data:
                features.update(self._extract_microstructure_features(current_data))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return {}
    
    def _extract_price_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Extract basic price-based features."""
        features = {}
        
        try:
            # Returns
            returns = np.diff(prices) / prices[:-1]
            features['return_1'] = returns[-1] if len(returns) > 0 else 0
            features['return_5'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features['return_volatility'] = np.std(returns) if len(returns) > 1 else 0
            
            # Price levels
            current_price = prices[-1]
            features['price_normalized'] = (current_price - np.min(prices)) / (np.max(prices) - np.min(prices)) if np.max(prices) != np.min(prices) else 0.5
            
            # Moving averages
            if len(prices) >= 5:
                ma5 = np.mean(prices[-5:])
                features['price_vs_ma5'] = (current_price - ma5) / ma5
            
            if len(prices) >= 10:
                ma10 = np.mean(prices[-10:])
                features['price_vs_ma10'] = (current_price - ma10) / ma10
                features['ma5_vs_ma10'] = (ma5 - ma10) / ma10 if 'ma5' in locals() else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting price features: {e}")
            return {}
    
    def _extract_technical_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Extract technical indicator features."""
        features = {}
        
        try:
            # RSI
            if len(prices) >= 14:
                rsi = self._calculate_rsi(prices, 14)
                features['rsi'] = rsi[-1] if len(rsi) > 0 else 50
                features['rsi_oversold'] = 1 if features['rsi'] < 30 else 0
                features['rsi_overbought'] = 1 if features['rsi'] > 70 else 0
            
            # Bollinger Bands
            if len(prices) >= 20:
                bb_upper, bb_lower, bb_mid = self._calculate_bollinger_bands(prices, 20, 2)
                current_price = prices[-1]
                bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
                features['bb_position'] = bb_position
                features['bb_squeeze'] = (bb_upper[-1] - bb_lower[-1]) / bb_mid[-1] if bb_mid[-1] != 0 else 0
            
            # MACD
            if len(prices) >= 26:
                macd_line, signal_line = self._calculate_macd(prices)
                if len(macd_line) > 0 and len(signal_line) > 0:
                    features['macd'] = macd_line[-1]
                    features['macd_signal'] = signal_line[-1]
                    features['macd_histogram'] = macd_line[-1] - signal_line[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            return {}
    
    def _extract_volume_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Extract volume-based features."""
        features = {}
        
        try:
            # Volume trend
            if len(volumes) > 1:
                volume_ma = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
                features['volume_ratio'] = volumes[-1] / volume_ma if volume_ma > 0 else 1
            
            # Price-volume relationship
            returns = np.diff(prices) / prices[:-1]
            if len(returns) > 0 and len(volumes) > len(returns):
                volume_changes = np.diff(volumes[-len(returns)-1:])
                if len(volume_changes) > 0:
                    price_volume_corr = np.corrcoef(returns[-len(volume_changes):], volume_changes)[0, 1]
                    features['price_volume_correlation'] = price_volume_corr if not np.isnan(price_volume_corr) else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting volume features: {e}")
            return {}
    
    def _extract_price_action_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Extract price action features."""
        features = {}
        
        try:
            # Trend strength
            if len(prices) >= 10:
                slope = np.polyfit(range(len(prices[-10:])), prices[-10:], 1)[0]
                features['trend_slope'] = slope / prices[-1] if prices[-1] != 0 else 0
            
            # Momentum features
            if len(prices) >= 5:
                momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0
                features['momentum_5'] = momentum_5
            
            if len(prices) >= 10:
                momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
                features['momentum_10'] = momentum_10
            
            # Support/Resistance levels
            if len(prices) >= 20:
                recent_highs = np.array([np.max(prices[i-5:i+5]) for i in range(5, len(prices)-5)])
                recent_lows = np.array([np.min(prices[i-5:i+5]) for i in range(5, len(prices)-5)])
                
                if len(recent_highs) > 0 and len(recent_lows) > 0:
                    resistance_level = np.max(recent_highs)
                    support_level = np.min(recent_lows)
                    
                    features['distance_to_resistance'] = (resistance_level - prices[-1]) / prices[-1] if prices[-1] != 0 else 0
                    features['distance_to_support'] = (prices[-1] - support_level) / prices[-1] if prices[-1] != 0 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting price action features: {e}")
            return {}
    
    def _extract_microstructure_features(self, current_data: Dict[str, float]) -> Dict[str, float]:
        """Extract market microstructure features."""
        features = {}
        
        try:
            high = current_data.get('high', 0)
            low = current_data.get('low', 0)
            open_price = current_data.get('open', 0)
            close = current_data.get('close', 0)
            
            if high > 0 and low > 0 and open_price > 0 and close > 0:
                # Intraday range
                features['intraday_range'] = (high - low) / close if close != 0 else 0
                
                # Body vs shadow ratio
                body_size = abs(close - open_price)
                total_range = high - low
                features['body_ratio'] = body_size / total_range if total_range != 0 else 0
                
                # Upper and lower shadows
                upper_shadow = high - max(open_price, close)
                lower_shadow = min(open_price, close) - low
                features['upper_shadow_ratio'] = upper_shadow / total_range if total_range != 0 else 0
                features['lower_shadow_ratio'] = lower_shadow / total_range if total_range != 0 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting microstructure features: {e}")
            return {}
    
    def _calculate_target(self, symbol: str, current_index: int) -> int:
        """Calculate target variable for training."""
        try:
            prices = self.price_history[symbol]
            
            if current_index + self.prediction_horizon >= len(prices):
                return 0  # No future data available
            
            current_price = prices[current_index]
            future_price = prices[current_index + self.prediction_horizon]
            
            return_threshold = 0.01  # 1% return threshold
            future_return = (future_price - current_price) / current_price
            
            if future_return > return_threshold:
                return 1  # Buy signal
            elif future_return < -return_threshold:
                return -1  # Sell signal
            else:
                return 0  # Hold signal
                
        except Exception as e:
            logger.error(f"Error calculating target for {symbol}: {e}")
            return 0
    
    def _train_models(self) -> None:
        """Train ML models for all symbols."""
        try:
            logger.info("Training ML models...")
            
            for symbol in self.symbols:
                if len(self.price_history[symbol]) >= self.min_training_samples + self.prediction_horizon:
                    self._train_symbol_model(symbol)
            
            self.training_counter += 1
            logger.info(f"Model training completed (iteration {self.training_counter})")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _train_symbol_model(self, symbol: str) -> None:
        """Train model for a specific symbol."""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(symbol)
            
            if len(X) < self.min_training_samples:
                logger.warning(f"Insufficient training data for {symbol}")
                return
            
            # Initialize model
            model = self._create_model()
            scaler = RobustScaler()
            
            # Scale features
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
            
            # Evaluate model
            cv_scores = cross_val_score(model, X_scaled, y, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            # Store performance metrics
            self.model_performance[symbol] = {
                'cv_accuracy': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'training_samples': len(X),
                'last_training': datetime.now().isoformat()
            }
            
            logger.info(f"Model trained for {symbol}: CV Accuracy = {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
    
    def _prepare_training_data(self, symbol: str) -> tuple:
        """Prepare training data for a symbol."""
        X, y = [], []
        
        try:
            prices = self.price_history[symbol]
            
            # Create features and targets
            for i in range(self.feature_lookback, len(prices) - self.prediction_horizon):
                # Extract features for this time point
                historical_prices = prices[:i+1]
                
                # Mock current data for feature extraction
                current_data = {'close': prices[i]}
                
                # Temporarily set price history to historical data
                original_prices = self.price_history[symbol]
                self.price_history[symbol] = historical_prices
                
                features = self._extract_features(symbol, current_data)
                
                # Restore original price history
                self.price_history[symbol] = original_prices
                
                if features:
                    target = self._calculate_target(symbol, i)
                    
                    # Convert features to list
                    if not self.feature_names:
                        self.feature_names = sorted(features.keys())
                    
                    feature_vector = [features.get(name, 0) for name in self.feature_names]
                    
                    X.append(feature_vector)
                    y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data for {symbol}: {e}")
            return np.array([]), np.array([])
    
    def _create_model(self):
        """Create ML model based on model type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using random forest.")
            return RandomForestClassifier(random_state=42)
    
    def _generate_symbol_signals(self, symbol: str, market_data: MarketEvent, 
                               portfolio: Portfolio) -> List[SignalEvent]:
        """Generate signals for a specific symbol."""
        signals = []
        
        try:
            if symbol not in self.models or symbol not in self.scalers:
                return signals
            
            # Extract features for current data
            current_data = market_data.data[symbol]
            features = self._extract_features(symbol, current_data)
            
            if not features:
                return signals
            
            # Prepare feature vector
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            
            if len(feature_vector) != len(self.feature_names):
                return signals
            
            # Scale features
            scaler = self.scalers[symbol]
            X_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            model = self.models[symbol]
            prediction = model.predict(X_scaled)[0]
            prediction_proba = model.predict_proba(X_scaled)[0]
            
            # Get probability for the predicted class
            max_proba = np.max(prediction_proba)
            
            # Generate signal if prediction confidence is high enough
            if max_proba >= self.prediction_threshold:
                if prediction == 1:  # Buy signal
                    signals.append(SignalEvent(
                        timestamp=market_data.timestamp,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=OrderSide.BUY,
                        strength=max_proba,
                        target_percent=self.position_size,
                        metadata={
                            'ml_prediction': prediction,
                            'prediction_confidence': max_proba,
                            'model_type': self.model_type,
                            'feature_count': len(feature_vector)
                        }
                    ))
                elif prediction == -1:  # Sell signal
                    signals.append(SignalEvent(
                        timestamp=market_data.timestamp,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=OrderSide.SELL,
                        strength=max_proba,
                        target_percent=0.0,
                        metadata={
                            'ml_prediction': prediction,
                            'prediction_confidence': max_proba,
                            'model_type': self.model_type,
                            'feature_count': len(feature_vector)
                        }
                    ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML signals for {symbol}: {e}")
            return []
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for analysis."""
        state = super().get_strategy_state()
        
        # Add ML-specific state
        state.update({
            'model_type': self.model_type,
            'trained_models': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'training_iterations': self.training_counter,
            'last_training': self.last_training_date.isoformat() if self.last_training_date else None,
            'model_performance': self.model_performance,
            'feature_set': self.feature_set,
            'prediction_threshold': self.prediction_threshold
        })
        
        return state
