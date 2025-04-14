"""
AI Orchestrator - Central AI orchestration engine.

This module serves as the central coordinator for all AI decision making processes,
integrating signals from various models and sources to guide the trading system.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """
    Central AI orchestration engine that integrates multiple AI systems and models
    to provide coordinated decision making for the trading system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI Orchestrator with configuration settings.
        
        Args:
            config: Configuration dictionary for the orchestrator
        """
        self.config = config
        self.models = {}
        self.active_strategies = []
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        logger.info("AI Orchestrator initialized")
        
    def register_model(self, model_id: str, model_instance: Any) -> None:
        """
        Register an AI model with the orchestrator.
        
        Args:
            model_id: Unique identifier for the model
            model_instance: The model object
        """
        self.models[model_id] = model_instance
        logger.info(f"Registered model: {model_id}")
        
    def register_strategy(self, strategy: Any) -> None:
        """
        Register a trading strategy with the orchestrator.
        
        Args:
            strategy: Strategy object to be registered
        """
        self.active_strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.name}")
        
    def collect_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect signals from all registered models.
        
        Args:
            market_data: Market data for signal generation
            
        Returns:
            Dictionary of signals from all models
        """
        signals = {}
        for model_id, model in self.models.items():
            try:
                signals[model_id] = model.generate_signals(market_data)
                logger.debug(f"Collected signals from {model_id}")
            except Exception as e:
                logger.error(f"Error collecting signals from {model_id}: {str(e)}")
                signals[model_id] = None
        return signals
    
    def weight_signals(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply weighting to signals based on model performance and configuration.
        
        Args:
            signals: Raw signals from models
            
        Returns:
            Weighted signals
        """
        weighted_signals = {}
        for model_id, signal in signals.items():
            if signal is None:
                continue
            
            model_weight = self.config.get('model_weights', {}).get(model_id, 1.0)
            weighted_signals[model_id] = {
                'signal': signal,
                'weight': model_weight
            }
        
        return weighted_signals
    
    def make_decision(self, weighted_signals: Dict[str, Dict[str, Any]], 
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a coordinated decision based on weighted signals and context.
        
        Args:
            weighted_signals: Signals with weights
            context: Additional contextual information
            
        Returns:
            Decision dictionary with action, confidence and rationale
        """
        if not weighted_signals:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'rationale': 'No valid signals available'
            }
        
        # Aggregate signal directions (buy/sell/hold)
        buy_confidence = 0.0
        sell_confidence = 0.0
        hold_confidence = 0.0
        total_weight = 0.0
        
        rationales = []
        
        for model_id, data in weighted_signals.items():
            signal = data['signal']
            weight = data['weight']
            
            if signal.get('action') == 'buy':
                buy_confidence += signal.get('confidence', 0.5) * weight
            elif signal.get('action') == 'sell':
                sell_confidence += signal.get('confidence', 0.5) * weight
            else:  # hold
                hold_confidence += signal.get('confidence', 0.5) * weight
                
            total_weight += weight
            
            if signal.get('rationale'):
                rationales.append(f"{model_id}: {signal.get('rationale')}")
        
        # Normalize confidences
        if total_weight > 0:
            buy_confidence /= total_weight
            sell_confidence /= total_weight
            hold_confidence /= total_weight
        
        # Determine final action
        action = 'hold'
        confidence = hold_confidence
        
        if buy_confidence > sell_confidence and buy_confidence > hold_confidence:
            action = 'buy'
            confidence = buy_confidence
        elif sell_confidence > buy_confidence and sell_confidence > hold_confidence:
            action = 'sell'
            confidence = sell_confidence
        
        # Apply risk adjustment based on context
        if context.get('high_volatility', False) and confidence < 0.8:
            confidence *= 0.8
            rationales.append("Confidence reduced due to high market volatility")
        
        # Check against threshold
        if confidence < self.confidence_threshold and action != 'hold':
            action = 'hold'
            rationales.append(f"Action changed to hold: confidence {confidence:.2f} below threshold {self.confidence_threshold}")
        
        return {
            'action': action,
            'confidence': confidence,
            'rationale': '; '.join(rationales)
        }
    
    def orchestrate(self, market_data: pd.DataFrame, 
                   additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main orchestration method to generate a coordinated decision.
        
        Args:
            market_data: Current market data
            additional_context: Any additional context information
            
        Returns:
            Final coordinated decision
        """
        context = additional_context or {}
        
        # Collect signals from all models
        raw_signals = self.collect_signals(market_data)
        
        # Weight the signals based on model performance
        weighted_signals = self.weight_signals(raw_signals)
        
        # Make coordinated decision
        decision = self.make_decision(weighted_signals, context)
        
        logger.info(f"Orchestrated decision: {decision['action']} with confidence {decision['confidence']:.2f}")
        logger.debug(f"Decision rationale: {decision['rationale']}")
        
        return decision 