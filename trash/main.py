#!/usr/bin/env python
"""
Richie Rich - Crypto Trading AI
Main application entry point
"""

import os
import sys
import time
import logging
import threading
from dotenv import load_dotenv
from loguru import logger

# Local imports
from utils.config import Config
from utils.database import Database
from models.ai_brain import AIBrain
from trading.trader import Trader
from trading.account_manager import AccountManager
from analysis.market_analyzer import MarketAnalyzer
from analysis.news_analyzer import NewsAnalyzer
from analysis.sentiment_analyzer import SentimentAnalyzer
from web.app import create_app, start_websocket

# Load environment variables
load_dotenv()

# Setup logging
logger.add("logs/richie_rich_{time}.log", rotation="500 MB", level="INFO")
logger.info("Starting Richie Rich AI Trading System")

def initialize_system():
    """Initialize all system components"""
    logger.info("Initializing system components")
    
    # Initialize configuration
    config = Config()
    
    # Initialize database
    db = Database(config.get_mongodb_uri())
    
    # Initialize market analyzer
    market_analyzer = MarketAnalyzer(
        config.get_binance_keys(),
        config.get_coinmarketcap_key()
    )
    
    # Initialize news analyzer
    news_analyzer = NewsAnalyzer(config.get_news_api_key())
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Initialize account manager
    account_manager = AccountManager(
        config.get_binance_keys(),
        config.get_demo_account_balance()
    )
    
    # Initialize trader
    trader = Trader(account_manager, market_analyzer)
    
    # Initialize AI brain
    ai_brain = AIBrain(
        market_analyzer=market_analyzer,
        news_analyzer=news_analyzer,
        sentiment_analyzer=sentiment_analyzer,
        trader=trader,
        db=db,
        config=config
    )
    
    return {
        'config': config,
        'db': db,
        'market_analyzer': market_analyzer,
        'news_analyzer': news_analyzer,
        'sentiment_analyzer': sentiment_analyzer,
        'account_manager': account_manager,
        'trader': trader,
        'ai_brain': ai_brain
    }

def start_ai_threads(components):
    """Start AI threads for continuous learning and trading"""
    ai_brain = components['ai_brain']
    
    # Start market data collection thread
    market_thread = threading.Thread(
        target=ai_brain.continuous_market_analysis,
        daemon=True
    )
    market_thread.start()
    
    # Start news analysis thread
    news_thread = threading.Thread(
        target=ai_brain.continuous_news_analysis,
        daemon=True
    )
    news_thread.start()
    
    # Start trading decision thread
    trading_thread = threading.Thread(
        target=ai_brain.continuous_trading,
        daemon=True
    )
    trading_thread.start()
    
    # Start learning thread
    learning_thread = threading.Thread(
        target=ai_brain.continuous_learning,
        daemon=True
    )
    learning_thread.start()
    
    return {
        'market_thread': market_thread,
        'news_thread': news_thread,
        'trading_thread': trading_thread,
        'learning_thread': learning_thread
    }

def main():
    """Main application function"""
    try:
        # Initialize system components
        components = initialize_system()
        
        # Start AI threads
        threads = start_ai_threads(components)
        
        # Create and start web application
        app = create_app(components)
        
        # Start WebSocket server for real-time updates
        start_websocket(app, components)
        
        # Run the Flask application
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 