"""
Binance Client - API wrapper with demo mode support.

This module provides a unified interface for interacting with the Binance API,
with support for both real trading and demo/simulation mode.
"""

import logging
import time
import hmac
import hashlib
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class BinanceClient:
    """
    Binance API client with support for both live trading and demo mode.
    
    This client wraps the Binance API and provides a consistent interface
    for both real trading and a simulated environment for testing strategies.
    """
    
    API_URL = "https://api.binance.com/api"
    API_TESTNET_URL = "https://testnet.binance.vision/api"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Binance client.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.use_testnet = config.get('use_testnet', False)
        self.demo_mode = config.get('demo_mode', True)
        
        # Set the appropriate base URL
        if self.use_testnet:
            self.base_url = self.API_TESTNET_URL
        else:
            self.base_url = self.API_URL
            
        # Demo mode account state
        self.demo_balance = config.get('demo_account_balance', 10000.0)
        self.demo_positions = {}
        self.demo_orders = []
        self.demo_order_id_counter = 1
        
        logger.info(f"Binance client initialized. Mode: {'Demo' if self.demo_mode else 'Live'}")

    # ------ API Request Methods ------
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC-SHA256 signature for API request.
        
        Args:
            params: Request parameters
            
        Returns:
            Signature string
        """
        query_string = '&'.join([f"{key}={params[key]}" for key in params])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, method: str, endpoint: str, 
                     params: Optional[Dict[str, Any]] = None, 
                     signed: bool = False) -> Dict[str, Any]:
        """
        Make a request to the Binance API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether the request needs signature
            
        Returns:
            API response
        """
        if self.demo_mode and signed:
            # In demo mode, simulate API responses for signed (private) endpoints
            return self._simulate_private_api(endpoint, params)
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = requests.post(url, headers=headers, params=params)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise
    
    # ------ Market Data Methods ------
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.
        
        Returns:
            Exchange information
        """
        return self._make_request('GET', '/v3/exchangeInfo')
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr ticker price change statistics.
        
        Args:
            symbol: Trading pair symbol (e.g. "BTCUSDT")
            
        Returns:
            Ticker information
        """
        params = {'symbol': symbol}
        return self._make_request('GET', '/v3/ticker/24hr', params)
    
    def get_klines(self, symbol: str, interval: str, 
                  limit: int = 500) -> pd.DataFrame:
        """
        Get kline/candlestick data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, etc)
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with candlestick data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = self._make_request('GET', '/v3/klines', params)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_asset_volume', 'taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # Convert time columns
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of bids and asks to retrieve
            
        Returns:
            Order book data
        """
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._make_request('GET', '/v3/depth', params)
    
    # ------ Account Methods ------
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information including balances
        """
        return self._make_request('GET', '/v3/account', signed=True)
    
    def get_balances(self) -> Dict[str, float]:
        """
        Get current balances.
        
        Returns:
            Dictionary of asset balances
        """
        if self.demo_mode:
            return {'USDT': self.demo_balance}
        
        account_info = self.get_account_info()
        balances = {}
        
        for balance in account_info.get('balances', []):
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0:
                balances[balance['asset']] = {
                    'free': free,
                    'locked': locked,
                    'total': total
                }
        
        return balances
    
    # ------ Trading Methods ------
    
    def create_order(self, symbol: str, side: str, order_type: str,
                    quantity: Optional[float] = None,
                    price: Optional[float] = None,
                    time_in_force: str = 'GTC',
                    **kwargs) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT, MARKET, etc)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            **kwargs: Additional order parameters
            
        Returns:
            Order information
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'timestamp': int(time.time() * 1000)
        }
        
        if quantity is not None:
            params['quantity'] = quantity
            
        if price is not None:
            params['price'] = price
            
        if order_type == 'LIMIT':
            params['timeInForce'] = time_in_force
            
        params.update(kwargs)
        
        return self._make_request('POST', '/v3/order', params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            Cancellation confirmation
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return self._make_request('DELETE', '/v3/order', params, signed=True)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders on a symbol or all symbols.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        return self._make_request('GET', '/v3/openOrders', params, signed=True)
    
    def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Check an order's status.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to query
            
        Returns:
            Order status and information
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return self._make_request('GET', '/v3/order', params, signed=True)
    
    # ------ Demo Mode Simulation Methods ------
    
    def _simulate_private_api(self, endpoint: str, 
                             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simulate private API endpoints for demo mode.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Simulated API response
        """
        if endpoint == '/v3/account':
            return self._simulate_account_info()
        elif endpoint == '/v3/order' and params.get('type'):
            return self._simulate_create_order(params)
        elif endpoint == '/v3/order' and not params.get('type'):
            return self._simulate_get_order(params)
        elif endpoint == '/v3/openOrders':
            return self._simulate_open_orders(params)
        else:
            logger.warning(f"Unhandled demo endpoint: {endpoint}")
            return {}
    
    def _simulate_account_info(self) -> Dict[str, Any]:
        """
        Simulate account information.
        
        Returns:
            Simulated account info
        """
        return {
            'makerCommission': 10,
            'takerCommission': 10,
            'buyerCommission': 0,
            'sellerCommission': 0,
            'canTrade': True,
            'canWithdraw': True,
            'canDeposit': True,
            'updateTime': int(time.time() * 1000),
            'accountType': 'SPOT',
            'balances': [
                {
                    'asset': 'USDT',
                    'free': str(self.demo_balance),
                    'locked': '0.00000000'
                }
            ]
        }
    
    def _simulate_create_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate order creation.
        
        Args:
            params: Order parameters
            
        Returns:
            Simulated order response
        """
        symbol = params['symbol']
        side = params['side']
        order_type = params['type']
        quantity = float(params.get('quantity', 0))
        price = float(params.get('price', 0))
        
        # For market orders, calculate price based on current market
        if order_type == 'MARKET':
            # In a real implementation, we would fetch the current market price
            # For demo, we'll use a simple approximation
            price = self._get_demo_market_price(symbol)
        
        # Calculate order cost
        cost = price * quantity
        
        # Check if we have enough balance for a buy
        if side == 'BUY' and cost > self.demo_balance:
            return {
                'code': -2010,
                'msg': 'Account has insufficient balance for requested action.'
            }
        
        # Generate order ID
        order_id = self.demo_order_id_counter
        self.demo_order_id_counter += 1
        
        # Create timestamp
        timestamp = int(time.time() * 1000)
        
        # Create the order object
        order = {
            'symbol': symbol,
            'orderId': order_id,
            'orderListId': -1,
            'clientOrderId': f'demo_{order_id}',
            'price': str(price),
            'origQty': str(quantity),
            'executedQty': '0',
            'cummulativeQuoteQty': '0',
            'status': 'NEW',
            'timeInForce': params.get('timeInForce', 'GTC'),
            'type': order_type,
            'side': side,
            'stopPrice': '0.00000000',
            'icebergQty': '0.00000000',
            'time': timestamp,
            'updateTime': timestamp,
            'isWorking': True,
            'origQuoteOrderQty': '0.00000000'
        }
        
        # For market orders, execute immediately
        if order_type == 'MARKET':
            order['status'] = 'FILLED'
            order['executedQty'] = str(quantity)
            order['cummulativeQuoteQty'] = str(cost)
            
            # Update balances
            if side == 'BUY':
                self.demo_balance -= cost
                self.demo_positions[symbol] = self.demo_positions.get(symbol, 0) + quantity
            else:  # SELL
                self.demo_balance += cost
                self.demo_positions[symbol] = self.demo_positions.get(symbol, 0) - quantity
        else:
            # Add to open orders
            self.demo_orders.append(order)
        
        return order
    
    def _simulate_get_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate get order info.
        
        Args:
            params: Query parameters
            
        Returns:
            Simulated order info
        """
        order_id = int(params.get('orderId', 0))
        
        for order in self.demo_orders:
            if order['orderId'] == order_id:
                return order
                
        return {
            'code': -2013,
            'msg': 'Order does not exist.'
        }
    
    def _simulate_open_orders(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simulate open orders list.
        
        Args:
            params: Query parameters
            
        Returns:
            List of simulated open orders
        """
        symbol = params.get('symbol')
        
        if symbol:
            return [order for order in self.demo_orders 
                   if order['symbol'] == symbol and order['status'] == 'NEW']
        else:
            return [order for order in self.demo_orders if order['status'] == 'NEW']
    
    def _get_demo_market_price(self, symbol: str) -> float:
        """
        Get a simulated market price for demo mode.
        
        In a real implementation, this would fetch the actual current price.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current market price
        """
        # Try to get real price if possible
        try:
            ticker = self.get_ticker(symbol)
            return float(ticker['lastPrice'])
        except:
            # Fallback to dummy values for pure demo mode
            if symbol == 'BTCUSDT':
                return 40000.0
            elif symbol == 'ETHUSDT':
                return 2500.0
            else:
                return 100.0 