"""
Real-time execution monitoring module for tracking trade execution.
Monitors fills, slippage, and execution quality in real-time.
"""
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from threading import Thread, Lock

logger = logging.getLogger(__name__)

class ExecutionMonitor:
    """Monitor for real-time trade execution tracking."""
    
    def __init__(self, 
                 slippage_threshold: float = 0.05,
                 execution_timeout_sec: int = 300,
                 log_frequency_sec: int = 10):
        """
        Initialize the execution monitor.
        
        Args:
            slippage_threshold: Maximum acceptable slippage percentage
            execution_timeout_sec: Timeout for execution in seconds
            log_frequency_sec: Frequency of logging updates in seconds
        """
        self.slippage_threshold = slippage_threshold
        self.execution_timeout_sec = execution_timeout_sec
        self.log_frequency_sec = log_frequency_sec
        
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        self._lock = Lock()
        self._monitor_thread = None
        self._running = False
        self._alert_callbacks: List[Callable] = []
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for execution alerts.
        
        Args:
            callback: Function to call when an alert is generated
        """
        with self._lock:
            self._alert_callbacks.append(callback)
    
    def start_execution_monitoring(self, 
                                  execution_id: str,
                                  symbol: str,
                                  order_details: Dict[str, Any],
                                  expected_price: float,
                                  expected_quantity: float,
                                  algorithm: str = 'market') -> bool:
        """
        Start monitoring a new execution.
        
        Args:
            execution_id: Unique execution identifier
            symbol: Symbol being traded
            order_details: Order details dictionary
            expected_price: Expected execution price
            expected_quantity: Expected execution quantity
            algorithm: Execution algorithm name
        
        Returns:
            Success status
        """
        with self._lock:
            if execution_id in self.active_executions:
                logger.warning(f"Execution ID {execution_id} already being monitored")
                return False
            
            # Create execution record
            self.active_executions[execution_id] = {
                'execution_id': execution_id,
                'symbol': symbol,
                'order_details': order_details,
                'algorithm': algorithm,
                'expected_price': expected_price,
                'expected_quantity': expected_quantity,
                'filled_quantity': 0.0,
                'avg_fill_price': 0.0,
                'fills': [],
                'start_time': datetime.now(),
                'last_update_time': datetime.now(),
                'status': 'active',
                'slippage': 0.0,
                'completion_percentage': 0.0,
                'warnings': [],
                'market_conditions': {}
            }
            
            logger.info(f"Started monitoring execution {execution_id} for {symbol}")
            
            # Start monitoring thread if not running
            self._ensure_monitor_running()
            
            return True
    
    def update_fill(self, 
                   execution_id: str,
                   fill_price: float,
                   fill_quantity: float,
                   fill_time: Optional[datetime] = None,
                   fill_id: Optional[str] = None) -> bool:
        """
        Update an execution with new fill information.
        
        Args:
            execution_id: Execution identifier
            fill_price: Price of the fill
            fill_quantity: Quantity filled
            fill_time: Timestamp of the fill
            fill_id: Unique fill identifier
            
        Returns:
            Success status
        """
        with self._lock:
            if execution_id not in self.active_executions:
                logger.warning(f"Execution ID {execution_id} not found")
                return False
            
            execution = self.active_executions[execution_id]
            
            # Record fill information
            if fill_time is None:
                fill_time = datetime.now()
                
            if fill_id is None:
                fill_id = f"fill_{len(execution['fills']) + 1}"
                
            fill_info = {
                'fill_id': fill_id,
                'price': fill_price,
                'quantity': fill_quantity,
                'time': fill_time
            }
            
            execution['fills'].append(fill_info)
            
            # Update execution status
            prev_filled = execution['filled_quantity']
            execution['filled_quantity'] += fill_quantity
            
            # Update average fill price using volume-weighted average
            if execution['filled_quantity'] > 0:
                execution['avg_fill_price'] = (
                    (prev_filled * execution['avg_fill_price'] + fill_quantity * fill_price) /
                    execution['filled_quantity']
                )
            
            # Calculate slippage from expected price
            if execution['expected_price'] > 0:
                side = execution['order_details'].get('side', 'buy').lower()
                if side == 'buy':
                    # For buys, slippage is positive if we pay more than expected
                    price_diff = execution['avg_fill_price'] - execution['expected_price']
                else:
                    # For sells, slippage is positive if we receive less than expected
                    price_diff = execution['expected_price'] - execution['avg_fill_price']
                    
                execution['slippage'] = price_diff / execution['expected_price'] * 100.0
            
            # Update completion percentage
            if execution['expected_quantity'] > 0:
                execution['completion_percentage'] = min(100.0, (
                    execution['filled_quantity'] / execution['expected_quantity'] * 100.0
                ))
            
            # Update timestamps
            execution['last_update_time'] = datetime.now()
            
            # Check for completed execution
            if (execution['completion_percentage'] >= 99.9 or 
                abs(execution['filled_quantity'] - execution['expected_quantity']) < 0.000001):
                execution['status'] = 'completed'
                logger.info(f"Execution {execution_id} completed with avg price {execution['avg_fill_price']:.6f}")
                
                # Move to history
                self._complete_execution(execution_id)
            
            # Check for high slippage
            if abs(execution['slippage']) > self.slippage_threshold:
                warning = f"High slippage detected: {execution['slippage']:.2f}%"
                execution['warnings'].append({
                    'time': datetime.now(),
                    'type': 'high_slippage',
                    'message': warning
                })
                self._generate_alert(execution_id, 'high_slippage', warning)
            
            return True
    
    def update_market_conditions(self, 
                                execution_id: str,
                                market_data: Dict[str, Any]) -> bool:
        """
        Update market conditions for an execution.
        
        Args:
            execution_id: Execution identifier
            market_data: Current market data
            
        Returns:
            Success status
        """
        with self._lock:
            if execution_id not in self.active_executions:
                return False
            
            execution = self.active_executions[execution_id]
            execution['market_conditions'] = market_data
            
            # Check if market conditions warrant alerts
            current_price = market_data.get('price', 0)
            if current_price > 0 and execution['expected_price'] > 0:
                price_change = (current_price - execution['expected_price']) / execution['expected_price']
                
                # Alert on large price movements during execution
                if abs(price_change) > 0.03:  # 3% price movement
                    warning = f"Large price movement during execution: {price_change:.2f}%"
                    execution['warnings'].append({
                        'time': datetime.now(),
                        'type': 'price_movement',
                        'message': warning
                    })
                    self._generate_alert(execution_id, 'price_movement', warning)
            
            return True
    
    def cancel_execution(self, execution_id: str, reason: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            execution_id: Execution identifier
            reason: Reason for cancellation
            
        Returns:
            Success status
        """
        with self._lock:
            if execution_id not in self.active_executions:
                logger.warning(f"Execution ID {execution_id} not found")
                return False
            
            execution = self.active_executions[execution_id]
            execution['status'] = 'cancelled'
            execution['cancellation_reason'] = reason
            execution['cancellation_time'] = datetime.now()
            
            logger.info(f"Execution {execution_id} cancelled: {reason}")
            
            # Move to history
            self._complete_execution(execution_id)
            
            return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of an execution.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            Execution status dictionary or None if not found
        """
        with self._lock:
            # Check active executions
            if execution_id in self.active_executions:
                return self._create_status_snapshot(self.active_executions[execution_id])
            
            # Check execution history
            if execution_id in self.execution_history:
                return self._create_status_snapshot(self.execution_history[execution_id])
            
            return None
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """
        Get a list of all active executions.
        
        Returns:
            List of active execution status dictionaries
        """
        with self._lock:
            return [self._create_status_snapshot(exec_data) 
                   for exec_data in self.active_executions.values()]
    
    def _create_status_snapshot(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """Create a clean snapshot of execution status."""
        # Create a copy without internal fields
        snapshot = {
            'execution_id': execution['execution_id'],
            'symbol': execution['symbol'],
            'algorithm': execution['algorithm'],
            'side': execution['order_details'].get('side', 'unknown'),
            'expected_quantity': execution['expected_quantity'],
            'filled_quantity': execution['filled_quantity'],
            'expected_price': execution['expected_price'],
            'avg_fill_price': execution['avg_fill_price'],
            'slippage': execution['slippage'],
            'completion_percentage': execution['completion_percentage'],
            'status': execution['status'],
            'start_time': execution['start_time'].isoformat(),
            'last_update_time': execution['last_update_time'].isoformat(),
            'elapsed_seconds': (datetime.now() - execution['start_time']).total_seconds(),
            'warnings_count': len(execution['warnings']),
            'last_warning': execution['warnings'][-1] if execution['warnings'] else None,
        }
        
        return snapshot
    
    def _complete_execution(self, execution_id: str) -> None:
        """Move an execution from active to history."""
        with self._lock:
            if execution_id not in self.active_executions:
                return
                
            # Add completion time
            execution = self.active_executions[execution_id]
            execution['completion_time'] = datetime.now()
            
            # Move to history
            self.execution_history[execution_id] = execution
            
            # Remove from active
            del self.active_executions[execution_id]
    
    def _generate_alert(self, execution_id: str, alert_type: str, message: str) -> None:
        """Generate an execution alert."""
        alert = {
            'execution_id': execution_id,
            'alert_type': alert_type,
            'message': message,
            'time': datetime.now()
        }
        
        self.alerts.append(alert)
        logger.warning(f"Execution alert: {message}")
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def _ensure_monitor_running(self) -> None:
        """Ensure the monitoring thread is running."""
        if not self._running:
            self._running = True
            self._monitor_thread = Thread(target=self._monitoring_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        last_log_time = time.time()
        
        while self._running:
            try:
                # Check for timeouts and log status
                current_time = time.time()
                with self._lock:
                    for execution_id, execution in list(self.active_executions.items()):
                        elapsed_sec = (datetime.now() - execution['start_time']).total_seconds()
                        
                        # Check for timeout
                        if (elapsed_sec > self.execution_timeout_sec and 
                            execution['status'] == 'active' and
                            execution['completion_percentage'] < 100.0):
                            
                            warning = f"Execution timeout after {elapsed_sec:.1f} seconds"
                            execution['warnings'].append({
                                'time': datetime.now(),
                                'type': 'timeout',
                                'message': warning
                            })
                            self._generate_alert(execution_id, 'timeout', warning)
                    
                    # Periodic logging
                    if current_time - last_log_time > self.log_frequency_sec:
                        for execution_id, execution in self.active_executions.items():
                            logger.info(
                                f"Execution {execution_id} status: "
                                f"{execution['completion_percentage']:.1f}% complete, "
                                f"slippage: {execution['slippage']:.3f}%"
                            )
                        last_log_time = current_time
                
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5.0)  # Wait a bit longer on error
    
    def shutdown(self) -> None:
        """Shut down the monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None 