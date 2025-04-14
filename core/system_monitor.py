"""
System Monitor - System health monitoring and alerts.

This module provides system monitoring capabilities to track the health
and performance of the trading system, with alerting mechanisms for
critical issues.
"""

import logging
import threading
import time
import os
import platform
import psutil
import datetime
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    System health monitoring and alerting system that tracks the
    performance and status of the trading system components.
    """
    
    # Alert severity levels
    SEVERITY_INFO = 'INFO'
    SEVERITY_WARNING = 'WARNING'
    SEVERITY_ERROR = 'ERROR'
    SEVERITY_CRITICAL = 'CRITICAL'
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the system monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.components = {}
        self.alert_handlers = {}
        self.component_statuses = {}
        self.performance_metrics = {}
        self.monitoring_thread = None
        self.alert_thread = None
        self.stop_event = threading.Event()
        self.alert_queue = []
        self.alert_queue_lock = threading.Lock()
        self.metrics_lock = threading.RLock()
        
        # Configure metrics storage
        self.metrics_history_size = config.get('metrics_history_size', 100)
        self.alert_history_size = config.get('alert_history_size', 100)
        self.alert_history = []
        
        # Config thresholds
        self.cpu_threshold = config.get('cpu_threshold', 80)  # percentage
        self.memory_threshold = config.get('memory_threshold', 80)  # percentage
        self.disk_threshold = config.get('disk_threshold', 80)  # percentage
        self.component_timeout = config.get('component_timeout', 60)  # seconds
        
        # Email alert settings
        self.email_alerts = config.get('email_alerts', False)
        self.email_config = config.get('email_config', {})
        
        # Register system components
        self.register_component('system_monitor', 'Core system monitoring component')
        
        logger.info("System monitor initialized")
    
    def start_monitoring(self, interval: int = 30) -> None:
        """
        Start the monitoring thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
            
        self.stop_event.clear()
        
        # Start the main monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            name="SystemMonitorThread"
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start the alert processing thread
        self.alert_thread = threading.Thread(
            target=self._alert_processing_loop,
            name="AlertProcessingThread"
        )
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        logger.info(f"System monitoring started with interval {interval}s")
    
    def stop_monitoring(self) -> None:
        """
        Stop the monitoring thread.
        """
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring not running")
            return
            
        self.stop_event.set()
        self.monitoring_thread.join(timeout=10)
        
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=10)
            
        self.monitoring_thread = None
        self.alert_thread = None
        
        logger.info("System monitoring stopped")
    
    def register_component(self, component_id: str, description: str,
                          alert_on_failure: bool = True) -> None:
        """
        Register a system component for monitoring.
        
        Args:
            component_id: Unique identifier for the component
            description: Description of the component
            alert_on_failure: Whether to generate alerts for component failures
        """
        with self.metrics_lock:
            self.components[component_id] = {
                'description': description,
                'alert_on_failure': alert_on_failure,
                'registered_at': datetime.datetime.now()
            }
            
            # Initialize status
            self.component_statuses[component_id] = {
                'status': 'UNKNOWN',
                'last_updated': datetime.datetime.now(),
                'last_heartbeat': None,
                'message': 'Component registered but no heartbeat received'
            }
            
            # Initialize performance metrics
            self.performance_metrics[component_id] = {
                'cpu_usage': [],
                'memory_usage': [],
                'execution_time': [],
                'custom_metrics': {}
            }
            
        logger.debug(f"Component registered: {component_id}")
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a system component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            True if component was unregistered, False if not found
        """
        with self.metrics_lock:
            if component_id not in self.components:
                return False
                
            del self.components[component_id]
            
            if component_id in self.component_statuses:
                del self.component_statuses[component_id]
                
            if component_id in self.performance_metrics:
                del self.performance_metrics[component_id]
                
        logger.debug(f"Component unregistered: {component_id}")
        return True
    
    def update_component_status(self, component_id: str, status: str,
                               message: Optional[str] = None) -> None:
        """
        Update the status of a component.
        
        Args:
            component_id: Component identifier
            status: Status string ('OK', 'WARNING', 'ERROR', 'CRITICAL')
            message: Optional status message
        """
        with self.metrics_lock:
            if component_id not in self.components:
                logger.warning(f"Attempted to update unknown component: {component_id}")
                return
                
            old_status = None
            if component_id in self.component_statuses:
                old_status = self.component_statuses[component_id].get('status')
                
            self.component_statuses[component_id] = {
                'status': status,
                'last_updated': datetime.datetime.now(),
                'last_heartbeat': datetime.datetime.now(),
                'message': message or 'Status updated'
            }
            
            # Generate alert if status degraded
            if (old_status in ('OK', 'UNKNOWN') and status in ('WARNING', 'ERROR', 'CRITICAL')) \
                    and self.components[component_id].get('alert_on_failure', True):
                severity = self.SEVERITY_WARNING
                if status == 'ERROR':
                    severity = self.SEVERITY_ERROR
                elif status == 'CRITICAL':
                    severity = self.SEVERITY_CRITICAL
                    
                self.add_alert(
                    component_id,
                    f"Component status degraded to {status}",
                    severity=severity,
                    details={
                        'old_status': old_status,
                        'new_status': status,
                        'message': message
                    }
                )
                
        logger.debug(f"Component {component_id} status updated to {status}")
    
    def component_heartbeat(self, component_id: str, status: str = 'OK',
                           message: Optional[str] = None) -> None:
        """
        Register a heartbeat for a component.
        
        Args:
            component_id: Component identifier
            status: Component status
            message: Optional status message
        """
        with self.metrics_lock:
            if component_id not in self.components:
                logger.warning(f"Heartbeat for unknown component: {component_id}")
                return
                
            if component_id in self.component_statuses:
                self.component_statuses[component_id]['last_heartbeat'] = datetime.datetime.now()
                
                # Only update status if explicitly provided
                if status:
                    old_status = self.component_statuses[component_id].get('status')
                    self.component_statuses[component_id]['status'] = status
                    
                    # Include a message if provided
                    if message:
                        self.component_statuses[component_id]['message'] = message
                        
                    # Generate alert if status degraded
                    if (old_status in ('OK', 'UNKNOWN') and status in ('WARNING', 'ERROR', 'CRITICAL')) \
                            and self.components[component_id].get('alert_on_failure', True):
                        severity = self.SEVERITY_WARNING
                        if status == 'ERROR':
                            severity = self.SEVERITY_ERROR
                        elif status == 'CRITICAL':
                            severity = self.SEVERITY_CRITICAL
                            
                        self.add_alert(
                            component_id,
                            f"Component status degraded to {status}",
                            severity=severity,
                            details={
                                'old_status': old_status,
                                'new_status': status,
                                'message': message
                            }
                        )
            else:
                # Initialize status if not present
                self.component_statuses[component_id] = {
                    'status': status or 'UNKNOWN',
                    'last_updated': datetime.datetime.now(),
                    'last_heartbeat': datetime.datetime.now(),
                    'message': message or 'Initial heartbeat'
                }
    
    def add_performance_metric(self, component_id: str, metric_type: str,
                              value: float, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a performance metric for a component.
        
        Args:
            component_id: Component identifier
            metric_type: Type of metric ('cpu_usage', 'memory_usage', 'execution_time', or custom)
            value: Metric value
            details: Optional additional details
        """
        with self.metrics_lock:
            if component_id not in self.components:
                logger.warning(f"Metrics for unknown component: {component_id}")
                return
                
            now = datetime.datetime.now()
            metric_data = {
                'timestamp': now,
                'value': value,
                'details': details or {}
            }
            
            if metric_type in ('cpu_usage', 'memory_usage', 'execution_time'):
                # Standard metric
                metrics_list = self.performance_metrics[component_id][metric_type]
                metrics_list.append(metric_data)
                
                # Trim list if too long
                if len(metrics_list) > self.metrics_history_size:
                    metrics_list.pop(0)
            else:
                # Custom metric
                custom_metrics = self.performance_metrics[component_id]['custom_metrics']
                if metric_type not in custom_metrics:
                    custom_metrics[metric_type] = []
                    
                custom_metrics[metric_type].append(metric_data)
                
                # Trim list if too long
                if len(custom_metrics[metric_type]) > self.metrics_history_size:
                    custom_metrics[metric_type].pop(0)
    
    def register_alert_handler(self, handler_id: str, handler_func: Callable,
                              severity_levels: Optional[List[str]] = None) -> None:
        """
        Register a function to handle alerts.
        
        Args:
            handler_id: Unique identifier for the handler
            handler_func: Alert handler function
            severity_levels: List of severity levels to handle
        """
        if severity_levels is None:
            severity_levels = [self.SEVERITY_WARNING, self.SEVERITY_ERROR, self.SEVERITY_CRITICAL]
            
        self.alert_handlers[handler_id] = {
            'handler': handler_func,
            'severity_levels': severity_levels
        }
        
        logger.debug(f"Alert handler registered: {handler_id}")
    
    def unregister_alert_handler(self, handler_id: str) -> bool:
        """
        Unregister an alert handler.
        
        Args:
            handler_id: Handler identifier
            
        Returns:
            True if handler was unregistered, False if not found
        """
        if handler_id not in self.alert_handlers:
            return False
            
        del self.alert_handlers[handler_id]
        logger.debug(f"Alert handler unregistered: {handler_id}")
        return True
    
    def add_alert(self, component_id: str, message: str,
                 severity: str = SEVERITY_WARNING,
                 details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new alert to the queue.
        
        Args:
            component_id: Component that generated the alert
            message: Alert message
            severity: Alert severity level
            details: Additional alert details
        """
        if component_id not in self.components and component_id != 'system':
            logger.warning(f"Alert for unknown component: {component_id}")
            
        alert = {
            'component_id': component_id,
            'message': message,
            'severity': severity,
            'timestamp': datetime.datetime.now(),
            'details': details or {}
        }
        
        # Add alert to queue for processing
        with self.alert_queue_lock:
            self.alert_queue.append(alert)
            
            # Add to history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.alert_history_size:
                self.alert_history.pop(0)
                
        # Log the alert
        log_level = logging.INFO
        if severity == self.SEVERITY_WARNING:
            log_level = logging.WARNING
        elif severity in (self.SEVERITY_ERROR, self.SEVERITY_CRITICAL):
            log_level = logging.ERROR
            
        logger.log(log_level, f"Alert: [{severity}] {component_id} - {message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the current system health status.
        
        Returns:
            Dictionary with system health information
        """
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check component statuses
        components_ok = 0
        components_warning = 0
        components_error = 0
        components_critical = 0
        components_unknown = 0
        
        with self.metrics_lock:
            for component_id, status_info in self.component_statuses.items():
                status = status_info.get('status', 'UNKNOWN')
                if status == 'OK':
                    components_ok += 1
                elif status == 'WARNING':
                    components_warning += 1
                elif status == 'ERROR':
                    components_error += 1
                elif status == 'CRITICAL':
                    components_critical += 1
                else:
                    components_unknown += 1
        
        # Determine overall system status
        system_status = 'OK'
        if components_critical > 0 or cpu_percent > self.cpu_threshold or memory.percent > self.memory_threshold:
            system_status = 'CRITICAL'
        elif components_error > 0 or disk.percent > self.disk_threshold:
            system_status = 'ERROR'
        elif components_warning > 0:
            system_status = 'WARNING'
            
        return {
            'timestamp': datetime.datetime.now(),
            'status': system_status,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'boot_time': datetime.datetime.fromtimestamp(psutil.boot_time()),
                'platform': platform.platform(),
                'python_version': platform.python_version()
            },
            'components': {
                'total': len(self.components),
                'ok': components_ok,
                'warning': components_warning,
                'error': components_error,
                'critical': components_critical,
                'unknown': components_unknown
            },
            'alerts': {
                'queued': len(self.alert_queue),
                'history_size': len(self.alert_history)
            }
        }
    
    def get_component_status(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status for a specific component or all components.
        
        Args:
            component_id: Optional component ID to get status for
            
        Returns:
            Component status dictionary or all component statuses
        """
        with self.metrics_lock:
            if component_id:
                if component_id in self.component_statuses:
                    return dict(self.component_statuses[component_id])
                return {'status': 'UNKNOWN', 'message': 'Component not found'}
            else:
                # Return all component statuses
                result = {}
                for cid, status in self.component_statuses.items():
                    result[cid] = dict(status)
                return result
    
    def get_performance_metrics(self, component_id: Optional[str] = None,
                              metric_type: Optional[str] = None,
                              limit: int = 100) -> Dict[str, Any]:
        """
        Get performance metrics for components.
        
        Args:
            component_id: Optional component ID to filter
            metric_type: Optional metric type to filter
            limit: Maximum number of data points per metric
            
        Returns:
            Metrics dictionary
        """
        with self.metrics_lock:
            result = {}
            
            # Filter by component ID if provided
            if component_id:
                if component_id not in self.performance_metrics:
                    return {}
                    
                components_to_check = {component_id: self.performance_metrics[component_id]}
            else:
                components_to_check = self.performance_metrics
                
            # Process each component
            for cid, metrics in components_to_check.items():
                result[cid] = {}
                
                # Filter by metric type if provided
                if metric_type:
                    if metric_type in ('cpu_usage', 'memory_usage', 'execution_time'):
                        if metrics[metric_type]:
                            result[cid][metric_type] = metrics[metric_type][-limit:]
                    elif metric_type in metrics['custom_metrics']:
                        result[cid][metric_type] = metrics['custom_metrics'][metric_type][-limit:]
                else:
                    # Include standard metrics
                    for std_metric in ('cpu_usage', 'memory_usage', 'execution_time'):
                        if metrics[std_metric]:
                            result[cid][std_metric] = metrics[std_metric][-limit:]
                            
                    # Include custom metrics
                    for custom_metric, data in metrics['custom_metrics'].items():
                        if data:
                            result[cid][custom_metric] = data[-limit:]
                            
            return result
    
    def get_alerts(self, limit: int = 100, 
                  severity: Optional[str] = None, 
                  component_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts with optional filtering.
        
        Args:
            limit: Maximum number of alerts to return
            severity: Optional severity level to filter
            component_id: Optional component ID to filter
            
        Returns:
            List of alert dictionaries
        """
        with self.alert_queue_lock:
            filtered_alerts = []
            
            for alert in reversed(self.alert_history):
                if severity and alert['severity'] != severity:
                    continue
                if component_id and alert['component_id'] != component_id:
                    continue
                    
                filtered_alerts.append(dict(alert))
                
                if len(filtered_alerts) >= limit:
                    break
                    
            return filtered_alerts
    
    def _monitoring_loop(self, interval: int) -> None:
        """
        Main monitoring loop that runs in a separate thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        logger.info("Monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                self._check_system_resources()
                self._check_component_heartbeats()
                
                # Update monitor's own heartbeat
                self.component_heartbeat('system_monitor', 'OK', 'Monitoring active')
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                logger.debug(traceback.format_exc())
                
            # Sleep for the monitoring interval
            if not self.stop_event.wait(interval):
                continue
                
        logger.info("Monitoring loop stopped")
    
    def _alert_processing_loop(self) -> None:
        """
        Alert processing loop that runs in a separate thread.
        """
        logger.info("Alert processing loop started")
        
        while not self.stop_event.is_set():
            try:
                # Process any queued alerts
                alerts_to_process = []
                
                with self.alert_queue_lock:
                    if self.alert_queue:
                        alerts_to_process = list(self.alert_queue)
                        self.alert_queue.clear()
                        
                for alert in alerts_to_process:
                    self._process_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error in alert processing loop: {str(e)}")
                logger.debug(traceback.format_exc())
                
            # Sleep for a short interval
            if not self.stop_event.wait(1):
                continue
                
        logger.info("Alert processing loop stopped")
    
    def _check_system_resources(self) -> None:
        """
        Check system resource usage and generate alerts if thresholds exceeded.
        """
        # Check CPU usage
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.cpu_threshold:
            self.add_alert(
                'system',
                f"High CPU usage: {cpu_percent}%",
                severity=self.SEVERITY_WARNING if cpu_percent < 90 else self.SEVERITY_ERROR,
                details={'cpu_percent': cpu_percent, 'threshold': self.cpu_threshold}
            )
            
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold:
            self.add_alert(
                'system',
                f"High memory usage: {memory.percent}%",
                severity=self.SEVERITY_WARNING if memory.percent < 90 else self.SEVERITY_ERROR,
                details={'memory_percent': memory.percent, 'threshold': self.memory_threshold}
            )
            
        # Check disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > self.disk_threshold:
            self.add_alert(
                'system',
                f"High disk usage: {disk.percent}%",
                severity=self.SEVERITY_WARNING if disk.percent < 90 else self.SEVERITY_ERROR,
                details={'disk_percent': disk.percent, 'threshold': self.disk_threshold}
            )
            
        # Add system resource metrics
        self.add_performance_metric('system', 'cpu_usage', cpu_percent, {
            'per_cpu': psutil.cpu_percent(percpu=True)
        })
        
        self.add_performance_metric('system', 'memory_usage', memory.percent, {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used
        })
        
        self.add_performance_metric('system', 'disk_usage', disk.percent, {
            'total': disk.total,
            'free': disk.free,
            'used': disk.used
        })
    
    def _check_component_heartbeats(self) -> None:
        """
        Check component heartbeats and generate alerts for missing heartbeats.
        """
        now = datetime.datetime.now()
        
        with self.metrics_lock:
            for component_id, status in self.component_statuses.items():
                last_heartbeat = status.get('last_heartbeat')
                
                # Skip if no heartbeat recorded yet
                if last_heartbeat is None:
                    continue
                    
                # Check if heartbeat is too old
                time_since_heartbeat = (now - last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.component_timeout:
                    # Generate alert for the missing heartbeat
                    if status.get('status') != 'ERROR':
                        self.add_alert(
                            component_id,
                            f"Component heartbeat missing for {int(time_since_heartbeat)}s",
                            severity=self.SEVERITY_ERROR,
                            details={
                                'last_heartbeat': last_heartbeat.isoformat(),
                                'timeout_threshold': self.component_timeout
                            }
                        )
                        
                        # Update component status
                        status['status'] = 'ERROR'
                        status['message'] = f"No heartbeat in {int(time_since_heartbeat)}s"
    
    def _process_alert(self, alert: Dict[str, Any]) -> None:
        """
        Process an alert by sending it to registered handlers.
        
        Args:
            alert: Alert dictionary
        """
        severity = alert.get('severity', self.SEVERITY_WARNING)
        
        # Call each relevant alert handler
        for handler_id, handler_info in self.alert_handlers.items():
            if severity in handler_info.get('severity_levels', []):
                try:
                    handler_func = handler_info['handler']
                    handler_func(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler {handler_id}: {str(e)}")
                    
        # Send email alert if configured
        if self.email_alerts and severity in (self.SEVERITY_ERROR, self.SEVERITY_CRITICAL):
            self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """
        Send an email alert.
        
        Args:
            alert: Alert dictionary
        """
        if not self.email_config:
            return
            
        try:
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender = self.email_config.get('sender')
            recipients = self.email_config.get('recipients', [])
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            
            if not (smtp_server and sender and recipients):
                logger.warning("Incomplete email configuration, skipping email alert")
                return
                
            # Create the email
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            
            severity = alert.get('severity', 'WARNING')
            component = alert.get('component_id', 'Unknown')
            subject = f"[{severity}] {component} Alert - {alert.get('message', 'No message')}"
            
            msg['Subject'] = subject
            
            # Create email body
            body = f"""
            <html>
            <body>
                <h2>System Alert: {severity}</h2>
                <p><strong>Component:</strong> {component}</p>
                <p><strong>Message:</strong> {alert.get('message', 'No message')}</p>
                <p><strong>Time:</strong> {alert.get('timestamp', datetime.datetime.now()).isoformat()}</p>
                <h3>Details:</h3>
                <pre>{json.dumps(alert.get('details', {}), indent=2)}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to SMTP server and send
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                if username and password:
                    server.login(username, password)
                server.send_message(msg)
                
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive system health report.
        
        Returns:
            Report dictionary with system health information
        """
        report = {
            'timestamp': datetime.datetime.now(),
            'system_health': self.get_system_health(),
            'components': self.get_component_status(),
            'recent_alerts': self.get_alerts(limit=10),
        }
        
        # Add some aggregated performance metrics
        performance_data = {}
        with self.metrics_lock:
            for component_id, metrics in self.performance_metrics.items():
                performance_data[component_id] = {}
                
                # Get the most recent standard metrics
                for metric_type in ('cpu_usage', 'memory_usage', 'execution_time'):
                    metric_values = metrics[metric_type]
                    if metric_values:
                        latest = metric_values[-1]['value']
                        # Calculate average of last 5 values if available
                        recent = [m['value'] for m in metric_values[-5:]]
                        avg = sum(recent) / len(recent) if recent else 0
                        
                        performance_data[component_id][metric_type] = {
                            'latest': latest,
                            'average': avg
                        }
                
                # Include a small set of custom metrics
                custom_summary = {}
                for metric_name, values in metrics['custom_metrics'].items():
                    if values:
                        latest = values[-1]['value']
                        custom_summary[metric_name] = latest
                
                if custom_summary:
                    performance_data[component_id]['custom'] = custom_summary
        
        report['performance'] = performance_data
        
        return report 