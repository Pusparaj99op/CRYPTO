"""
Trade scheduler module for 15-minute trade execution scheduling.
Handles scheduling, timing, and coordination of trade executions.
"""
import time
import datetime
import logging
import threading
from typing import Dict, List, Callable, Optional

logger = logging.getLogger(__name__)

class TradeScheduler:
    """Scheduler for periodic trade execution and monitoring."""
    
    def __init__(self, interval_minutes: int = 15):
        """
        Initialize the trade scheduler.
        
        Args:
            interval_minutes: Time between scheduled trade executions in minutes
        """
        self.interval_minutes = interval_minutes
        self.scheduled_tasks: Dict[str, Dict] = {}
        self.running = False
        self.scheduler_thread = None
        self._lock = threading.Lock()
        
    def add_task(self, task_id: str, callback: Callable, 
                 assets: List[str] = None, 
                 start_time: Optional[datetime.time] = None,
                 end_time: Optional[datetime.time] = None) -> bool:
        """
        Add a scheduled trading task.
        
        Args:
            task_id: Unique identifier for the task
            callback: Function to call when task is triggered
            assets: List of assets to trade
            start_time: Optional time to start scheduling (None = anytime)
            end_time: Optional time to stop scheduling (None = anytime)
            
        Returns:
            Success status of adding the task
        """
        with self._lock:
            if task_id in self.scheduled_tasks:
                logger.warning(f"Task ID {task_id} already exists")
                return False
                
            self.scheduled_tasks[task_id] = {
                'callback': callback,
                'assets': assets or [],
                'start_time': start_time,
                'end_time': end_time,
                'last_run': None
            }
            logger.info(f"Added scheduled task: {task_id}")
            return True
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a scheduled task.
        
        Args:
            task_id: Identifier of task to remove
            
        Returns:
            Success status of removing the task
        """
        with self._lock:
            if task_id not in self.scheduled_tasks:
                logger.warning(f"Task ID {task_id} not found")
                return False
                
            del self.scheduled_tasks[task_id]
            logger.info(f"Removed scheduled task: {task_id}")
            return True
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Trade scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
            
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Trade scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            now = datetime.datetime.now()
            
            # Check if it's a 15-minute boundary (0, 15, 30, 45 minutes)
            if now.minute % self.interval_minutes == 0 and now.second < 10:
                self._execute_tasks(now)
                
                # Sleep until the next cycle (minus a few seconds to ensure we don't miss it)
                time.sleep((self.interval_minutes * 60) - 10)
            else:
                # Sleep for a second and check again
                time.sleep(1)
    
    def _execute_tasks(self, now: datetime.datetime):
        """
        Execute scheduled tasks.
        
        Args:
            now: Current datetime
        """
        current_time = now.time()
        
        for task_id, task in self.scheduled_tasks.items():
            # Check time constraints
            if task['start_time'] and current_time < task['start_time']:
                continue
                
            if task['end_time'] and current_time > task['end_time']:
                continue
                
            # Check if already run in this interval
            if task['last_run'] and (now - task['last_run']).total_seconds() < (self.interval_minutes * 60 - 30):
                continue
                
            try:
                logger.info(f"Executing scheduled task: {task_id}")
                task['callback'](task_id, task['assets'])
                task['last_run'] = now
            except Exception as e:
                logger.error(f"Error executing task {task_id}: {str(e)}")

def get_next_schedule_time(interval_minutes: int = 15) -> datetime.datetime:
    """
    Calculate the next schedule time based on 15-minute intervals.
    
    Args:
        interval_minutes: Interval in minutes
        
    Returns:
        Datetime of next schedule
    """
    now = datetime.datetime.now()
    minutes_to_add = interval_minutes - (now.minute % interval_minutes)
    
    if minutes_to_add == interval_minutes and now.second == 0:
        return now
    
    next_time = now + datetime.timedelta(minutes=minutes_to_add)
    return next_time.replace(second=0, microsecond=0) 