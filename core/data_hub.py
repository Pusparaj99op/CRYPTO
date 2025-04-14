"""
Data Hub - Unified data collection and management.

This module serves as the central point for all data collection, processing,
and storage within the trading system. It handles market data, news, social media,
on-chain data, and other data sources.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import json
import os

logger = logging.getLogger(__name__)

class DataHub:
    """
    Unified data collection and management system that serves as the 
    central point for all data operations in the trading system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataHub with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_sources = {}
        self.data_cache = {}
        self.data_subscriptions = {}
        self.update_intervals = {}
        self.last_updated = {}
        self.lock = threading.RLock()
        self.update_thread = None
        self.stop_event = threading.Event()
        self.update_queue = queue.Queue()
        
        # Configure data storage
        self.storage_directory = config.get('storage_directory', 'data')
        os.makedirs(self.storage_directory, exist_ok=True)
        
        logger.info("DataHub initialized")
        
    def register_data_source(self, source_id: str, source_instance: Any, 
                            update_interval: int = 3600) -> None:
        """
        Register a data source with the hub.
        
        Args:
            source_id: Unique identifier for the data source
            source_instance: Data source object/client
            update_interval: Update interval in seconds
        """
        with self.lock:
            self.data_sources[source_id] = source_instance
            self.update_intervals[source_id] = update_interval
            self.last_updated[source_id] = 0  # Initialize with 0 to force first update
            
        logger.info(f"Registered data source: {source_id} with update interval {update_interval}s")
    
    def subscribe(self, subscriber_id: str, data_type: str, 
                 callback: Optional[callable] = None) -> None:
        """
        Subscribe to a data type for updates.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            data_type: Type of data to subscribe to
            callback: Optional callback function for updates
        """
        with self.lock:
            if data_type not in self.data_subscriptions:
                self.data_subscriptions[data_type] = {}
            
            self.data_subscriptions[data_type][subscriber_id] = callback
            
        logger.debug(f"Subscription added: {subscriber_id} for {data_type}")
    
    def unsubscribe(self, subscriber_id: str, data_type: Optional[str] = None) -> None:
        """
        Unsubscribe from data updates.
        
        Args:
            subscriber_id: Subscriber ID to unsubscribe
            data_type: Optional data type to unsubscribe from (None for all)
        """
        with self.lock:
            if data_type is None:
                # Unsubscribe from all data types
                for dt in self.data_subscriptions:
                    if subscriber_id in self.data_subscriptions[dt]:
                        del self.data_subscriptions[dt][subscriber_id]
            elif data_type in self.data_subscriptions:
                if subscriber_id in self.data_subscriptions[data_type]:
                    del self.data_subscriptions[data_type][subscriber_id]
                    
        logger.debug(f"Unsubscribed: {subscriber_id} from {data_type or 'all data types'}")
    
    def start_data_collection(self) -> None:
        """
        Start the automatic data collection process.
        """
        if self.update_thread is not None and self.update_thread.is_alive():
            logger.warning("Data collection already running")
            return
            
        self.stop_event.clear()
        self.update_thread = threading.Thread(target=self._data_collection_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Started data collection process")
    
    def stop_data_collection(self) -> None:
        """
        Stop the automatic data collection process.
        """
        if self.update_thread is None or not self.update_thread.is_alive():
            logger.warning("Data collection not running")
            return
            
        self.stop_event.set()
        self.update_thread.join(timeout=10)
        self.update_thread = None
        
        logger.info("Stopped data collection process")
    
    def _data_collection_loop(self) -> None:
        """
        Main loop for automatic data collection.
        """
        logger.info("Data collection loop started")
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check which data sources need updating
                for source_id, interval in self.update_intervals.items():
                    last_update = self.last_updated.get(source_id, 0)
                    if current_time - last_update >= interval:
                        self.update_queue.put(source_id)
                
                # Process the update queue
                while not self.update_queue.empty():
                    try:
                        source_id = self.update_queue.get(block=False)
                        self._update_data_source(source_id)
                        self.update_queue.task_done()
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error updating {source_id}: {str(e)}")
                
                # Sleep for a short interval to prevent CPU overuse
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {str(e)}")
                time.sleep(5)  # Wait longer on error
                
        logger.info("Data collection loop stopped")
    
    def _update_data_source(self, source_id: str) -> None:
        """
        Update data from a specific source.
        
        Args:
            source_id: ID of the data source to update
        """
        if source_id not in self.data_sources:
            logger.warning(f"Unknown data source: {source_id}")
            return
            
        logger.debug(f"Updating data from source: {source_id}")
        
        try:
            source = self.data_sources[source_id]
            
            # Call the appropriate update method based on source type
            if hasattr(source, 'get_latest_data'):
                data = source.get_latest_data()
            elif hasattr(source, 'fetch_data'):
                data = source.fetch_data()
            else:
                logger.warning(f"Source {source_id} has no standard data retrieval method")
                return
                
            if data is not None:
                # Update cache and notify subscribers
                self._update_cache(source_id, data)
                self._notify_subscribers(source_id, data)
                
            # Update the last updated timestamp
            with self.lock:
                self.last_updated[source_id] = time.time()
                
        except Exception as e:
            logger.error(f"Error updating data from {source_id}: {str(e)}")
    
    def _update_cache(self, source_id: str, data: Any) -> None:
        """
        Update the data cache with new data.
        
        Args:
            source_id: Data source ID
            data: New data to cache
        """
        with self.lock:
            self.data_cache[source_id] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
        # Save to persistent storage if configured
        if self.config.get('persist_data', False):
            self._save_to_storage(source_id, data)
            
        logger.debug(f"Updated cache for {source_id}")
    
    def _notify_subscribers(self, source_id: str, data: Any) -> None:
        """
        Notify subscribers of data updates.
        
        Args:
            source_id: Data source ID
            data: New data
        """
        if source_id not in self.data_subscriptions:
            return
            
        for subscriber_id, callback in self.data_subscriptions[source_id].items():
            try:
                if callback is not None:
                    callback(source_id, data)
            except Exception as e:
                logger.error(f"Error notifying subscriber {subscriber_id}: {str(e)}")
    
    def _save_to_storage(self, source_id: str, data: Any) -> None:
        """
        Save data to persistent storage.
        
        Args:
            source_id: Data source ID
            data: Data to save
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.storage_directory}/{source_id}_{timestamp}.json"
            
            # Convert to appropriate format for storage
            if isinstance(data, pd.DataFrame):
                data_to_save = data.to_json(orient='records')
            elif isinstance(data, (dict, list)):
                data_to_save = json.dumps(data)
            else:
                logger.warning(f"Unsupported data type for storage: {type(data)}")
                return
                
            with open(filename, 'w') as f:
                f.write(data_to_save)
                
            logger.debug(f"Saved {source_id} data to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving data to storage: {str(e)}")
    
    def get_latest_data(self, source_id: str) -> Optional[Any]:
        """
        Get the latest data from a specific source.
        
        Args:
            source_id: Data source ID
            
        Returns:
            Latest data or None if not available
        """
        with self.lock:
            if source_id in self.data_cache:
                return self.data_cache[source_id]['data']
            return None
    
    def get_historical_data(self, source_id: str, 
                           start_time: datetime, 
                           end_time: Optional[datetime] = None) -> Optional[Any]:
        """
        Get historical data for a specific time range.
        
        Args:
            source_id: Data source ID
            start_time: Start time for historical data
            end_time: Optional end time (defaults to now)
            
        Returns:
            Historical data or None if not available
        """
        if end_time is None:
            end_time = datetime.now()
            
        # Check if the source supports historical data retrieval
        source = self.data_sources.get(source_id)
        if source is None:
            logger.warning(f"Unknown data source: {source_id}")
            return None
            
        # Try to get historical data from the source
        if hasattr(source, 'get_historical_data'):
            try:
                return source.get_historical_data(start_time, end_time)
            except Exception as e:
                logger.error(f"Error fetching historical data from {source_id}: {str(e)}")
                return None
                
        # If not supported by the source, try to load from storage
        return self._load_from_storage(source_id, start_time, end_time)
    
    def _load_from_storage(self, source_id: str, 
                          start_time: datetime, 
                          end_time: datetime) -> Optional[Any]:
        """
        Load historical data from storage.
        
        Args:
            source_id: Data source ID
            start_time: Start time
            end_time: End time
            
        Returns:
            Historical data from storage or None
        """
        try:
            # Generate file name pattern
            start_str = start_time.strftime('%Y%m%d')
            end_str = end_time.strftime('%Y%m%d')
            
            # List all relevant files
            files = []
            for filename in os.listdir(self.storage_directory):
                if filename.startswith(f"{source_id}_") and filename.endswith(".json"):
                    file_date_str = filename.split('_')[1].split('.')[0]
                    if start_str <= file_date_str <= end_str:
                        files.append(os.path.join(self.storage_directory, filename))
                        
            if not files:
                logger.warning(f"No historical data files found for {source_id} in range {start_str} to {end_str}")
                return None
                
            # Load and combine data from files
            all_data = []
            for file_path in sorted(files):
                with open(file_path, 'r') as f:
                    data = json.loads(f.read())
                    all_data.append(data)
                    
            # Combine data - assume list format for simplicity
            if all_data:
                if isinstance(all_data[0], list):
                    combined_data = [item for sublist in all_data for item in sublist]
                    return combined_data
                else:
                    return all_data[-1]  # Return most recent if not list
                    
            return None
            
        except Exception as e:
            logger.error(f"Error loading historical data from storage: {str(e)}")
            return None
    
    def force_update(self, source_id: str) -> bool:
        """
        Force an immediate update from a data source.
        
        Args:
            source_id: Data source ID to update
            
        Returns:
            Whether the update was successful
        """
        if source_id not in self.data_sources:
            logger.warning(f"Unknown data source: {source_id}")
            return False
            
        try:
            # Update the data source
            self._update_data_source(source_id)
            return True
        except Exception as e:
            logger.error(f"Error forcing update for {source_id}: {str(e)}")
            return False
    
    def get_data_status(self) -> Dict[str, Any]:
        """
        Get status information about all data sources.
        
        Returns:
            Status dictionary with information about data sources
        """
        with self.lock:
            status = {}
            current_time = time.time()
            
            for source_id in self.data_sources:
                last_update = self.last_updated.get(source_id, 0)
                next_update = last_update + self.update_intervals.get(source_id, 3600)
                
                status[source_id] = {
                    'last_updated': datetime.fromtimestamp(last_update) if last_update > 0 else None,
                    'next_update': datetime.fromtimestamp(next_update) if last_update > 0 else None,
                    'update_interval': self.update_intervals.get(source_id, 3600),
                    'data_available': source_id in self.data_cache,
                    'subscriber_count': len(self.data_subscriptions.get(source_id, {}))
                }
                
            return status 