"""
Config Manager - Dynamic configuration management.

This module provides a dynamic configuration management system
that supports runtime configuration changes, validation, and persistence.
"""

import logging
import os
import json
import yaml
import threading
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Dynamic configuration management system that handles
    runtime configuration changes, validation, and persistence.
    """
    
    def __init__(self, config_path: Optional[str] = None, 
               environment: str = "development"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file or directory
            environment: Environment name (development, testing, production)
        """
        self.config_path = config_path
        self.environment = environment
        self.config = {}
        self.default_config = {}
        self.validators = {}
        self.callbacks = {}
        self.lock = threading.RLock()
        self.validator_lock = threading.Lock()
        self.last_modified = datetime.now()
        
        if config_path:
            self.load_config()
        
        logger.info(f"ConfigManager initialized with environment: {environment}")
    
    def load_config(self) -> bool:
        """
        Load configuration from config file or directory.
        
        Returns:
            True if config loaded successfully, False otherwise
        """
        if not self.config_path:
            logger.warning("No config path specified")
            return False
            
        try:
            with self.lock:
                if os.path.isdir(self.config_path):
                    # Load from directory - construct config from multiple files
                    self._load_from_dir()
                else:
                    # Load from single file
                    self._load_from_file(self.config_path)
                    
                # Set default configuration as a deep copy of the initial config
                self.default_config = copy.deepcopy(self.config)
                
                # Validate the loaded configuration
                self._validate_config()
                
                self.last_modified = datetime.now()
                
            logger.info(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def _load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a single file.
        
        Args:
            file_path: Path to the configuration file
        """
        if not os.path.exists(file_path):
            logger.warning(f"Config file not found: {file_path}")
            return
            
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ('.yaml', '.yml'):
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif file_extension == '.json':
            with open(file_path, 'r') as f:
                config_data = json.load(f)
        else:
            logger.warning(f"Unsupported config file format: {file_extension}")
            return
            
        # Extract environment-specific configuration if available
        if self.environment in config_data:
            env_config = config_data[self.environment]
            # Merge with common config if exists
            if "common" in config_data:
                self._deep_merge(config_data["common"], env_config)
            self.config = env_config
        else:
            # Use whole config if no environment segmentation
            self.config = config_data
    
    def _load_from_dir(self) -> None:
        """
        Load configuration from a directory containing multiple config files.
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Config directory not found: {self.config_path}")
            return
            
        # First load base config if it exists
        base_config_paths = [
            os.path.join(self.config_path, "config.yaml"),
            os.path.join(self.config_path, "config.yml"),
            os.path.join(self.config_path, "config.json"),
        ]
        
        for path in base_config_paths:
            if os.path.exists(path):
                self._load_from_file(path)
                break
                
        # Then load environment-specific config
        env_config_paths = [
            os.path.join(self.config_path, f"{self.environment}.yaml"),
            os.path.join(self.config_path, f"{self.environment}.yml"),
            os.path.join(self.config_path, f"{self.environment}.json"),
        ]
        
        for path in env_config_paths:
            if os.path.exists(path):
                env_config = {}
                file_extension = os.path.splitext(path)[1].lower()
                
                if file_extension in ('.yaml', '.yml'):
                    with open(path, 'r') as f:
                        env_config = yaml.safe_load(f)
                elif file_extension == '.json':
                    with open(path, 'r') as f:
                        env_config = json.load(f)
                        
                # Merge environment config with base config
                self._deep_merge(self.config, env_config)
                break
    
    def _deep_merge(self, source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, modifying destination in place.
        
        Args:
            source: Source dictionary
            destination: Destination dictionary to be updated
            
        Returns:
            Updated destination dictionary
        """
        for key, value in source.items():
            if key in destination and isinstance(destination[key], dict) and isinstance(value, dict):
                # If both values are dicts, merge them recursively
                self._deep_merge(value, destination[key])
            else:
                # Otherwise just override
                destination[key] = value
                
        return destination
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value for a given key.
        
        Args:
            key: Configuration key with dot notation for nested values (e.g. 'database.host')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        with self.lock:
            if '.' in key:
                # Handle nested keys
                parts = key.split('.')
                value = self.config
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return default
                return value
            else:
                # Simple key
                return self.config.get(key, default)
    
    def set(self, key: str, value: Any, persist: bool = False) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            value: Configuration value
            persist: Whether to persist the change to disk
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            old_value = self.get(key)
            
            # Don't do anything if value is the same
            if old_value == value:
                return True
                
            if '.' in key:
                # Handle nested keys
                parts = key.split('.')
                config_ref = self.config
                
                # Navigate to the parent of the final key
                for part in parts[:-1]:
                    if part not in config_ref:
                        config_ref[part] = {}
                    config_ref = config_ref[part]
                    
                # Set the value on the last key
                config_ref[parts[-1]] = value
            else:
                # Simple key
                self.config[key] = value
                
            # Validate the changed configuration
            if not self._validate_key(key, value):
                # Revert to old value if validation fails
                if old_value is not None:
                    self.set(key, old_value, False)
                else:
                    # Remove the key if it was newly added
                    if '.' in key:
                        parts = key.split('.')
                        config_ref = self.config
                        for part in parts[:-1]:
                            config_ref = config_ref[part]
                        if parts[-1] in config_ref:
                            del config_ref[parts[-1]]
                    elif key in self.config:
                        del self.config[key]
                        
                return False
                
            # Call callbacks for this key
            self._notify_callbacks(key, value, old_value)
            
            # Persist changes if requested
            if persist and self.config_path and not os.path.isdir(self.config_path):
                self._save_config()
                
            self.last_modified = datetime.now()
            return True
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset configuration to default values.
        
        Args:
            key: Specific key to reset, or None to reset all
        """
        with self.lock:
            if key is None:
                # Reset entire config
                self.config = copy.deepcopy(self.default_config)
                logger.info("Reset all configuration to defaults")
            else:
                # Reset specific key
                default_value = self.get_default(key)
                if default_value is not None:
                    self.set(key, default_value)
                    logger.info(f"Reset configuration key '{key}' to default")
                else:
                    # Remove key if it doesn't exist in defaults
                    self.remove(key)
                    logger.info(f"Removed non-default configuration key '{key}'")
    
    def get_default(self, key: str, default: Any = None) -> Any:
        """
        Get the default value for a configuration key.
        
        Args:
            key: Configuration key
            default: Default value if key isn't in default config
            
        Returns:
            Default configuration value
        """
        with self.lock:
            if '.' in key:
                # Handle nested keys
                parts = key.split('.')
                value = self.default_config
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return default
                return value
            else:
                # Simple key
                return self.default_config.get(key, default)
    
    def remove(self, key: str, persist: bool = False) -> bool:
        """
        Remove a configuration key.
        
        Args:
            key: Configuration key to remove
            persist: Whether to persist the change
            
        Returns:
            True if key was removed, False otherwise
        """
        with self.lock:
            old_value = self.get(key)
            
            if old_value is None:
                # Key doesn't exist
                return False
                
            if '.' in key:
                # Handle nested keys
                parts = key.split('.')
                config_ref = self.config
                
                # Navigate to the parent of the final key
                for part in parts[:-1]:
                    if part not in config_ref:
                        return False
                    config_ref = config_ref[part]
                    
                # Remove the key
                if parts[-1] in config_ref:
                    del config_ref[parts[-1]]
                else:
                    return False
            else:
                # Simple key
                if key in self.config:
                    del self.config[key]
                else:
                    return False
                    
            # Call callbacks for this key
            self._notify_callbacks(key, None, old_value)
            
            # Persist changes if requested
            if persist and self.config_path and not os.path.isdir(self.config_path):
                self._save_config()
                
            self.last_modified = datetime.now()
            return True
    
    def register_validator(self, key_pattern: str, validator_func: callable) -> None:
        """
        Register a validation function for configuration keys.
        
        Args:
            key_pattern: Key pattern to validate (can use * as wildcard)
            validator_func: Function taking (key, value) and returning bool
        """
        with self.validator_lock:
            self.validators[key_pattern] = validator_func
            logger.debug(f"Registered validator for key pattern: {key_pattern}")
    
    def register_callback(self, key_pattern: str, callback_func: callable) -> None:
        """
        Register a callback for configuration changes.
        
        Args:
            key_pattern: Key pattern to watch (can use * as wildcard)
            callback_func: Function taking (key, new_value, old_value)
        """
        with self.lock:
            if key_pattern not in self.callbacks:
                self.callbacks[key_pattern] = []
                
            self.callbacks[key_pattern].append(callback_func)
            logger.debug(f"Registered change callback for key pattern: {key_pattern}")
    
    def unregister_callback(self, key_pattern: str, callback_func: callable) -> bool:
        """
        Unregister a callback function.
        
        Args:
            key_pattern: Key pattern the callback was registered for
            callback_func: The callback function to unregister
            
        Returns:
            True if callback was removed, False otherwise
        """
        with self.lock:
            if key_pattern in self.callbacks:
                if callback_func in self.callbacks[key_pattern]:
                    self.callbacks[key_pattern].remove(callback_func)
                    logger.debug(f"Unregistered callback for key pattern: {key_pattern}")
                    
                    # Clean up empty callback lists
                    if not self.callbacks[key_pattern]:
                        del self.callbacks[key_pattern]
                        
                    return True
                    
            return False
    
    def _validate_config(self) -> bool:
        """
        Validate the entire configuration.
        
        Returns:
            True if all validations pass, False otherwise
        """
        with self.validator_lock:
            # If no validators, consider it valid
            if not self.validators:
                return True
                
            # Flatten config for easier validation
            flat_config = self._flatten_dict(self.config)
            
            # Apply all validators
            for key, value in flat_config.items():
                if not self._validate_key(key, value):
                    return False
                    
            return True
    
    def _validate_key(self, key: str, value: Any) -> bool:
        """
        Validate a single configuration key-value pair.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            True if validation passes, False otherwise
        """
        with self.validator_lock:
            # If no validators, consider it valid
            if not self.validators:
                return True
                
            for pattern, validator in self.validators.items():
                if self._key_matches_pattern(key, pattern):
                    try:
                        if not validator(key, value):
                            logger.warning(f"Validation failed for {key}={value} with pattern {pattern}")
                            return False
                    except Exception as e:
                        logger.error(f"Error in validator for {key}: {str(e)}")
                        return False
                        
            return True
    
    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """
        Check if a key matches a pattern with wildcards.
        
        Args:
            key: Configuration key
            pattern: Pattern with optional * wildcards
            
        Returns:
            True if key matches pattern, False otherwise
        """
        if pattern == '*':
            return True
            
        if '*' not in pattern:
            return key == pattern
            
        parts = pattern.split('*')
        if parts[0] and not key.startswith(parts[0]):
            return False
            
        if parts[-1] and not key.endswith(parts[-1]):
            return False
            
        current_pos = 0
        for part in parts:
            if part:
                pos = key.find(part, current_pos)
                if pos == -1:
                    return False
                current_pos = pos + len(part)
                
        return True
    
    def _notify_callbacks(self, key: str, new_value: Any, old_value: Any) -> None:
        """
        Notify callbacks about a configuration change.
        
        Args:
            key: Changed configuration key
            new_value: New configuration value
            old_value: Previous configuration value
        """
        with self.lock:
            for pattern, callbacks in self.callbacks.items():
                if self._key_matches_pattern(key, pattern):
                    for callback in callbacks:
                        try:
                            callback(key, new_value, old_value)
                        except Exception as e:
                            logger.error(f"Error in callback for {key}: {str(e)}")
    
    def _save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self.config_path:
                logger.warning("No config path for saving")
                return False
                
            # Create backup of current file
            if os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.bak"
                try:
                    with open(self.config_path, 'r') as src, open(backup_path, 'w') as dst:
                        dst.write(src.read())
                except Exception as e:
                    logger.warning(f"Could not create config backup: {str(e)}")
                    
            # Write new config
            file_extension = os.path.splitext(self.config_path)[1].lower()
            
            if file_extension in ('.yaml', '.yml'):
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif file_extension == '.json':
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.warning(f"Unsupported config file format for saving: {file_extension}")
                return False
                
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator character
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
                
        return dict(items)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Deep copy of the configuration dictionary
        """
        with self.lock:
            return copy.deepcopy(self.config)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section: Section key
            
        Returns:
            Deep copy of the section or empty dict if section doesn't exist
        """
        with self.lock:
            section_config = self.config.get(section, {})
            return copy.deepcopy(section_config)
    
    def set_many(self, config_dict: Dict[str, Any], prefix: str = '', 
                persist: bool = False) -> bool:
        """
        Set multiple configuration values at once.
        
        Args:
            config_dict: Dictionary of configuration values
            prefix: Prefix for all keys in the dictionary
            persist: Whether to persist changes
            
        Returns:
            True if all values were set successfully
        """
        all_success = True
        
        with self.lock:
            flat_dict = self._flatten_dict(config_dict)
            
            for key, value in flat_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if not self.set(full_key, value, False):
                    all_success = False
                    
            if persist and all_success and self.config_path and not os.path.isdir(self.config_path):
                self._save_config()
                
        return all_success
    
    def import_env_vars(self, prefix: str, lowercase: bool = True) -> int:
        """
        Import configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix to match
            lowercase: Whether to convert keys to lowercase
            
        Returns:
            Number of imported variables
        """
        count = 0
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase if needed
                config_key = key[len(prefix):]
                if config_key.startswith('_'):
                    config_key = config_key[1:]
                    
                if lowercase:
                    config_key = config_key.lower()
                    
                # Convert all _ to . for nested keys
                config_key = config_key.replace('_', '.')
                
                # Try to interpret value type
                parsed_value = self._parse_env_value(value)
                
                # Set the value
                if self.set(config_key, parsed_value):
                    count += 1
                    
        logger.info(f"Imported {count} configuration values from environment variables")
        return count
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value into appropriate type.
        
        Args:
            value: Environment variable value
            
        Returns:
            Parsed value
        """
        # Check for boolean
        if value.lower() in ('true', 'yes', 'y', '1'):
            return True
        if value.lower() in ('false', 'no', 'n', '0'):
            return False
            
        # Check for None/null
        if value.lower() in ('none', 'null'):
            return None
            
        # Check for integer
        try:
            return int(value)
        except ValueError:
            pass
            
        # Check for float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Check for JSON
        if (value.startswith('{') and value.endswith('}')) or \
           (value.startswith('[') and value.endswith(']')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
        # Default to string
        return value
    
    def get_modified_timestamp(self) -> datetime:
        """
        Get the last modification timestamp.
        
        Returns:
            Datetime of last configuration change
        """
        return self.last_modified 