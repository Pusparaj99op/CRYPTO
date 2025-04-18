import os
import json
import datetime
import shutil
import uuid
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Model versioning and registry system for ML models.
    Handles model storage, versioning, metadata tracking, and retrieval.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Base directory for the model registry
        """
        self.registry_path = registry_path
        self._ensure_registry_exists()
        
    def _ensure_registry_exists(self):
        """Create registry directory structure if it doesn't exist"""
        if not os.path.exists(self.registry_path):
            os.makedirs(self.registry_path, exist_ok=True)
            logger.info(f"Created model registry at {self.registry_path}")
            
    def register_model(self, 
                      model_name: str, 
                      model_path: str, 
                      metadata: Dict[str, Any],
                      version: Optional[str] = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model files
            metadata: Dictionary of model metadata
            version: Optional version string (if None, will generate based on timestamp)
            
        Returns:
            version_id: The version ID of the registered model
        """
        # Generate version if not provided
        if version is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            version = f"{timestamp}_{uuid.uuid4().hex[:8]}"
            
        # Create model directory if it doesn't exist
        model_dir = os.path.join(self.registry_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create version directory
        version_dir = os.path.join(model_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model files to registry
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                src_file = os.path.join(model_path, file)
                dst_file = os.path.join(version_dir, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
        else:
            # Single file case
            shutil.copy2(model_path, os.path.join(version_dir, os.path.basename(model_path)))
            
        # Add metadata
        metadata.update({
            "registered_at": datetime.datetime.now().isoformat(),
            "version": version
        })
        
        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Registered model {model_name} version {version}")
        return version
    
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> str:
        """
        Get the path to a registered model.
        
        Args:
            model_name: Name of the model
            version: Version string (if None, will use latest version)
            
        Returns:
            Path to the model directory
        """
        if version is None:
            version = self.get_latest_version(model_name)
            if version is None:
                raise ValueError(f"No versions found for model {model_name}")
                
        model_path = os.path.join(self.registry_path, model_name, version)
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_name} version {version} not found")
            
        return model_path
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string or None if no versions exist
        """
        model_dir = os.path.join(self.registry_path, model_name)
        if not os.path.exists(model_dir):
            return None
            
        versions = os.listdir(model_dir)
        if not versions:
            return None
            
        # Sort versions by creation time of the directory
        versions.sort(key=lambda v: os.path.getctime(os.path.join(model_dir, v)), reverse=True)
        return versions[0]
    
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a model version.
        
        Args:
            model_name: Name of the model
            version: Version string (if None, will use latest version)
            
        Returns:
            Dictionary of model metadata
        """
        model_path = self.get_model_path(model_name, version)
        metadata_file = os.path.join(model_path, "metadata.json")
        
        if not os.path.exists(metadata_file):
            raise ValueError(f"Metadata file not found for model {model_name} version {version or 'latest'}")
            
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def list_models(self) -> List[str]:
        """
        List all models in the registry.
        
        Returns:
            List of model names
        """
        if not os.path.exists(self.registry_path):
            return []
            
        return [d for d in os.listdir(self.registry_path) 
                if os.path.isdir(os.path.join(self.registry_path, d))]
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        model_dir = os.path.join(self.registry_path, model_name)
        if not os.path.exists(model_dir):
            return []
            
        return [d for d in os.listdir(model_dir) 
                if os.path.isdir(os.path.join(model_dir, d))]
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            True if deletion was successful
        """
        version_dir = os.path.join(self.registry_path, model_name, version)
        if not os.path.exists(version_dir):
            logger.warning(f"Model {model_name} version {version} not found, nothing to delete")
            return False
            
        shutil.rmtree(version_dir)
        logger.info(f"Deleted model {model_name} version {version}")
        return True
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model and all its versions.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if deletion was successful
        """
        model_dir = os.path.join(self.registry_path, model_name)
        if not os.path.exists(model_dir):
            logger.warning(f"Model {model_name} not found, nothing to delete")
            return False
            
        shutil.rmtree(model_dir)
        logger.info(f"Deleted model {model_name} and all versions")
        return True 