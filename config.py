"""
Configuration loader for the carbon credit optimization problem.
Loads global constants from config.json - never hardcode values.
"""

import json
from pathlib import Path
from typing import Any, Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from config.json file.
    
    Args:
        config_path: Path to config.json. Defaults to data/config.json
        
    Returns:
        Dictionary containing all configuration parameters
    """
    if config_path is None:
        config_path = DATA_DIR / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


class Config:
    """
    Singleton configuration class for easy access to global constants.
    """
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        """Load configuration on first instantiation."""
        try:
            self._config = load_config()
        except FileNotFoundError:
            print("Warning: config.json not found. Using empty configuration.")
            self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config
    
    @property
    def all(self) -> Dict[str, Any]:
        """Return the full configuration dictionary."""
        return self._config.copy()


# Create singleton instance for import
config = Config()
