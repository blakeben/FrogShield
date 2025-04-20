import yaml
import logging
import os
from copy import deepcopy

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    'InputValidator': {
        'context_window': 5
    },
    'RealtimeMonitor': {
        'sensitivity_threshold': 0.8,
        'initial_avg_length': 50,
        'behavior_monitoring_factor': 2
    },
    'TextAnalysis': {
        'syntax_non_alnum_threshold': 0.3,
        'syntax_max_word_length': 50
    }
}

_CONFIG = None

def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file, merging with defaults."""
    global _CONFIG
    config = deepcopy(_DEFAULT_CONFIG) # Start with defaults

    # Use absolute path for config relative to project root if relative path given
    if not os.path.isabs(config_path):
        # Assuming this script is run from project root or similar
        # A more robust approach might involve finding project root dynamically
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')) # Go up two levels from utils
        config_path = os.path.join(base_path, config_path)

    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found at {config_path}. Using default settings.")
        _CONFIG = config
        return _CONFIG

    try:
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        if file_config:
            # Merge file config into defaults (simple merge, nested dicts not deeply merged here)
            for section, settings in file_config.items():
                if section in config:
                    config[section].update(settings)
                else:
                    config[section] = settings # Add new sections if any
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Configuration file {config_path} is empty. Using default settings.")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}. Using default settings.", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration file {config_path}: {e}. Using default settings.", exc_info=True)

    _CONFIG = config
    return _CONFIG

def get_config(config_path='config.yaml'):
    """Returns the loaded configuration, loading it if necessary."""
    global _CONFIG
    if _CONFIG is None:
        load_config(config_path)
    return _CONFIG 