"""Handles loading and accessing configuration settings for FrogShield.

Provides functions to load configuration from a YAML file, merging it with
default settings, and access the configuration globally.

Author: Ben Blake <ben.blake@tcu.edu>
Contributor: Tanner Hendrix <t.hendrix@tcu.edu>
"""
import yaml
import logging
import os
from copy import deepcopy

logger = logging.getLogger(__name__)

# Default configuration values
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

# Global variable to hold the loaded configuration (initialized lazily)
_CONFIG = None


def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file, merging it with defaults.

    Reads the specified YAML configuration file, merges its content with
    the default settings defined in `_DEFAULT_CONFIG`. If the file is not found
    or is invalid, defaults are used, and appropriate warnings/errors are logged.
    The resulting configuration is stored globally in `_CONFIG`.

    Args:
        config_path (str, optional): Path to the configuration file.
            Defaults to 'config.yaml'. If a relative path is given, it's
            resolved relative to the current working directory.

    Returns:
        dict: The loaded and merged configuration dictionary.
    """
    global _CONFIG
    # Start with a deep copy of the defaults to avoid modifying the original
    config = deepcopy(_DEFAULT_CONFIG)

    # Resolve config_path relative to the current working directory
    absolute_config_path = os.path.abspath(config_path)

    if not os.path.exists(absolute_config_path):
        logger.warning(f"Configuration file not found at {absolute_config_path}. "
                       f"Using default settings.")
        _CONFIG = config
        return _CONFIG

    try:
        with open(absolute_config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        if file_config:
            # Merge file config into defaults
            # Note: This is a simple merge; nested dicts are not deeply merged.
            for section, settings in file_config.items():
                if section in config and isinstance(config[section], dict) and isinstance(settings, dict):
                    config[section].update(settings)
                else:
                    # Add new sections or overwrite non-dict sections
                    config[section] = settings
            logger.info(f"Loaded configuration from {absolute_config_path}")
        else:
            logger.warning(f"Configuration file {absolute_config_path} is empty. "
                           f"Using default settings.")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {absolute_config_path}: {e}. "
                       f"Using default settings.", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration file {absolute_config_path}: {e}. "
                       f"Using default settings.", exc_info=True)

    _CONFIG = config
    return _CONFIG


def get_config(config_path='config.yaml'):
    """Returns the loaded configuration, loading it lazily if needed.

    Provides access to the global configuration dictionary (`_CONFIG`).
    If the configuration hasn't been loaded yet (i.e., `_CONFIG` is None),
    it calls `load_config` first using the provided `config_path`.

    Args:
        config_path (str, optional): The path to the configuration file,
            used only if the configuration needs to be loaded for the first time.
            Defaults to 'config.yaml'.

    Returns:
        dict: The globally loaded configuration dictionary.
    """
    global _CONFIG
    if _CONFIG is None:
        load_config(config_path)
    return _CONFIG 