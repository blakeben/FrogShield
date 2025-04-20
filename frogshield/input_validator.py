"""
Defines the InputValidator class for FrogShield.
Contributors:
    - Ben Blake <ben.blake@tcu.edu>
    - Tanner Hendrix <t.hendrix@tcu.edu>
"""
import logging
import os
from .utils.text_analysis import analyze_syntax, analyze_context
from .utils import config_loader

logger = logging.getLogger(__name__)

# Default path relative to this file's directory
_DEFAULT_PATTERNS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'patterns.txt'))

class InputValidator:
    """
    Validates user input against known prompt injection patterns and techniques.
    """
    def __init__(self, context_window, patterns=None, patterns_file=None):
        """
        Initializes the InputValidator.

        Args:
            context_window (int): The number of previous messages to consider for context analysis.
            patterns (list, optional): A list of known malicious patterns (regex or simple strings).
                                       If provided, this list is used directly.
            patterns_file (str, optional): Path to a file containing patterns (one per line).
                                          If provided, patterns are loaded from this file.
                                          If both `patterns` and `patterns_file` are None,
                                          attempts to load from default path (`frogshield/patterns.txt`).
        """
        if patterns is not None:
            self.patterns = patterns
            logger.info(f"InputValidator initialized with {len(self.patterns)} patterns provided directly.")
        else:
            load_path = patterns_file if patterns_file is not None else _DEFAULT_PATTERNS_FILE
            self.patterns = self._load_patterns_from_file(load_path)

        # Ensure context_window is a positive integer
        if not isinstance(context_window, int) or context_window < 0:
            raise ValueError("context_window must be a non-negative integer")
        self.context_window = context_window
        self.conversation_history = [] # Stores (user_input, llm_response) tuples

    def _load_patterns_from_file(self, filepath):
        """Loads patterns from a file, one pattern per line."""
        patterns = []
        try:
            # Ensure path exists
            if not os.path.exists(filepath):
                logger.warning(f"Patterns file not found at specified path: {filepath}. Using empty patterns list.")
                return []
            with open(filepath, 'r') as f:
                for line in f:
                    pattern = line.strip()
                    if pattern and not pattern.startswith('#'): # Ignore empty lines and comments
                        patterns.append(pattern)
            logger.info(f"InputValidator loaded {len(patterns)} patterns from {filepath}.")
        except Exception as e:
            logger.error(f"Failed to load patterns from {filepath}: {e}. Using empty patterns list.", exc_info=True)
            patterns = []
        return patterns

    def _match_patterns(self, text):
        """Checks if the text matches any known malicious patterns."""
        for pattern in self.patterns:
            # Basic string matching for demonstration
            # In a real implementation, use regex or more sophisticated matching
            if pattern in text:
                logger.debug(f"Potential injection detected: Matched pattern '{pattern}'")
                return True
        return False

    def _analyze_syntax(self, text):
        """Analyzes the syntax for unusual formatting or hidden instructions."""
        # Placeholder for syntax analysis logic
        config = config_loader.get_config() # Fetch config here
        is_unusual = analyze_syntax(text, config) # Pass config to function
        if is_unusual:
            logger.debug(f"Potential injection detected: Unusual syntax found.")
            return True
        return False

    def _filter_context(self, text):
        """Analyzes the conversation history for context manipulation attempts."""
        # Placeholder for context-aware filtering logic
        history_slice = self.conversation_history[-self.context_window:] if self.conversation_history else []
        is_manipulation = analyze_context(text, history_slice)
        if is_manipulation:
            logger.debug(f"Potential injection detected: Context manipulation suspected.")
            return True
        return False

    def validate(self, user_input, conversation_history=None):
        """
        Performs all validation checks on the user input.

        Args:
            user_input (str): The input text from the user.
            conversation_history (list, optional): The recent conversation history.
                                                   If provided, updates the internal history.
                                                   If None, uses the existing internal history.

        Returns:
            bool: True if the input is considered potentially malicious, False otherwise.
        """
        if conversation_history is not None:
            # Update internal history if an external one is provided.
            # This allows external management of the conversation flow if needed.
            self.conversation_history = conversation_history

        # Check against all validation methods
        return (
            self._match_patterns(user_input) or
            self._analyze_syntax(user_input) or
            self._filter_context(user_input)
        )

    def add_to_history(self, user_input, llm_response):
        """Adds the latest exchange to the internal conversation history."""
        self.conversation_history.append((user_input, llm_response))
