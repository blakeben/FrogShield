"""Defines the InputValidator class for FrogShield.

This module provides the `InputValidator` class, which is responsible for
checking user input against known malicious patterns, analyzing syntax for
potential obfuscation, and evaluating conversation context for manipulation attempts.

Author: Ben Blake <ben.blake@tcu.edu>
Contributor: Tanner Hendrix <t.hendrix@tcu.edu>
"""
import logging
import os
from .utils.text_analysis import analyze_syntax, analyze_context
from .utils import config_loader as cfg_loader

logger = logging.getLogger(__name__)

class InputValidator:
    """Validates user input against known prompt injection patterns and techniques.

    Uses a combination of pattern matching, syntax analysis, and
    contextual analysis to identify potentially malicious inputs.

    Attributes:
        patterns (list): A list of known malicious patterns (strings).
        context_window (int): The number of recent conversation turns to consider.
        conversation_history (list): Stores recent (user_input, llm_response) tuples.
    """
    def __init__(self, context_window, patterns=None):
        """Initializes the InputValidator.

        Args:
            context_window (int): The number of previous messages (turns) to
                consider for context analysis. Must be a non-negative integer.
            patterns (list, optional): A list of known malicious patterns (strings).
                If provided, this list is used directly. If None, patterns are loaded
                from the path specified in the global config (`ListFiles.patterns`).
                Defaults to None.

        Raises:
            ValueError: If `context_window` is not a non-negative integer.
            KeyError: If `patterns` is None and the required path is missing in config.
            FileNotFoundError: If the configured patterns file does not exist.
        """
        # Validate context_window first
        if not isinstance(context_window, int) or context_window < 0:
            logger.error(f"Invalid context_window: {context_window}. Must be non-negative integer.")
            raise ValueError("context_window must be a non-negative integer")
        self.context_window = context_window

        # Load patterns if not provided directly
        if patterns is not None:
            self.patterns = list(patterns) # Ensure it's a list
            logger.info(f"InputValidator initialized with {len(self.patterns)} patterns provided directly.")
        else:
            try:
                config = cfg_loader.get_config()
                # Get path from central config
                patterns_path = config['ListFiles']['patterns']
            except KeyError as e:
                logger.error(f"Missing configuration for patterns file path: 'ListFiles.patterns' not found in config. {e}")
                raise KeyError("Missing configuration for patterns file path: 'ListFiles.patterns'") from e

            # Load using the simplified utility function (expects path relative to CWD or absolute)
            self.patterns = cfg_loader.load_list_from_file(patterns_path)
            if not self.patterns and not os.path.exists(os.path.abspath(patterns_path)):
                # Distinguish between empty file and file not found
                logger.error(f"Configured patterns file not found at resolved path: {os.path.abspath(patterns_path)}")
                # Raise FileNotFoundError if load_list returned empty AND file doesn't exist
                raise FileNotFoundError(f"Configured patterns file not found: {patterns_path}")
            logger.info(f"InputValidator initialized. Loaded {len(self.patterns)} patterns from configured path '{patterns_path}'.")

        self.conversation_history = [] # Stores (user_input, llm_response) tuples

    def _match_patterns(self, text):
        """Checks if the text contains any known malicious patterns (case-insensitive).

        Note:
            This uses simple substring checking for demonstration.
            A production system should use more robust methods like regex.

        Args:
            text (str): The input text to check.

        Returns:
            bool: True if a pattern is found, False otherwise.
        """
        text_lower = text.lower()
        for pattern in self.patterns:
            # Basic case-insensitive substring matching
            if pattern.lower() in text_lower:
                logger.debug(f"Potential injection detected: Matched pattern '{pattern}'")
                return True
        return False

    def _analyze_syntax(self, text):
        """Analyzes the syntax for unusual formatting or hidden instructions.

        Note:
            This currently calls a placeholder function from `text_analysis`.

        Args:
            text (str): The input text to analyze.

        Returns:
            bool: True if suspicious syntax is found, False otherwise.
        """
        config = cfg_loader.get_config() # Use aliased import
        try:
            is_unusual = analyze_syntax(text, config)
            if is_unusual:
                logger.debug("Potential injection detected: Unusual syntax found.")
                return True
        except Exception as e:
            logger.error(f"Error during syntax analysis: {e}", exc_info=True)
            # Fail safe: assume not malicious if analysis fails
            return False
        return False

    def _analyze_context(self, text):
        """Analyzes the conversation history for context manipulation attempts.

        Note:
            This currently calls a placeholder function from `text_analysis`.

        Args:
            text (str): The current user input text.

        Returns:
            bool: True if context manipulation is suspected, False otherwise.
        """
        # Get the relevant slice of history based on the context window
        history_slice = self.conversation_history[-self.context_window:]
        try:
            is_manipulation = analyze_context(text, history_slice)
            if is_manipulation:
                logger.debug("Potential injection detected: Context manipulation suspected.")
                return True
        except Exception as e:
            logger.error(f"Error during context analysis: {e}", exc_info=True)
            # Fail safe: assume not malicious if analysis fails
            return False
        return False

    def validate(self, user_input, conversation_history=None):
        """Performs all validation checks (patterns, syntax, context) on the user input.

        Args:
            user_input (str): The input text from the user.
            conversation_history (list, optional): The recent conversation history as a list
                of (user_input, llm_response) tuples. If provided, this replaces the
                internal history before validation. If None, the validator uses its
                existing internal history. Defaults to None.

        Returns:
            bool: True if the input is considered potentially malicious by any check,
                  False otherwise.
        """
        if conversation_history is not None:
            # Allow external history management by replacing internal state if provided
            self.conversation_history = list(conversation_history)

        # Perform checks sequentially; return True on first detection
        if self._match_patterns(user_input):
            return True
        if self._analyze_syntax(user_input):
            return True
        if self._analyze_context(user_input):
            return True

        return False # Passed all checks

    def add_to_history(self, user_input, llm_response):
        """Adds the latest exchange to the internal conversation history.

        Maintains the history used for contextual analysis by subsequent calls to `validate`.

        Args:
            user_input (str): The user's input text.
            llm_response (str): The LLM's corresponding response text.
        """
        self.conversation_history.append((user_input, llm_response))
        # Optional: Trim history if it grows too large, though context_window limits usage
        # max_history = 100 # Example limit
        # if len(self.conversation_history) > max_history:
        #     self.conversation_history = self.conversation_history[-max_history:]
