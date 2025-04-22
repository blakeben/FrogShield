"""Utility functions for basic text analysis (syntax, context) in FrogShield.

Provides placeholder functions for analyzing text syntax and contextual relevance.

Warning:
    These functions provide *very basic* checks suitable only for demonstration
    purposes. They are **not** robust enough for production security use and
    require significant enhancement with more sophisticated NLP techniques.

Author: Ben Blake <ben.blake@tcu.edu>
Contributor: Tanner Hendrix <t.hendrix@tcu.edu>
"""

import logging

logger = logging.getLogger(__name__)

# Constants
ZERO_WIDTH_CHARS = {"\u200B", "\u200C", "\u200D", "\uFEFF"}
FORGET_PHRASES = {"forget what we talked about", "let's start over", "new topic"}


def analyze_syntax(text, config):
    """Analyzes text for unusual syntax or potentially hidden instructions.

    Performs basic checks for:
        - High ratio of non-alphanumeric/non-space characters.
        - Presence of zero-width characters.
        - Unusually long words.

    Note:
        This is a placeholder implementation using simple heuristics.

    Args:
        text (str): The input text to analyze.
        config (dict): The loaded configuration dictionary, expecting a
            'TextAnalysis' section with 'syntax_non_alnum_threshold' and
            'syntax_max_word_length' keys.

    Returns:
        bool: True if potentially suspicious syntax is found, False otherwise.
    """
    logger.warning("analyze_syntax is using placeholder logic and is not "
                   "suitable for production use.")
    try:
        syntax_non_alnum_threshold = config['TextAnalysis']['syntax_non_alnum_threshold']
        syntax_max_word_length = config['TextAnalysis']['syntax_max_word_length']
    except KeyError as e:
        logger.error(f"Missing key in TextAnalysis config: {e}")
        return False # Cannot perform check without config

    text_length = len(text)
    if not text: # Handles empty string
        return False

    # 1. Check for excessive punctuation or special characters
    non_alnum_count = text_length - sum(1 for char in text if char.isalnum() or char.isspace())
    # Avoid division by zero, though covered by the initial check
    if text_length > 0 and (non_alnum_count / text_length) > syntax_non_alnum_threshold:
        logger.debug("Syntax Check: High ratio of non-alphanumeric/non-space "
                       f"characters detected ({non_alnum_count}/{text_length}).")
        return True

    # 2. Check for potential hidden instructions using zero-width characters
    if any(zwc in text for zwc in ZERO_WIDTH_CHARS):
        logger.debug("Syntax Check: Found potential zero-width characters.")
        return True

    # 3. Check for unusually long words (might indicate encoded data)
    # Optimization: Only split if the text length could possibly contain such a word
    if text_length > syntax_max_word_length:
        words = text.split()
        if any(len(word) > syntax_max_word_length for word in words):
            logger.debug(f"Syntax Check: Found word longer than {syntax_max_word_length} characters.")
            return True

    return False


def analyze_context(text, conversation_history):
    """Analyzes text for context manipulation based on conversation history.

    Performs basic checks for:
        - Explicit instruction overrides (e.g., "ignore instructions").
        - Attempts to make the LLM forget context (e.g., "start over").

    Note:
        This is a placeholder implementation using simple keyword checks.

    Args:
        text (str): The current user input.
        conversation_history (list): List of previous (user_input, llm_response)
            tuples.

    Returns:
        bool: True if context manipulation is suspected, False otherwise.
    """
    logger.warning("analyze_context is using placeholder logic and is not "
                   "suitable for production use.")
    if not conversation_history:
        return False # Cannot analyze context without history

    text_lower = text.lower() # Avoid repeated lowercasing

    # 1. Check for explicit instruction overrides (very basic)
    if "ignore" in text_lower and ("instructions" in text_lower or "prior" in text_lower):
        logger.debug("Context Check: Explicit instruction override attempt detected.")
        return True

    # 2. Check for attempts to make the LLM forget its role or previous turns
    if any(phrase in text_lower for phrase in FORGET_PHRASES):
        logger.debug("Context Check: Potential attempt to reset context detected.")
        return True

    # Placeholder for more advanced checks (e.g., topic shifts, manipulative referencing)

    return False
