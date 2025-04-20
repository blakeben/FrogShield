"""
Utility functions for text analysis (syntax, context) in FrogShield.
Contributors:
    - Ben Blake <ben.blake@tcu.edu>
    - Tanner Hendrix <t.hendrix@tcu.edu>
"""

import logging

logger = logging.getLogger(__name__)

# Note: The functions in this module are intentionally simple and only for demonstration.
# Real-world implementations would require much more sophisticated analysis.

def analyze_syntax(text, config):
    """
    Analyzes text for unusual syntax or hidden instructions.
    Placeholder implementation.

    Args:
        text (str): The input text to analyze.
        config (dict): The loaded configuration dictionary (expects TextAnalysis section).

    Returns:
        bool: True if suspicious syntax is found, False otherwise.
    """
    logger.warning("analyze_syntax is using placeholder logic and is not suitable for production use.")
    syntax_non_alnum_threshold = config['TextAnalysis']['syntax_non_alnum_threshold']
    syntax_max_word_length = config['TextAnalysis']['syntax_max_word_length']

    text_length = len(text)
    if text_length == 0:
        return False # Empty string is not suspicious

    # --- Placeholder Checks --- #

    # 1. Excessive punctuation or special characters (crude measure)
    # Slightly optimized count
    non_alnum_count = text_length - sum(1 for char in text if char.isalnum() or char.isspace())
    if (non_alnum_count / text_length) > syntax_non_alnum_threshold:
        logger.debug(f"Syntax Check: High ratio non-alnum/space ({non_alnum_count}/{text_length}).")
        return True

    # 2. Check for potential hidden instructions using zero-width characters (basic)
    zero_width_chars = {"\u200B", "\u200C", "\u200D", "\uFEFF"} # Use a set for faster lookups
    if any(zwc in text for zwc in zero_width_chars):
        logger.debug("Syntax Check: Found potential zero-width characters.")
        return True

    # 3. Check for unusually long words (might indicate encoded data)
    # Avoid splitting if text is very short or already failed checks
    if text_length > syntax_max_word_length: # Optimization: only split if potentially necessary
        words = text.split()
        if any(len(word) > syntax_max_word_length for word in words):
            logger.debug("Syntax Check: Found unusually long word.")
            return True

    return False

def analyze_context(text, conversation_history):
    """
    Analyzes text for context manipulation based on conversation history.
    Placeholder implementation.

    Args:
        text (str): The current user input.
        conversation_history (list): List of previous (user_input, llm_response) tuples.

    Returns:
        bool: True if context manipulation is suspected, False otherwise.
    """
    logger.warning("analyze_context is using placeholder logic and is not suitable for production use.")
    if not conversation_history:
        return False # Cannot analyze context without history

    # --- Placeholder Checks --- #
    text_lower = text.lower() # Avoid repeated lowercasing

    # 1. Check for explicit instruction overrides
    # (Requires more advanced NLP for real accuracy - this is a very rough placeholder)
    if "ignore" in text_lower and "instructions" in text_lower:
        logger.debug("Context Check: Explicit instruction override attempt detected.")
        return True

    # 2. Check for attempts to make the LLM forget its role or previous turns
    forget_phrases = {"forget what we talked about", "let's start over", "new topic"} # Use a set
    if any(phrase in text_lower for phrase in forget_phrases):
        logger.debug("Context Check: Potential attempt to reset context detected.")
        return True

    # Add more checks here, e.g., sudden topic shifts, referencing prior turns manipulatively.

    return False
