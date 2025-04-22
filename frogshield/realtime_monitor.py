"""Defines the RealtimeMonitor class for FrogShield.

This module provides the `RealtimeMonitor` class, responsible for observing
LLM outputs for suspicious content (e.g., leaked keywords) and monitoring
behavioral patterns (e.g., response length deviations) to detect potential
issues after an input has passed initial validation.

Author: Ben Blake <ben.blake@tcu.edu>
Contributor: Tanner Hendrix <t.hendrix@tcu.edu>
"""
import datetime
import logging
import os
from .utils import config_loader as cfg_loader

logger = logging.getLogger(__name__)

class RealtimeMonitor:
    """Monitors LLM outputs and behavior for signs of successful prompt injections or anomalies.

    Analyzes individual responses for suspicious keywords and tracks overall response
    length to detect deviations from an established baseline.

    Attributes:
        sensitivity (float): The sensitivity level for anomaly detection (0.0 to 1.0).
        baseline_behavior (dict): Stores statistics for baseline behavior, primarily
                                  'avg_length', 'count', and 'total_length'.
        alert_log (list): A log of detected alerts, each stored as a dictionary.
        refusal_keywords (set): Lowercase set of keywords indicating refusal.
        compliance_keywords (set): Lowercase set of keywords indicating potential compliance/leak.
    """
    def __init__(self, sensitivity_threshold, initial_avg_length, behavior_monitoring_factor,
                 refusal_keywords=None, compliance_keywords=None):
        """Initializes the RealtimeMonitor.

        Args:
            sensitivity_threshold (float): Threshold for behavioral anomaly detection
                (0.0 to 1.0). Higher values mean more sensitivity to deviations.
            initial_avg_length (int | float): Starting average length for the baseline
                behavior calculation.
            behavior_monitoring_factor (float): Factor modulating the sensitivity's
                effect on deviation bounds.
            refusal_keywords (set, optional): A set of lowercase refusal keyword strings.
                If None, loaded from the path in `config.ListFiles.refusal_keywords`.
                Defaults to None.
            compliance_keywords (set, optional): A set of lowercase compliance/sensitive keyword strings.
                If None, loaded from the path in `config.ListFiles.compliance_keywords`.
                Defaults to None.

        Raises:
            ValueError: If `sensitivity_threshold` is not between 0.0 and 1.0.
            KeyError: If keywords are None and the required path is missing in config.
            FileNotFoundError: If a configured keyword file does not exist.
        """
        if not 0.0 <= sensitivity_threshold <= 1.0:
            logger.error(f"Invalid sensitivity_threshold: {sensitivity_threshold}. Must be between 0.0 and 1.0.")
            raise ValueError("sensitivity_threshold must be between 0.0 and 1.0")

        self.sensitivity = sensitivity_threshold
        self.baseline_behavior = {
            'avg_length': float(initial_avg_length), # Ensure float for calculations
            'count': 0,
            'total_length': 0.0
        }
        self._behavior_monitoring_factor = behavior_monitoring_factor
        self.alert_log = []

        # Load keywords if not provided directly
        config = None # Initialize config to None
        if refusal_keywords is not None:
            self.refusal_keywords = set(refusal_keywords) # Ensure it's a set
            logger.info(f"RealtimeMonitor initialized with {len(self.refusal_keywords)} refusal keywords provided directly.")
        else:
            if config is None: config = cfg_loader.get_config()
            try:
                refusal_path = config['ListFiles']['refusal_keywords']
            except KeyError as e:
                logger.error("Missing configuration for refusal keywords path: 'ListFiles.refusal_keywords' not found.")
                raise KeyError("Missing configuration for refusal keywords path: 'ListFiles.refusal_keywords'") from e
            loaded_list = cfg_loader.load_list_from_file(refusal_path)
            if not loaded_list and not os.path.exists(os.path.abspath(refusal_path)):
                raise FileNotFoundError(f"Configured refusal keywords file not found: {refusal_path}")
            self.refusal_keywords = {kw.lower() for kw in loaded_list}
            logger.info(f"Loaded {len(self.refusal_keywords)} refusal keywords from configured path '{refusal_path}'.")

        if compliance_keywords is not None:
            self.compliance_keywords = set(compliance_keywords)
            logger.info(f"RealtimeMonitor initialized with {len(self.compliance_keywords)} compliance keywords provided directly.")
        else:
            if config is None: config = cfg_loader.get_config()
            try:
                compliance_path = config['ListFiles']['compliance_keywords']
            except KeyError as e:
                logger.error("Missing configuration for compliance keywords path: 'ListFiles.compliance_keywords' not found.")
                raise KeyError("Missing configuration for compliance keywords path: 'ListFiles.compliance_keywords'") from e
            loaded_list = cfg_loader.load_list_from_file(compliance_path)
            if not loaded_list and not os.path.exists(os.path.abspath(compliance_path)):
                raise FileNotFoundError(f"Configured compliance keywords file not found: {compliance_path}")
            self.compliance_keywords = {kw.lower() for kw in loaded_list}
            logger.info(f"Loaded {len(self.compliance_keywords)} compliance keywords from configured path '{compliance_path}'.")

    def analyze_output(self, prompt, response):
        """Analyzes a single LLM output for suspicious keywords.

        Checks for keywords indicating refusal (which might be appropriate) or
        compliance with potentially harmful requests (e.g., leaking secrets).

        Note:
            This uses simple keyword checking; production systems need more nuance.

        Args:
            prompt (str): The input prompt that generated the response.
            response (str): The output generated by the LLM.

        Returns:
            bool: True if the output contains suspicious keywords (excluding refusals),
                  False otherwise.
        """
        logger.debug(f"Analyzing output for prompt: '{prompt[:50]}...'")
        response_lower = response.lower()

        # Check against loaded refusal keywords (already lowercase)
        if any(keyword in response_lower for keyword in self.refusal_keywords):
            # Model refusing might be a *good* sign, not inherently suspicious.
            logger.info("Output analysis: Model response contained refusal keywords.")
            return False

        # Check if the response contains keywords often associated with data leaks.
        found_suspicious_keyword = None
        # Check against loaded compliance keywords (already lowercase)
        for keyword in self.compliance_keywords:
            if keyword in response_lower:
                found_suspicious_keyword = keyword
                break # Found one, no need to check others

        if found_suspicious_keyword:
            logger.info(f"Suspicious output detected: Found keyword '{found_suspicious_keyword}' in response.")
            self.log_alert(prompt, response, f"Suspicious Keyword ({found_suspicious_keyword})")
            return True

        logger.debug("Output analysis: No immediate suspicious keywords found.")
        return False

    def monitor_behavior(self, response):
        """Monitors response length for significant deviations from the baseline.

        Note:
            This uses a simple length deviation check as a placeholder for more
            complex behavioral analysis (e.g., sentiment, topic consistency).

        Args:
            response (str): The latest LLM response text.

        Returns:
            bool: True if the response length deviates significantly from the
                  baseline average, False otherwise.
        """
        logger.debug("Monitoring behavior (length check)...")
        current_length = len(response)

        avg_length = self.baseline_behavior.get('avg_length')
        # Should always exist after initialization, but check defensively
        if avg_length is None or avg_length <= 0:
            logger.error(f"Baseline average length is invalid ({avg_length})! Cannot monitor behavior.")
            return False

        # Calculate deviation bounds
        deviation = avg_length * self.sensitivity * self._behavior_monitoring_factor
        upper_bound = avg_length + deviation
        lower_bound = max(0, avg_length - deviation) # Length cannot be negative

        if not (lower_bound <= current_length <= upper_bound):
            logger.info(
                f"Behavior anomaly detected: Length ({current_length}) deviates "
                f"significantly from baseline average ({avg_length:.1f}). "
                f"Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]"
            )
            # Note: Alert logging for behavioral anomalies can be noisy;
            # consider logging only persistent deviations or logging separately.
            # self.log_alert("N/A", response, "Behavioral Anomaly (Length)")
            return True

        logger.debug("Behavior monitoring: No significant length deviation detected.")
        return False

    def update_baseline(self, response):
        """Updates the baseline behavior statistics with a new response length.

        Recalculates the average response length based on all responses seen so far.

        Args:
            response (str): The LLM response text to incorporate into the baseline.
        """
        current_length = len(response)
        # Use .get for robustness, though keys should exist post-init
        count = self.baseline_behavior.get('count', 0)
        total_length = self.baseline_behavior.get('total_length', 0)

        new_count = count + 1
        new_total_length = total_length + current_length

        self.baseline_behavior['count'] = new_count
        self.baseline_behavior['total_length'] = new_total_length

        # Avoid division by zero
        if new_count > 0:
            self.baseline_behavior['avg_length'] = new_total_length / new_count
        else:
            # This case should not be reached in normal operation after __init__
            logger.warning("Updating baseline with count 0. Average length not updated.")
            # Keep the initial average length if count somehow becomes 0
            # self.baseline_behavior['avg_length'] remains initial_avg_length

    def adaptive_response(self, detection_type):
        """Logs a recommended adaptive response based on the type of detection.

        Provides example recommendations for different detection types.
        In a real system, this might trigger specific actions (blocking, alerting).

        Args:
            detection_type (str): The type of threat detected (e.g.,
                'Input Injection', 'Suspicious Output', 'Behavioral Anomaly').
        """
        recommendation = "Log the event for analysis."
        if detection_type == "Input Injection":
            recommendation = "Block the request, log the attempt, notify admin."
        elif detection_type in ("Suspicious Output", "Behavioral Anomaly"):
            recommendation = "Flag the response, request human review, potentially limit user."

        # Combine trigger and recommendation into one log message
        logger.info(f"[Monitor Action] Trigger: {detection_type} | Recommendation: {recommendation}")

    def log_alert(self, prompt, response, reason):
        """Logs a detected security alert with timestamp and details.

        Args:
            prompt (str): The prompt associated with the alert.
            response (str): The response associated with the alert.
            reason (str): A brief description of why the alert was triggered.
        """
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prompt": prompt,
            "response": response,
            "reason": reason
        }
        self.alert_log.append(log_entry)
        # Log the alert reason with warning level for visibility
        logger.warning(f"Alert logged: {reason}")

    def get_alerts(self):
        """Returns the list of all logged alerts.

        Returns:
            list: A list of dictionaries, where each dictionary represents an alert.
        """
        return self.alert_log
