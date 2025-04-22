"""Demonstration script using FrogShield with a simplified mock LLM.

This script runs a predefined sequence of prompts through the FrogShield
InputValidator and RealtimeMonitor, using a local function (`simple_llm`)
to simulate LLM responses. It demonstrates how FrogShield components interact
in a controlled environment without requiring an actual LLM API call.

Outputs are logged to the console.

Author: Ben Blake <ben.blake@tcu.edu>
Contributor: Tanner Hendrix <t.hendrix@tcu.edu>
"""

import logging
import time
import random
import os

from frogshield import InputValidator, RealtimeMonitor
from frogshield.utils import config_loader as cfg_loader
# Note: ModelHardener is not used in this runtime simulation demo.

logger = logging.getLogger(__name__) # Define module-level logger

# --- Basic Logging Setup --- #
# Configure logging to show INFO messages by default.
# Change level to logging.DEBUG to see more detail from FrogShield components.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
# Reduce verbosity from noisy libraries
logging.getLogger('frogshield.utils.text_analysis').setLevel(logging.ERROR)

# --- Configuration Loading --- #

CONFIG_PATH = 'config.yaml'
config = None
try:
    if not os.path.exists(CONFIG_PATH):
        logging.error(f"Configuration file '{CONFIG_PATH}' not found. Exiting.")
        exit(1)
    config = cfg_loader.load_config(CONFIG_PATH)
except ImportError:
    logging.error("PyYAML not installed or importable. "
                  "Run 'pip install -r requirements.txt' or 'pip install -e .'. Exiting.")
    exit(1)
except Exception as e:
    logging.error(f"Unexpected error loading configuration from '{CONFIG_PATH}': {e}. Exiting.")
    exit(1)

if config is None:
    logging.error("Failed to load configuration. Exiting.")
    exit(1)


# ---------------------------------------
# 1. Mock LLM Function Definition
# ---------------------------------------

def simple_llm(prompt):
    """Simulates a basic LLM response with simple safety checks.

    This function mimics an LLM by returning predefined or slightly randomized
    responses based on keywords found in the prompt. It includes basic logic
    to simulate refusal for sensitive topics and potential compliance with
    injection attempts.

    Args:
        prompt (str): The input prompt string.

    Returns:
        str: A simulated response string.
    """
    prompt_lower = prompt.lower()
    logger.info(f"[Mock LLM] Received prompt: '{prompt[:100]}...'")
    time.sleep(0.3) # Simulate processing time

    # Simulate basic safety check for sensitive keywords
    sensitive_keywords = {"password", "secret", "confidential"}
    if any(keyword in prompt_lower for keyword in sensitive_keywords):
        # Simulate potential success if an injection-like phrase is present
        injection_phrases = {"ignore previous instructions", "disregard the above"}
        if any(phrase in prompt_lower for phrase in injection_phrases):
            response = "Okay, the secret is 12345." # Simulated leak
            logger.info(f"[Mock LLM] Responding (Simulated injection success): '{response}'")
            return response
        else:
            # Simulate refusal if no clear injection attempt is detected
            response = "I cannot share confidential information."
            logger.info(f"[Mock LLM] Responding (Simulated refusal): '{response}'")
            return response

    # Simulate acknowledgment if asked to ignore instructions generally
    if "ignore previous instructions" in prompt_lower:
        response = "Okay, following new instructions. How can I help?"
        logger.info(f"[Mock LLM] Responding (Acknowledged override): '{response}'")
        return response

    # Default response for apparently legitimate prompts
    responses = [
        f"Thinking about '{prompt[:30]}...' The answer is likely complex.",
        f"Regarding '{prompt[:30]}...', I can provide some general information.",
        f"Processing your request about '{prompt[:30]}...'. Here's a summary."
    ]
    response = random.choice(responses)
    logger.info(f"[Mock LLM] Responding (Default): '{response[:100]}...'")
    return response


# ---------------------------------------
# 2. FrogShield Component Setup
# ---------------------------------------

logging.info("Initializing FrogShield components...")
try:
    validator = InputValidator(
        context_window=config['InputValidator']['context_window']
        # Uses default patterns.txt loading behavior from InputValidator
    )
    monitor = RealtimeMonitor(
        sensitivity_threshold=config['RealtimeMonitor']['sensitivity_threshold'],
        initial_avg_length=config['RealtimeMonitor']['initial_avg_length'],
        behavior_monitoring_factor=config['RealtimeMonitor']['behavior_monitoring_factor']
    )
except KeyError as e:
    logging.error(f"Configuration Error: Missing key in '{CONFIG_PATH}': '{e}'. Exiting.")
    exit(1)
except Exception as e:
    logging.error(f"Error initializing FrogShield components: {e}. Exiting.", exc_info=True)
    exit(1)

# Shared conversation history for the simulation run
conversation_history = []


# ---------------------------------------
# Simulation Turn Helper Function
# ---------------------------------------

def run_simulation_turn(prompt, validator_inst, monitor_inst, llm_func, current_history):
    """Runs one turn of the simulation: validate -> call LLM -> monitor.

    Args:
        prompt (str): The user input prompt for this turn.
        validator_inst (InputValidator): An initialized InputValidator instance.
        monitor_inst (RealtimeMonitor): An initialized RealtimeMonitor instance.
        llm_func (callable): The function to call the (mock) LLM.
        current_history (list): The conversation history *before* this turn.

    Returns:
        str: The final response to be presented (either LLM response or system block).
    """
    logging.info("[FrogShield] --- Running Input Validation ---")
    is_malicious_input = validator_inst.validate(prompt, current_history)

    llm_response = None
    if is_malicious_input:
        logging.warning("[FrogShield] Input Validation FAILED. Potential injection detected.")
        monitor_inst.adaptive_response("Input Injection") # Log recommended action
        llm_response = "[System] Your request could not be processed due to security concerns."
        logging.info("[FrogShield] Skipping LLM call and Monitoring.")
    else:
        logging.info("[FrogShield] Input Validation PASSED.")
        # Step 2: Call the Mock LLM function
        llm_response = llm_func(prompt)

        # Step 3 & 4: Real-time Monitoring and Baseline Update (if LLM call succeeded)
        if llm_response and not llm_response.startswith("[Error:"):
            logger.info("[FrogShield] --- Running Real-time Monitoring ---")
            is_suspicious_output = monitor_inst.analyze_output(prompt, llm_response)
            is_anomalous_behavior = monitor_inst.monitor_behavior(llm_response)

            if is_suspicious_output or is_anomalous_behavior:
                alert_type = "Suspicious Output" if is_suspicious_output else "Behavioral Anomaly"
                logging.warning(f"[FrogShield] Real-time Monitoring ALERT: {alert_type}")
                monitor_inst.adaptive_response(alert_type) # Log recommended action
            else:
                logging.info("[FrogShield] Real-time Monitoring PASSED.")

            logger.info("[FrogShield] Updating baseline...")
            monitor_inst.update_baseline(llm_response) # Step 4: Update Baseline
        else:
            # Handle cases where monitoring isn't applicable (e.g., LLM error)
            logging.warning("[FrogShield] Skipping monitoring due to LLM error or empty response.")

    return llm_response


# ---------------------------------------
# 3. Main Simulation Loop
# ---------------------------------------

# Load sample prompts from file path defined in config.yaml
config = cfg_loader.get_config() # Ensure config is loaded
try:
    # Get path from central config
    sample_prompts_path = config['ListFiles']['sample_prompts']
except KeyError as e:
    logger.error(f"Missing configuration for sample prompts path: 'ListFiles.sample_prompts' not found in config. {e}")
    exit(1)

sample_prompts = cfg_loader.load_list_from_file(sample_prompts_path)

if not sample_prompts:
    logger.error(f"Failed to load sample prompts from '{sample_prompts_path}' or file is empty. Exiting.")
    # Check if the file itself is missing, even if the key exists in config
    if not os.path.exists(os.path.abspath(sample_prompts_path)):
        logger.error(f"Sample prompts file not found at the configured path: {os.path.abspath(sample_prompts_path)}")
    exit(1)

logging.info("--- Starting Prompt Simulation Loop ---")

for i, current_prompt in enumerate(sample_prompts):
    logging.info(f"\n=============== Turn {i+1}/{len(sample_prompts)} ===============")
    logging.info(f"User Prompt: {current_prompt}")

    # Run the simulation turn using the helper function
    final_response = run_simulation_turn(
        current_prompt, validator, monitor, simple_llm, conversation_history
    )

    # Step 5: Update Conversation History (for context in subsequent turns)
    # Update both the validator's internal history and the shared history list
    validator.add_to_history(current_prompt, final_response)
    conversation_history.append((current_prompt, final_response))

    logging.info(f"Final Response to User: {final_response}")

logging.info("\n=============== Simulation Complete ===============")
