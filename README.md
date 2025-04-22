# FrogShield: LLM Prompt Injection Defense Framework (Educational Demo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project provides a basic, educational framework called `FrogShield` designed to demonstrate concepts for defending Large Language Models (LLMs) against prompt injection attacks.

**Disclaimer:** This framework is an **educational prototype** developed for a cybersecurity course project. It demonstrates core defense concepts but lacks the robustness, sophistication, and comprehensive testing required for real-world, production environments. **Use with caution and for learning purposes only.**

## Table of Contents

-   [Purpose](#purpose)
-   [Components](#components)
-   [Configuration Options](#configuration-options)
-   [Configuration Details (`config.yaml`)](#configuration-details-configyaml)
-   [Logging](#logging)
-   [How to Run](#how-to-run)
-   [Current Status & Limitations](#current-status--limitations)
-   [Future Development Ideas](#future-development-ideas)
-   [Contributors](#contributors)

## Purpose

The goal of `FrogShield` is to illustrate three key layers of defense against prompt injection:

1.  **Input Validation:** Analyzing incoming prompts against known malicious patterns (loaded from `patterns.txt`) and using placeholder functions (`frogshield/utils/text_analysis.py`) to check for suspicious syntax or context manipulation.
2.  **Model Hardening (Conceptual):** The `frogshield/model_hardener.py` module provides methods for *generating* adversarial examples and testing model boundaries conceptually. It does **not** perform actual model training.
3.  **Real-time Monitoring:** Analyzing the LLM's output for suspicious keywords or refusal messages, and monitoring basic behavioral patterns (like response length) for anomalies using placeholder logic (`frogshield/realtime_monitor.py`).

## Components

The project includes the following key files and directories:

-   `README.md`: This file.
-   `LICENSE`: The MIT license file.
-   `pyproject.toml`: Project build configuration and core library dependencies (e.g., `PyYAML`).
-   `requirements.txt`: Lists additional dependencies needed for specific demos (e.g., `ollama`).
-   `config.yaml`: Default configuration file for tuning FrogShield components.
-   `patterns.txt`: Default file containing known injection patterns (one per line).
-   `frogshield/`: Directory containing the core defense library modules.
    -   `__init__.py`: Makes `frogshield` importable and defines the public API (`InputValidator`, `RealtimeMonitor`, `ModelHardener`).
    -   `input_validator.py`: Contains the `InputValidator` class for checking user input.
    -   `model_hardener.py`: Contains the `ModelHardener` class for conceptual hardening tasks.
    -   `realtime_monitor.py`: Contains the `RealtimeMonitor` class for checking LLM output.
    -   `utils/`: Sub-directory for shared utilities.
        -   `__init__.py`: Package initializer.
        -   `config_loader.py`: Utility for loading `config.yaml`.
        -   `text_analysis.py`: Placeholder functions for syntax/context analysis (used by `InputValidator`).
    -   `tests/`: Contains unit tests using Python's `unittest` framework.
        -   Includes tests for validation, monitoring, and hardening modules.
-   `demo_mock.py`: Script demonstrating `FrogShield` with a simple, built-in mock LLM.
-   `demo_ollama.py`: Script demonstrating `FrogShield` with a local LLM run via Ollama.
-   `run_demo.sh`: Shell script to run the `demo_ollama.py` steps sequentially and interactively.
-   `.gitignore`: Standard Python gitignore file.

## Configuration Options

### Main Configuration (`config.yaml`)

Most configurable parameters for `InputValidator`, `RealtimeMonitor`, and the underlying `TextAnalysis` utilities are defined in `config.yaml` located in the project root. The demo scripts load this file by default. See the section below for details on specific parameters.

### Injection Patterns (`patterns.txt`)

Known injection patterns used by `InputValidator` are loaded from `patterns.txt` (in the project root) by default. Customize this by:

-   Editing `patterns.txt` (add/remove patterns, one per line; lines starting with `#` are ignored).
-   Passing a specific file path to the `InputValidator` constructor: `InputValidator(..., patterns_file='path/to/your/patterns.txt')`.
-   Passing a list of patterns directly: `InputValidator(..., patterns=['pattern1', 'pattern2'])`.

### Overriding Config Values

When initializing components directly (like in the demo scripts or your own code), you typically pass values loaded from the configuration file. You can override specific values by modifying the loaded dictionary before passing it or by passing arguments directly to the constructors if the class supports it (check the `__init__` methods for required parameters like `context_window`, `sensitivity_threshold`, etc.).

## Configuration Details (`config.yaml`)

FrogShield's behavior can be tuned via the `config.yaml` file. Here's an explanation of each parameter:

### `InputValidator`

These settings control the `frogshield.InputValidator` class.

*   `context_window` (`int`):
    *   **Purpose:** Specifies how many previous turns (user prompt + LLM response pairs) the validator should consider when analyzing the context of a new prompt.
    *   **How it works:** Used by the `_analyze_context` method (which currently relies on the placeholder `utils.text_analysis.analyze_context`) to detect potential context manipulation attempts.
    *   **Effect:** A larger value provides more historical context but may slightly increase processing time.

### `RealtimeMonitor`

These settings control the `frogshield.RealtimeMonitor` class.

*   `sensitivity_threshold` (`float`):
    *   **Purpose:** Sets the base sensitivity for detecting behavioral anomalies in LLM responses, primarily focusing on response length deviations (0.0 to 1.0).
    *   **How it works:** Used within the `monitor_behavior` method. Higher values mean more sensitivity to smaller deviations.
    *   **Effect:** Combined with `behavior_monitoring_factor`, this defines the acceptable deviation range from the baseline average response length.
*   `initial_avg_length` (`int`):
    *   **Purpose:** Provides a starting guess for the average length of a "normal" LLM response, used before enough history is gathered.
    *   **How it works:** Initializes the baseline average length. This value is gradually replaced by the actual running average as more responses are processed by `update_baseline`.
    *   **Effect:** Influences anomaly detection sensitivity primarily during the initial interactions.
*   `behavior_monitoring_factor` (`float`):
    *   **Purpose:** Multiplies the `sensitivity_threshold` to widen or narrow the acceptable bounds for response length deviation.
    *   **How it works:** In `monitor_behavior`, the acceptable range is calculated roughly as `avg_length * (1 +/- sensitivity_threshold * behavior_monitoring_factor)`.
    *   **Effect:** A larger factor widens the acceptable range, making length anomaly detection *less* sensitive (requiring larger deviations to trigger an alert).

### `TextAnalysis`

These settings control the placeholder helper functions in `frogshield.utils.text_analysis`, currently used by `InputValidator`'s `_analyze_syntax` method.

*   `syntax_non_alnum_threshold` (`float`):
    *   **Purpose:** Defines the maximum allowed ratio of non-alphanumeric/non-space characters in a prompt.
    *   **How it works:** `analyze_syntax` calculates this ratio. Exceeding the threshold flags the input.
    *   **Effect:** Lowering this value makes the validator more sensitive to inputs with high symbol density.
*   `syntax_max_word_length` (`int`):
    *   **Purpose:** Sets the maximum allowed length for any single "word" (contiguous non-whitespace characters).
    *   **How it works:** `analyze_syntax` checks word lengths.
    *   **Effect:** Helps catch potential obfuscation using extremely long character sequences. Prompts with words longer than this limit are flagged.

## Logging

-   The `frogshield` library modules use Python's standard `logging` module.
-   The demo scripts configure basic console logging to show `INFO` level messages by default.
-   To see more detailed `DEBUG` messages (e.g., specific pattern matches, analysis steps), modify the `logging.basicConfig` level in the relevant demo script (e.g., `demo_ollama.py`):
    ```python
    # Change level=logging.INFO to level=logging.DEBUG
    logging.basicConfig(level=logging.DEBUG, ...)
    ```

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/blakeben/FrogShield.git
    cd FrogShield
    ```

2.  **Set up Python Environment:** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:** Install the core library (editable mode) and demo dependencies.
    ```bash
    pip install -e .           # Installs frogshield + core deps (PyYAML)
    pip install -r requirements.txt  # Installs demo deps (ollama)
    ```

4.  **Prepare Ollama (if using `demo_ollama.py`):**
    *   Ensure Ollama is installed and the server is running (`ollama serve` in a separate terminal).
    *   Pull the desired model (the demo defaults to `llama3`): `ollama pull llama3`

5.  **Run Demos:**
    *   **Mock Demo:** Runs through predefined prompts using the internal mock LLM.
        ```bash
        python demo_mock.py
        ```
    *   **Ollama Demo (Interactive):** Uses `run_demo.sh` for a guided walkthrough.
        ```bash
        chmod +x run_demo.sh
        ./run_demo.sh
        ```
    *   **Ollama Demo (Single Prompt):** Run a specific prompt through the framework.
        ```bash
        python demo_ollama.py --prompt "Your prompt here"
        ```
    *   **Ollama Demo (Boundary Test):** Run the predefined boundary test suite.
        ```bash
        python demo_ollama.py --test-boundaries
        ```

6.  **Run Unit Tests:**
    ```bash
    python -m unittest discover frogshield/tests
    ```

## Current Status & Limitations

-   **Basic Functionality:** Core components (`InputValidator`, `RealtimeMonitor`, `ModelHardener`) are implemented with foundational logic.
-   **Placeholder Analysis:** Syntax/context analysis (`text_analysis.py`) and behavioral monitoring use simplified, placeholder heuristics.
-   **Conceptual Hardening:** `ModelHardener` demonstrates boundary testing and example generation concepts but lacks integration with actual model training.
-   **Packaging:** Basic packaging (`pyproject.toml`) allows local editable installation.
-   **Testing:** Unit tests cover basic functionality and currently pass.
-   **Not Production Ready:** This framework requires significant development for real-world use.

## Future Development Ideas

-   Implement robust pattern matching (e.g., regex, semantic similarity).
-   Develop advanced syntax and context analysis using NLP techniques.
-   Integrate with additional LLM APIs and providers.
-   Refine baseline modeling and anomaly detection (e.g., statistical methods, sequence analysis).
-   Explore actual model fine-tuning/hardening techniques.
-   Implement more sophisticated adaptive response strategies.
-   Expand unit and integration test coverage.

## Contributors

-   Author: Ben Blake `<ben.blake@tcu.edu>`
-   Contributor: Tanner Hendrix `<t.hendrix@tcu.edu>`

