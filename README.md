# FrogShield: LLM Prompt Injection Defense Framework (Educational Demo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`FrogShield` is an educational framework that demonstrates defensive strategies against prompt injection attacks in Large Language Models (LLMs).

**⚠️ Disclaimer:** This framework is an **educational prototype**. It lacks the robustness, sophistication, and comprehensive testing required for production environments. **Use with caution and for learning purposes only.**

## Table of Contents

-   [Purpose](#purpose)
-   [Components](#components)
-   [Configuration](#configuration)
    -   [Main Configuration (`config.yaml`)](#main-configuration-configyaml)
    -   [List Files (`config_data/`)](#list-files-config_data)
-   [Logging](#logging)
-   [How to Run](#how-to-run)
-   [Current Status & Limitations](#current-status--limitations)
-   [Future Development Ideas](#future-development-ideas)
-   [Contributors](#contributors)

## Purpose

The goal of `FrogShield` is to illustrate three key layers of defense against prompt injection:

1.  **Input Validation:** Analyzing incoming prompts against known malicious patterns (loaded from `config_data/patterns.txt`) and using placeholder functions (`frogshield/utils/text_analysis.py`) to check for suspicious syntax or context manipulation.
2.  **Model Hardening (Conceptual):** The `frogshield/model_hardener.py` module provides methods for *generating* adversarial examples and testing model boundaries conceptually. It does **not** perform actual model training.
3.  **Real-time Monitoring:** Analyzing the LLM's output for suspicious keywords (loaded from `config_data/`) or refusal messages, and monitoring basic behavioral patterns (like response length) for anomalies using placeholder logic (`frogshield/realtime_monitor.py`).

## Components

The project includes the following key files and directories:

-   `README.md`: This file.
-   `LICENSE`: The MIT license file.
-   `pyproject.toml`: Project build configuration and core library dependencies (e.g., `PyYAML`).
-   `requirements.txt`: Lists additional dependencies needed for specific demos (e.g., `ollama`).
-   `config.yaml`: **Main configuration file** for tuning FrogShield components and specifying list file paths.
-   `config_data/`: **Directory containing external data files:**
    -   `patterns.txt`: Default file containing known injection patterns.
    -   `refusal_keywords.txt`: Keywords indicating appropriate LLM refusal.
    -   `compliance_keywords.txt`: Keywords indicating potential sensitive data leaks or compliance.
    -   `sample_prompts.txt`: Sample prompts used by `demo_mock.py`.
    -   `boundary_refusal_keywords.txt`: Refusal keywords used by `demo_ollama.py` boundary tests.
-   `frogshield/`: Directory containing the core defense library modules.
    -   `__init__.py`: Makes `frogshield` importable and defines the public API (`InputValidator`, `RealtimeMonitor`, `ModelHardener`).
    -   `input_validator.py`: Contains the `InputValidator` class for checking user input.
    -   `model_hardener.py`: Contains the `ModelHardener` class for conceptual hardening tasks.
    -   `realtime_monitor.py`: Contains the `RealtimeMonitor` class for checking LLM output.
    -   `utils/`: Sub-directory for shared utilities.
        -   `__init__.py`: Package initializer.
        -   `config_loader.py`: Utility for loading `config.yaml` and list files.
        -   `text_analysis.py`: Placeholder functions for syntax/context analysis (used by `InputValidator`).
    -   `tests/`: Contains unit tests using Python's `unittest` framework.
        -   Includes tests for validation, monitoring, and hardening modules.
-   `demo_mock.py`: Script demonstrating `FrogShield` with a simple, built-in mock LLM.
-   `demo_ollama.py`: Script demonstrating `FrogShield` with a local LLM run via Ollama.
-   `run_demo.sh`: Shell script to run the `demo_ollama.py` steps sequentially and interactively.
-   `.gitignore`: Standard Python gitignore file.

## Configuration

FrogShield's behavior and external data sources are configured primarily through `config.yaml`.

### Main Configuration (`config.yaml`)

Most configurable parameters for `InputValidator`, `RealtimeMonitor`, and the underlying `TextAnalysis` utilities are defined in `config.yaml` located in the project root. The components load these settings automatically if not overridden during instantiation.

#### Parameter Details:

##### `InputValidator`

*   `context_window` (`int`): Number of past conversation turns to consider for context analysis.

##### `RealtimeMonitor`

*   `sensitivity_threshold` (`float`, 0.0-1.0): Base sensitivity for detecting behavioral anomalies (e.g., response length deviations).
*   `initial_avg_length` (`int`): Starting guess for average response length (used initially).
*   `behavior_monitoring_factor` (`float`): Multiplier applied to `sensitivity_threshold` to adjust the acceptable deviation range for length checks.

##### `TextAnalysis`

*   `syntax_non_alnum_threshold` (`float`): Max allowed ratio of non-alphanumeric/non-space characters in a prompt.
*   `syntax_max_word_length` (`int`): Max allowed length for a single "word".

##### `ListFiles`

*   **Purpose:** Defines the paths (relative to the project root) for external data files used by components and demos.
*   **Keys:**
    *   `patterns`: Path to injection patterns file (used by `InputValidator`).
    *   `refusal_keywords`: Path to refusal keywords file (used by `RealtimeMonitor`).
    *   `compliance_keywords`: Path to compliance/sensitive keywords file (used by `RealtimeMonitor`).
    *   `sample_prompts`: Path to sample prompts file (used by `demo_mock.py`).
    *   `boundary_refusal_keywords`: Path to boundary test refusal keywords (used by `demo_ollama.py`).

### List Files (`config_data/`)

External lists like injection patterns and keywords are stored as plain text files (one item per line, `#` comments ignored) within the `config_data/` directory. The specific file used for each list type is determined by the paths set in the `ListFiles` section of `config.yaml`.

To customize these lists:

1.  **Edit the files** within `config_data/` directly.
2.  **Modify `config.yaml`** to point to different files (ensure they are placed correctly relative to the project root).
3.  **Pass the content directly:** When initializing components like `InputValidator` or `RealtimeMonitor` programmatically, you can pass the list/set content directly via arguments (e.g., `patterns=[...]`, `refusal_keywords={...}`), bypassing the file loading mechanism.

## Logging

-   The `frogshield` library modules use Python's standard `logging` module.
-   The demo scripts configure basic console logging to show `INFO` level messages by default.
-   To see more detailed `DEBUG` messages (e.g., specific pattern matches, analysis steps), use the `--debug` flag when running the demo scripts (e.g., `python demo_ollama.py --prompt "Hello" --debug`).

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/blakeben/FrogShield.git # Or your fork
    cd FrogShield
    ```

2.  **Set up Python Environment:** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows use venv\Scripts\activate
    ```

3.  **Install Dependencies:** Install the core library (editable mode) and demo dependencies.
    ```bash
    # Installs frogshield + core deps (PyYAML)
    pip install -e .

    # Installs demo deps (ollama)
    pip install -r requirements.txt
    ```

4.  **Verify Configuration:** Ensure `config.yaml` and the necessary files within `config_data/` exist.

5.  **Prepare Ollama (if using `demo_ollama.py`):**
    *   Ensure Ollama is installed and the server is running (`ollama serve` in a separate terminal).
    *   Pull the desired model (the demo defaults to `llama3`): `ollama pull llama3`

6.  **Run Demos:**
    *   **Mock Demo:** Runs through predefined prompts loaded from `config_data/sample_prompts.txt`.
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
    *   **Ollama Demo (Boundary Test):** Run the predefined boundary test suite using keywords from `config_data/boundary_refusal_keywords.txt`.
        ```bash
        python demo_ollama.py --test-boundaries
        ```
    *   **Specify Model/Debug:** Use `--model <model_name>` or `--debug` with `demo_ollama.py`.

7.  **Run Unit Tests:**
    ```bash
    python -m unittest discover -v
    ```

## Current Status & Limitations

-   **Basic Functionality:** Core components (`InputValidator`, `RealtimeMonitor`, `ModelHardener`) are implemented with foundational logic.
-   **Centralized Config:** List data (patterns, keywords) is externalized to `config_data/` and managed via `config.yaml`.
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

