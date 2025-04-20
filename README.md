# FrogShield: LLM Prompt Injection Defense Framework (Educational Demo)

This project provides a basic, educational framework called `FrogShield` designed to demonstrate concepts for defending Large Language Models (LLMs) against prompt injection attacks.

**Disclaimer:** This is a simplified implementation for educational purposes and a cybersecurity course project. It is **not** production-ready and lacks the robustness required for real-world deployment.

## Purpose

The goal of FrogShield is to illustrate three key layers of defense against prompt injection:

1.  **Input Validation:** Analyzing incoming prompts against known malicious patterns loaded from `patterns.txt` and using placeholder functions (`utils/text_analysis.py`) to check for suspicious syntax or context manipulation.
2.  **Model Hardening (Conceptual):** The `model_hardener.py` module provides methods for *generating* adversarial examples and testing model boundaries conceptually.
3.  **Real-time Monitoring:** Analyzing the LLM's output for suspicious keywords or refusal messages, and monitoring basic behavioral patterns (like response length) for anomalies using placeholder logic (`realtime_monitor.py`).

## Components

The code is organized into the following structure:

-   `config.yaml`: Default configuration file for FrogShield components.
-   `frogshield/`: Directory containing the core defense logic modules.
    -   `__init__.py`: Makes `frogshield` importable and defines the public API.
    -   `input_validator.py`: Contains the `InputValidator` class.
    -   `model_hardener.py`: Contains the `ModelHardener` class.
    -   `realtime_monitor.py`: Contains the `RealtimeMonitor` class.
    -   `utils/`: Sub-directory for shared resources.
        -   `config_loader.py`: Utility for loading `config.yaml`.
        -   `__init__.py`: Package initializer (currently empty).
        -   `text_analysis.py`: Placeholder functions for syntax/context analysis.
    -   `tests/`: Contains basic unit tests using Python's `unittest`.
        -   `__init__.py`
        -   `test_validation.py`
        -   `test_monitoring.py`
        -   `test_hardening.py`
-   `pyproject.toml`: Project build and metadata configuration.
-   `patterns.txt`: External file containing known injection patterns (one per line).
-   `demo_mock.py`: Script demonstrating FrogShield with a mock LLM.
-   `demo_ollama.py`: Script demonstrating FrogShield with a local Ollama LLM.
-   `requirements.txt`: Lists dependencies needed for development or specific demos (e.g., `ollama`). Core library dependencies are listed in `pyproject.toml`.
-   `.gitignore`: Standard Python gitignore file.
-   `README.md`: This file.

## Configuration

-   **Main Configuration (`config.yaml`):** Most configurable parameters for `InputValidator`, `RealtimeMonitor`, and `TextAnalysis` utilities are defined in `config.yaml` located in the project root. The demo scripts load this file by default.
-   **Injection Patterns (`patterns.txt`):** Known injection patterns are loaded from `patterns.txt` (in the project root) by default. Customize by:
    -   Editing `patterns.txt` (add/remove patterns, one per line; lines starting with `#` are ignored).
    -   Passing a specific file path to the `InputValidator` constructor: `InputValidator(..., patterns_file='path/to/your/patterns.txt')`.
    -   Passing a list of patterns directly: `InputValidator(..., patterns=['pattern1', 'pattern2'])`.
-   **Overriding Config Values:** When initializing components directly (like in the demo scripts or your own code), you pass values loaded from the configuration file. You can override specific values by modifying the loaded dictionary before passing it or by passing arguments directly to the constructors if the class supports it (currently, explicit arguments are required for core settings like `context_window`, `sensitivity_threshold`, etc., overriding any potential defaults). Check the `__init__` methods for required parameters.

## Logging

-   The `frogshield` library modules use Python's standard `logging` module for internal messages instead of printing directly.
-   The demo scripts (`demo_mock.py`, `demo_ollama.py`) configure basic logging to show `INFO` level messages by default.
-   To see more detailed `DEBUG` messages (e.g., specific pattern matches, analysis steps), modify the `logging.basicConfig` level in the demo scripts:
    ```python
    # Change level=logging.INFO to level=logging.DEBUG
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ```

## How to Run

1.  **Clone the Repository:** Obtain the code.
2.  **Set up Environment:**
    *   Navigate to the project directory.
    *   Create and activate a Python virtual environment (Recommended):
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Install dependencies for demos/development:
        ```bash
        pip install -r requirements.txt
        ```
    *   Install the `frogshield` package in editable mode (also installs core dependencies like `PyYAML` listed in `pyproject.toml`):
        ```bash
        pip install -e .
        ```
3.  **Run Demos:**
    *   To run the basic demo with a mock LLM:
        ```bash
        python demo_mock.py
        ```
    *   To run the demo with a local Ollama model (ensure Ollama is installed, running, and the model is pulled, e.g., `ollama pull llama3`):
        ```bash
        python demo_ollama.py
        ```
4.  **Run Unit Tests:** To verify the individual components:
    ```bash
    python -m unittest discover frogshield/tests
    ```
    All tests should pass in the current implementation.

## Current Status & Limitations

-   The project includes basic packaging configuration (`pyproject.toml`) and can be installed locally (`pip install -e .`).
-   The core logic (pattern matching, syntax/context analysis, behavioral monitoring) uses simplified placeholders suitable for demonstration.
-   The `model_hardener.py` module demonstrates boundary testing but requires integration with an actual model training pipeline for other hardening techniques.
-   The framework has been tested via the included demo scripts and unit tests, which currently pass.

## Future Development

-   Implement more sophisticated pattern matching (regex, semantic analysis).
-   Develop robust syntax and context analysis logic (potentially using NLP libraries).
-   Integrate with other LLM APIs.
-   Implement more reliable baseline modeling and anomaly detection for behavioral analysis.
-   Develop actual model hardening techniques (adversarial training, RLHF integration).
-   Create more sophisticated adaptive response mechanisms.
-   Expand unit test coverage.

## Contributors

-   Ben Blake <ben.blake@tcu.edu>
-   Tanner Hendrix <t.hendrix@tcu.edu>

