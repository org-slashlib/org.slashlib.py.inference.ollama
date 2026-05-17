[Bottom](#license) [AI](AI.md) [CHANGELOG](CHANGELOG.md) [LICENSE](LICENSE.md)
# org.slashlib.py.inference.ollama

The official **Ollama** inference adapter for the `org.slashlib.py.agent` framework.

[![PyPI version](https://img.shields.io/pypi/v/org.slashlib.py.inference.ollama.svg?color=blue)](https://pypi.org/project/org.slashlib.py.inference.ollama/) [![PyPI-Test version](https://img.shields.io/badge/pypitest-latest-blue)](https://test.pypi.org/project/org.slashlib.py.inference.ollama/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 1. Overview & Architecture

This project provides a specialized implementation of the `InferenceAdapter` interface from the core **Agent Framework**. By decoupling the Ollama logic from the main framework, we ensure a lightweight core and allow for independent updates to the inference logic.

### Architectural Role
- **Adapter Pattern**: Bridges the standardized framework calls to the Ollama-specific API.
- **Provider Agnostic**: The framework consumes this adapter through a unified interface.
- **Plugin-Based**: Leverages Python entry points for seamless, zero-config integration.

---

## 2. Prerequisites

- **Ollama Service**: Must be installed and reachable (Default: `http://localhost:11434`).
- **Python**: Version 3.10 or higher.
- **Base Framework**: `org.slashlib.py.agent` must be installed in the same environment.

---

## 3. Installation

### Production / Standard
```bash
pip install org.slashlib.py.inference.ollama
```

### Development (Editable Mode)
If you are developing both the framework and this adapter, install both in editable mode to ensure the **Entry Points** are correctly registered in your current Python environment:

```bash
# 1. Install the framework
cd path/to/org.slashlib.py.agent
pip install -e .

# 2. Install this adapter
cd path/to/org.slashlib.py.inference.ollama
pip install -e .
```

---

## 4. Integration Logic (Entry Points)

This adapter is designed to be \"invisible\" to the end user. It registers itself via the `pyproject.toml` entry points:

```toml
[project.entry-points."org.slashlib.py.agent.inference"]
ollama = "org.slashlib.py.inference.ollama.adapter:OllamaInferenceAdapter"
```

The framework automatically scans this group and maps the name `ollama` to the `OllamaInferenceAdapter` class.

---

## 5. Configuration

The adapter's behavior is controlled via the framework's configuration loader (usually `pyproject.json`).

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | string | (required) | The name of the Ollama model (e.g., \"llama3\", \"gemma\"). |
| `think` | boolean | `true` | Enables/Disables thinking process visibility if supported. |
| `timeout` | float | `600.0` | Connection timeout in seconds. |

**Example `pyproject.json`:**
```json
{
  "inference": {
    "ollama": {
      "model": "gemma4",
      "think": true,
      "timeout": 300.0
    }
  }
}
```

---

## 6. Usage Example

You do not need to import any classes from this package directly. Use the Framework's factory method:

```python
import asyncio
from org.slashlib.py.agent.agent import Agent

async def main():
    # Load the agent - the framework resolves 'ollama' via Entry Points
    my_agent = Agent.from_plugin(
        identifier="my-local-assistant",
        plugin_name="ollama"
    )

    # All standard Agent methods are now available
    response = await my_agent.chat("How does the plugin system work?")
    print(f"Assistant: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Development & Testing

### Testing with Pytest
```bash
pytest tests/
```

### Logging
The adapter logs under the namespace `org.slashlib.py.inference.ollama`.

---

## 8. Documentation & Obsidian

The project root is fully prepared as an **Obsidian Vault**.

---

## License

Distributed under the **MIT License**. See `LICENSE.md` for more information.

---
© 2026 org.slashlib