# -*- coding: utf-8 -*-
# file tests/01_01_00_result.py
# @AI:
# - INTEGRITY RULES:
#   - STRICT PRESERVATION: Do not remove, move, or modify ANY existing lines of code or comments 
#     unless they are the explicit target of the requested change. 
#   - DEBUG MARKERS: Commented-out code (e.g., debug prints) MUST be kept exactly where they are.
#   - WHITESPACE & STRUCTURE: Maintain all original empty lines and the existing file structure. 
#     Structural integrity takes precedence over "clean code" or "elegance".
#   - LEAD-IN/OUT: The very first and last lines (and all comments in between) are immutable anchors.
# - MAINTENANCE:
#   - Only update pydoc strings (args, returns, raises) if the function signature changes.
#   - Do NOT delete existing examples or descriptions in pydoc.
# - LANGUAGE: en-US for all comments and documentation.
#

# Python imports
import typing
from unittest.mock import AsyncMock, patch, MagicMock

# Third party imports
import pytest
import ollama
from org.slashlib.py.agent import (
    InferencePayloadError,
    InferenceConfigError,
    InferenceConnectionError,
    InferenceError
)

# Internal imports
from org.slashlib.py.inference.ollama.result import OllamaInferenceResult

def test_ollama_result_valid_text():
    """Test valid text response parsing."""
    raw = {"role": "assistant", "content": "Hello world", "tool_calls": None}
    result = OllamaInferenceResult(raw)
    assert result.role == "assistant"
    assert result.content == "Hello world"
    assert result.to_dict()["content"] == "Hello world"


def test_ollama_result_valid_tool():
    """Test valid tool call response (content may be None)."""
    raw = {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "test"}}]}
    result = OllamaInferenceResult(raw)
    assert result.tool_calls[0]["function"]["name"] == "test"


def test_ollama_result_invalid_missing_role():
    """Trigger InferencePayloadError due to missing role."""
    with pytest.raises(InferencePayloadError, match="role' is missing"):
        OllamaInferenceResult({"content": "no role"})


def test_ollama_result_invalid_empty():
    """Trigger InferencePayloadError when both content and tools are missing."""
    with pytest.raises(InferencePayloadError, match="Both 'content' and 'tool_calls' are empty"):
        OllamaInferenceResult({"role": "assistant", "content": None, "tool_calls": []})

# end of file tests/01_01_00_result.py