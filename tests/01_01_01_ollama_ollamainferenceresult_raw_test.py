# -*- coding: utf-8 -*-
# file tests/01_01_01_ollama_ollamainferenceresult_raw_test.py
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

"""
Module: org.slashlib.py.inference.ollama.result
Class: OllamaInferenceResult
Method: raw (property)

Special Considerations:
Verifies that the `raw` property correctly returns the exact dictionary 
passed during initialization without modifications, ensuring data integrity.
"""

# Python imports
import typing

# Third party imports
import pytest

# Internal imports
from org.slashlib.py.inference.ollama.result import OllamaInferenceResult


def test_raw_property_existence():
    """
    Verify the integrity and existence of the raw property.

    Why:
        Mandated by the testing methodology to ensure the test object is 
        importable and has the expected interface before executing behavioral tests.

    Assumptions & Design Decisions:
        Assumes OllamaInferenceResult can be instantiated with valid mock data.
        Uses `hasattr` and `isinstance` checks to validate type and structure.
    """
    raw_data = {"role": "assistant", "content": "Test infrastructure check."}
    result = OllamaInferenceResult(raw_data)
    
    assert hasattr(OllamaInferenceResult, "raw"), "OllamaInferenceResult missing 'raw' attribute."
    assert isinstance(OllamaInferenceResult.raw, property), "'raw' must be implemented as a property."


def test_raw_property_returns_exact_dict():
    """
    Test that the raw property returns the identical dictionary reference.

    Why:
        To ensure that the raw response is preserved in its entirety and not 
        copied, mutated, or lost during initialization.

    Assumptions & Design Decisions:
        Verifies both object identity (`is`) and value equality (`==`) to 
        guarantee zero side-effects or unexpected data transformations.
    """
    raw_data = {
        "role": "assistant", 
        "content": "Hello world", 
        "tool_calls": [{"function": {"name": "calculate"}}]
    }
    result = OllamaInferenceResult(raw_data)
    
    assert result.raw == raw_data, "The content of the raw property does not match the input data."
    assert result.raw is raw_data, "The raw property did not return the exact object reference."


def test_raw_property_with_minimal_payload():
    """
    Test the raw property behavior with a minimal valid payload edge-case.

    Why:
        Ensures 100%++ path and edge-case coverage for alternative valid 
        initialization structures (e.g., tool calls only, content is None).

    Assumptions & Design Decisions:
        Ollama results can validly omit text content if tool calls are present. 
        The raw property must still transparently return this payload structure.
    """
    raw_data = {
        "role": "assistant", 
        "content": None, 
        "tool_calls": [{"function": {"name": "fetch_data"}}]
    }
    result = OllamaInferenceResult(raw_data)
    
    assert result.raw.get("content") is None
    assert result.raw == raw_data

# end of file tests/01_01_01_ollama_ollamainferenceresult_raw_test.py