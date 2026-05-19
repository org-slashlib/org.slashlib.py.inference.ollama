# -*- coding: utf-8 -*-
# file tests/02_01_00_adapter.py
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
from org.slashlib.py.inference.ollama.adapter import OllamaInferenceAdapter

@pytest.mark.asyncio
@patch("org.slashlib.py.configloader.resolve")
@patch("ollama.AsyncClient")
async def test_ollama_adapter_chat_success(mock_client_class, mock_config):
    """
    Test successful chat call and parameter resolution.
    Targeting the priority chain: kwargs -> method arg -> config.
    """
    # Setup mocks
    mock_config.side_effect = lambda key: "config_val" if "model" in key else 30
    mock_instance = AsyncMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = {
        "message": {"role": "assistant", "content": "AI response"}
    }

    adapter = OllamaInferenceAdapter()
    
    # 1. Test priority: Explicit 'model' arg should win over config
    result = await adapter.chat(model="explicit-model", messages=[{"role": "user", "content": "hi"}])
    
    assert result.content == "AI response"
    mock_instance.chat.assert_called_once()
    args, kwargs = mock_instance.chat.call_args
    assert kwargs["model"] == "explicit-model"


@pytest.mark.asyncio
@patch("org.slashlib.py.configloader.resolve")
@patch("ollama.AsyncClient")
async def test_ollama_adapter_error_mapping(mock_client_class, mock_config):
    """
    Test mapping of ollama exceptions to internal inference exceptions.
    """
    mock_instance = AsyncMock()
    mock_client_class.return_value = mock_instance
    adapter = OllamaInferenceAdapter()

    # Test ResponseError -> InferenceConfigError
    mock_instance.chat.side_effect = ollama.ResponseError("Model not found")
    with pytest.raises(InferenceConfigError):
        await adapter.chat(messages=[])

    # Test RequestError -> InferenceConnectionError
    mock_instance.chat.side_effect = ollama.RequestError("Conn failed")
    with pytest.raises(InferenceConnectionError):
        await adapter.chat(messages=[])

    # Test Generic Exception -> InferenceError
    mock_instance.chat.side_effect = RuntimeError("Boom")
    with pytest.raises(InferenceError, match="Internal adapter error"):
        await adapter.chat(messages=[])


@pytest.mark.asyncio
@patch("ollama.AsyncClient")
async def test_ollama_adapter_empty_message_payload(mock_client_class):
    """Targets the 'if not message' branch in chat."""
    mock_instance = AsyncMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = {"something": "else"} # no 'message' key
    
    adapter = OllamaInferenceAdapter()
    with pytest.raises(InferencePayloadError, match="returned an empty response"):
        await adapter.chat(messages=[])


@pytest.mark.asyncio
@patch("org.slashlib.py.configloader.resolve")
@patch("ollama.AsyncClient")
async def test_ollama_adapter_think_kwarg_override(mock_client_class, mock_config):
    """
    Test that explicit 'think' in kwargs overrides any other logic (Line 68 coverage).

    Why:
        To ensure that passing an explicit boolean for 'think' short-circuits 
        the tool checking and configuration defaults, hitting line 68.

    Assumptions & Design Decisions:
        Utilizes an explicit `think=False` kwarg while providing tools, 
        proving that the explicit argument takes structural priority.
    """
    mock_instance = AsyncMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = {"message": {"role": "assistant", "content": "ok"}}
    
    adapter = OllamaInferenceAdapter()
    
    # Even with tools provided, think=False kwarg should win and bypass line 71
    await adapter.chat(messages=[], tools=[{"type": "function"}], think=False)
    
    args, kwargs = mock_instance.chat.call_args
    assert kwargs["think"] is False


@pytest.mark.asyncio
@patch("org.slashlib.py.configloader.resolve")
@patch("ollama.AsyncClient")
async def test_ollama_adapter_think_auto_true_with_tools(mock_client_class, mock_config):
    """
    Test that 'think' defaults to True if non-empty tools are provided (Line 71 coverage).

    Why:
        To achieve 100%++ path coverage for the conditional branch where 
        tools trigger automated thinking behavior without explicit user overriding.

    Assumptions & Design Decisions:
        Omits the 'think' parameter from kwargs entirely, relying on the 
        non-empty list structure to execute the truthy validation path.
    """
    mock_instance = AsyncMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = {"message": {"role": "assistant", "content": "ok"}}
    
    adapter = OllamaInferenceAdapter()
    
    # No think kwarg, but non-empty tools list triggers line 71 -> True
    await adapter.chat(messages=[], tools=[{"type": "function"}])
    
    args, kwargs = mock_instance.chat.call_args
    assert kwargs["think"] is True

# end of file tests/02_01_00_adapter.py