# -*- coding: utf-8 -*-
# file src/org/slashlib/py/inference/ollama/adapter.py
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
import logging
import pathlib
import typing

# Third party imports
import ollama
import org.slashlib.py.agent as agent
import org.slashlib.py.configloader as config

# Internal imports
from org.slashlib.py.inference.ollama.result import OllamaInferenceResult

class OllamaInferenceAdapter(agent.InferenceAdapter):
    """
    Inference Adapter for the Ollama API.

    Handles asynchronous communication with an Ollama server, including 
    parameter resolution from config and error mapping to the neutral 
    inference exception hierarchy.
    """

    def __init__(self):
        """
        Initialize the adapter and its logger.
        """
        self.log = logging.getLogger(f"org.slashlib.py.inference.ollama.{pathlib.Path(__file__).stem}.{self.__class__.__name__}")

    async def chat(
        self, 
        model: typing.Optional[str] = None, 
        messages: typing.List[typing.Dict[str, typing.Any]] = None, 
        tools: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None,
        **kwargs
    ) -> agent.InferenceResult:
        """
        Sends a chat request to Ollama and returns a validated result.

        Resolves model parameters using a priority chain: 
        1. Explicit kwargs, 2. Method arguments, 3. pyproject.json via configloader.

        Args:
            model (Optional[str]): The model name. Defaults to config value.
            messages (List[Dict[str, Any]]): The message history/context.
            tools (Optional[List[Dict[str, Any]]]): JSON schemas of available tools.
            **kwargs: Additional parameters like 'timeout' or 'think'.

        Returns:
            agent.InferenceResult: An instance of OllamaInferenceResult.

        Raises:
            agent.InferenceConfigError: If the model is not found or config is invalid.
            agent.InferenceConnectionError: If the Ollama server is unreachable or times out.
            agent.InferencePayloadError: If the response from Ollama is malformed.
            agent.InferenceError: For any other unexpected errors during agent.
        """
        # Resolve parameters: priority is kwargs -> method arg -> pyproject.json
        target_model = model or kwargs.get("model", config.resolve("adapter.ollama.model"))
        timeout = kwargs.get("timeout", config.resolve("adapter.ollama.timeout"))
        think = kwargs.get("think", config.resolve("adapter.ollama.think"))

        try:
            self.log.debug(f"Ollama request: timeout={timeout}, model={target_model}, messages={messages}, tools={tools}, think={think}")
            
            client = ollama.AsyncClient(timeout=timeout)
            response = await client.chat(
                model=target_model,
                messages=messages,
                tools=tools,
                think=think
            )

            self.log.debug(f"Ollama response: {response}")
            
            message = response.get("message")
            if not message:
                raise agent.InferencePayloadError(f"Ollama API returned an empty response for model '{target_model}'.")

            return OllamaInferenceResult(message)

        except (ollama.ResponseError) as e:
            self.log.error(f"Ollama logical/config error: {e}", exc_info=True)
            raise agent.InferenceConfigError(f"Ollama model or config invalid: {str(e)}")
        except (ollama.RequestError) as e:
            self.log.error(f"Ollama communication error: {e}", exc_info=True)
            raise agent.InferenceConnectionError(f"Failed to connect to Ollama: {str(e)}")
        except agent.InferenceError:
            # Re-raise internal inference errors to prevent them from being caught by the general Exception block
            raise
        except Exception as e:
            self.log.error(f"Unexpected error in OllamaInferenceAdapter: {e}", exc_info=True)
            raise agent.InferenceError(f"Internal adapter error: {str(e)}")


__all__ = ["OllamaInferenceAdapter"]

# end of file file src/org/slashlib/py/inference/ollama/adapter.py