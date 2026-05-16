# -*- coding: utf-8 -*-
# file src/org/slashlib/py/inference/ollama/__init__.py
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

from org.slashlib.py.inference.ollama.adapter import OllamaInferenceAdapter
from org.slashlib.py.inference.ollama.result  import OllamaInferenceResult

__all__ = ["OllamaInferenceAdapter", "OllamaInferenceResult"]

# end of file src/org/slashlib/py/inference/ollama/__init__.py