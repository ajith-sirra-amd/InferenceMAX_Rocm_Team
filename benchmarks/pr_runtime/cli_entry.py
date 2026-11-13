#!/opt/runtime-venv/bin/python
"""Use the prepared runtime's interpreter for the native vLLM CLI."""
from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    main()
