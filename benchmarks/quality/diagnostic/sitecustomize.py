"""Opt-in diagnostics only; untouched installed serving source."""
import os

if os.environ.get("KIMI_DIAGNOSTIC_ENABLE") == "1":
    try:
        import diagnostic_hooks
        diagnostic_hooks.install()
        if not diagnostic_hooks.INSTALLED:
            raise RuntimeError("Diagnostic hooks did not finish installing")
    except BaseException:
        # Ordinary sitecustomize exceptions are swallowed by Python. A requested
        # diagnostic must fail before serving rather than run without its hooks.
        import traceback
        traceback.print_exc()
        os._exit(78)
