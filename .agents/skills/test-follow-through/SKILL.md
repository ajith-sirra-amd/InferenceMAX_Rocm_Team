---
name: test-follow-through
description: Follow agent-dispatched GitHub Actions tests and GPU validation through completion, inspect evidence, and act on results without waiting for the user to ask for status.
---

# Test follow-through

Use this workflow when initiating or resuming a test or validation run for this repository.
Dispatch is the start of the task, not its completion.

## Own the result

- Record the run, attempt, job, workflow revision, execution revision, tested PR revision, and intended acceptance checks.
- Keep the agent turn active until the dispatched test finishes and its result has been inspected. Do not end with only a run link or a background poller when the user expects follow-through.
- Check setup and test jobs every 30-60 seconds. For long performance replays, use the user's requested cadence, currently five minutes, while continuing useful authorized work between checks.
- Inspect tool output at each check. A poller that merely saves status does not react to failures or wake a finished conversation. Never describe it as autonomous follow-through.
- On completion or failure, inspect logs, annotations, test reports, and artifacts immediately; do not wait for a user status message. A successful workflow is not by itself a numerical, accuracy, or performance pass.

## Respond to evidence

- On failure, identify the first failing operation and distinguish setup, reference-code, kernel, numerical-threshold, and workload failures. Do not declare an asynchronous GPU error harmless without evidence.
- Preserve the failed evidence, fix the demonstrated problem within existing authorization, run appropriate checks, and dispatch the corrected test. Do not rerun an unchanged failure blindly.
- On success, verify expected tests actually executed, their pass/fail/skip counts, source provenance, and required optimized-path execution. Visible GPUs are not proof that every worker exercised the path.
- Continue to the next already-authorized validation step when its prerequisites pass. Keep native PR evidence separate from prototype results and fixed thresholds separate from task-level accuracy.
- Never widen tolerances, skip failing cases, replace the intended optimized path with a fallback, or mark a PR ready merely to obtain a green result.

## Communicate and preserve continuity

Report meaningful stage changes and failures promptly. State exactly what passed, what failed, what is still running, and the next action. Keep the task notes current, including relevant English/Chinese evidence documents.

Stop only when the requested validation work is complete, the user pauses it, or further action requires unavailable access or authorization. For a blocker, preserve the exact last result and the next required action. This skill does not grant new write permissions, authorize unrelated runs or messages, install a scheduler, or guarantee readiness for merge.
