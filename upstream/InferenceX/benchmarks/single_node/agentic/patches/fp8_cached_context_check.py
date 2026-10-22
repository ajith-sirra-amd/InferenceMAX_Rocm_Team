"""Populate and reuse a long prefix before GSM8K; verify real HTTP responses."""
import json
import pathlib
import sys
import urllib.request

port, model, result_dir = sys.argv[1:]
prefix = ("This is reference filler for a cached-context execution check. " * 1400)
responses = []
for tail in ["What is 2 plus 3? Answer with one number.",
             "What is 2 plus 4? Answer with one number."]:
    data = json.dumps({"model": model, "messages": [
        {"role": "user", "content": prefix + "\n" + tail}],
        "temperature": 0, "max_tokens": 128, "stream": False}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.load(response)
    message = payload["choices"][0]["message"]
    if not any(message.get(key) for key in ["content", "reasoning", "reasoning_content", "tool_calls"]):
        raise RuntimeError("Cached-context check returned empty response")
    responses.append(payload)
pathlib.Path(result_dir, "cached_context_responses.json").write_text(json.dumps(responses, indent=2))
print("[FP8-FUSED-CACHED-CONTEXT-HTTP-PASSED] requests=2 nonempty=True")
