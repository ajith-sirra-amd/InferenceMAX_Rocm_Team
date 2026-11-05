"""Observe HTTP requests; preserve payloads, timeouts, responses and exceptions."""
import hashlib
import itertools
import json
import os
import time

import aiohttp

PATH = os.environ.get("REVIEW_GSM8K_TRACE")
_COUNTER = itertools.count()
_ORIGINAL_REQUEST = aiohttp.ClientSession._request
_ORIGINAL_JSON = aiohttp.ClientResponse.json

def record(event):
    if not PATH:
        return
    data = (json.dumps(event, ensure_ascii=True) + "\n").encode()
    fd = os.open(PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        while data:
            data = data[os.write(fd, data):]
    finally:
        os.close(fd)

async def observed_request(self, method, url, **kwargs):
    payload = kwargs.get("json")
    tracked = PATH and method.upper() == "POST" and "/v1/chat/completions" in str(url)
    if not tracked:
        return await _ORIGINAL_REQUEST(self, method, url, **kwargs)
    request_id = next(_COUNTER)
    started = time.monotonic()
    record({"event": "request_started", "id": request_id, "time": time.time(),
            "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "payload": payload})
    try:
        response = await _ORIGINAL_REQUEST(self, method, url, **kwargs)
        response._pr64288_trace = (request_id, started)
        record({"event": "response_headers", "id": request_id, "time": time.time(),
                "elapsed": time.monotonic()-started, "status": response.status})
        return response
    except BaseException as error:
        record({"event": "request_failed", "id": request_id, "time": time.time(),
                "elapsed": time.monotonic()-started, "error": type(error).__name__})
        raise

async def observed_json(self, *args, **kwargs):
    tracking = getattr(self, "_pr64288_trace", None)
    try:
        result = await _ORIGINAL_JSON(self, *args, **kwargs)
    except BaseException as error:
        if tracking:
            record({"event": "response_failed", "id": tracking[0], "time": time.time(),
                    "elapsed": time.monotonic()-tracking[1], "error": type(error).__name__})
        raise
    if tracking:
        record({"event": "request_completed", "id": tracking[0], "time": time.time(),
                "elapsed": time.monotonic()-tracking[1], "usage": result.get("usage"),
                "choices": [{"finish_reason": c.get("finish_reason"),
                             "content_chars": len((c.get("message") or {}).get("content") or ""),
                             "reasoning_chars": len((c.get("message") or {}).get("reasoning_content") or "")}
                            for c in result.get("choices", [])]})
    return result

if PATH:
    aiohttp.ClientSession._request = observed_request
    aiohttp.ClientResponse.json = observed_json
