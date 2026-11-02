import subprocess

import pytest

import wait_native_run as queue


def test_queue_retries_transient_api_failure_then_waits_for_success(monkeypatch):
    clock = [0]
    replies = iter([
        subprocess.CalledProcessError(1, ["gh", "api"]),
        {"head_sha": "a" * 40, "status": "in_progress", "conclusion": None},
        {"head_sha": "a" * 40, "status": "completed", "conclusion": "success"},
    ])

    def fetch(repo, run_id):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(queue, "get_run", fetch)
    monkeypatch.setattr(queue.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    queue.wait_for_run("owner/repo", 123, "a" * 40, 100, 10)
    assert clock[0] == 20


@pytest.mark.parametrize("sha,result,error", [
    ("b" * 40, "success", "different source SHA"),
    ("a" * 40, "failure", "ended with failure"),
])
def test_queue_rejects_unvalidated_predecessor(monkeypatch, sha, result, error):
    monkeypatch.setattr(queue, "get_run", lambda repo, run_id: {
        "head_sha": sha, "status": "completed", "conclusion": result,
    })
    with pytest.raises(RuntimeError, match=error):
        queue.wait_for_run("owner/repo", 123, "a" * 40, 100, 10)
