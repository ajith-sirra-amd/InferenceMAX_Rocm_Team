"""Monitor evaluation without changing request deadlines or scoring."""
import json
import subprocess
import time

def tail(path, limit=1800):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode(errors="replace")

def run_evaluation(args, *, env, directory, server, server_log, stop, arm):
    client = None
    log = directory / "gsm8k.log"
    try:
        with log.open("w") as output:
            client = subprocess.Popen(list(map(str, args)), env=env, cwd=directory,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 7200
            next_report = 0
            while client.poll() is None:
                now = time.monotonic()
                if server.poll() is not None:
                    raise RuntimeError("Server exited during GSM8K: " + tail(server_log))
                if now >= deadline:
                    raise TimeoutError("GSM8K exceeded unchanged 7200-second deadline")
                if now >= next_report:
                    print("GSM8K_PROGRESS", json.dumps({
                        "arm": arm, "remaining_seconds": round(deadline-now),
                        "client_tail": tail(log), "server_tail": tail(server_log),
                        "request_events_tail": tail(directory / "gsm8k-request-events.jsonl"),
                    }), flush=True)
                    next_report = now + 60
                try:
                    client.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if client.returncode:
            text = log.read_text(errors="replace")
            index = max(0, text.find("Traceback (most recent call last):"))
            raise RuntimeError("GSM8K failed: " + text[index:index+4000] + "\n" + text[-2000:])
    finally:
        stop(client)
