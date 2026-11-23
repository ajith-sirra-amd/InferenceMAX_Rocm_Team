"""Run an owned benchmark arm only while its allocation remains verifiable."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from allocation import verify


def release_owned(output):
    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                                   'label=ipat.pr.output=' + str(output)], text=True, timeout=15).split()
    actions = []
    for cid in ids:
        info = json.loads(subprocess.check_output(['docker', 'inspect', cid], text=True, timeout=15))[0]
        assert info['Config']['Labels'].get('ipat.pr.output') == str(output)
        immutable = info['Id']
        result = subprocess.run(['docker', 'stop', '--time', '30', immutable],
                                capture_output=True, text=True, timeout=45)
        actions.append(dict(container_id=immutable, exit_code=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
    return actions


def run(argv, journal, output, interval=15):
    journal, output = Path(journal), Path(output).resolve()
    base = Path('/share_nfs/gguasti/benchmarks').resolve()
    assert output.is_relative_to(base) and output.parts[len(base.parts)].startswith('pr-perf-')
    assert journal.resolve().is_relative_to(output)
    assert interval > 0
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('x') as stream:
        def emit(event, **values):
            stream.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                         event=event, **values)) + '\n')
            stream.flush()
        try:
            allocation = verify()
        except Exception as error:
            emit('refused_before_launch', error=str(error))
            return 125
        emit('allocation_verified_before_launch', allocation=allocation, argv=argv)
        child = subprocess.Popen(argv, start_new_session=True)
        while child.poll() is None:
            time.sleep(interval)
            if child.poll() is not None:
                break
            try:
                allocation = verify()
                emit('allocation_verified', allocation=allocation, child_pid=child.pid)
            except Exception as error:
                emit('allocation_unavailable_stop_owned_arm', error=str(error), child_pid=child.pid)
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    emit('owned_container_release', actions=release_owned(output))
                except Exception as cleanup_error:
                    emit('owned_release_error', error=str(cleanup_error))
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=15)
                emit('allocation_failure_terminal', child_returncode=child.returncode)
                return 125
        try:
            emit('allocation_verified_after_arm', allocation=verify())
        except Exception as error:
            emit('allocation_unavailable_after_arm', error=str(error), child_returncode=child.returncode)
            return 125
        emit('arm_terminal', returncode=child.returncode)
        return child.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert argv
    raise SystemExit(run(argv, args.journal, args.output, args.interval))
