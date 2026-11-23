"""Explicit allocation binding for portable frozen PR execution."""
import datetime
import os
import re
import subprocess


def verify(minimum_seconds=0, job=None, node=None):
    job = str(job or os.environ.get('IPAT_ALLOCATION_JOB', '29088'))
    node = node or os.environ.get('IPAT_ALLOCATION_NODE', 'pit2-p03-g34')
    if not re.fullmatch(r'[0-9]+', job) or not re.fullmatch(r'pit2-p03-g[0-9]+', node):
        raise ValueError('Invalid allocation identity')
    actual = subprocess.check_output(['hostname'], text=True, timeout=15).strip()
    assert actual == node, 'Execution node differs from frozen allocation binding'
    raw = subprocess.check_output(['scontrol', 'show', 'job', job], text=True, timeout=15)
    fields = dict(re.findall(r'(\w+)=(\S+)', raw))
    assert fields.get('JobId') == job
    assert fields.get('JobState') == 'RUNNING', f'Allocation {job} state is {fields.get("JobState")}'
    assert fields.get('NodeList') == node, 'Scheduler node binding differs'
    assert fields.get('UserId') == 'giovanni.guasti@amd.com', 'Allocation owner differs'
    # Slurm emits local wall-clock EndTime; preserve the verified node timezone.
    end = datetime.datetime.fromisoformat(fields['EndTime']).astimezone()
    remaining = (end - datetime.datetime.now().astimezone()).total_seconds()
    assert remaining >= minimum_seconds, f'Allocation needs extension: remaining={remaining:.0f}s required={minimum_seconds}s'
    return dict(job_id=job, node=node, owner=fields['UserId'],
                expiry=end.isoformat(), remaining_seconds=remaining)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-seconds', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(verify(args.minimum_seconds)), flush=True)
