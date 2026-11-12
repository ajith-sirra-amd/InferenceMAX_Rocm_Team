"""Check candidate activation across all eight serving workers."""

import json
from pathlib import Path
import re
import sys


def validate(output):
    effective = json.loads((output / 'effective-server.json').read_text())
    mode = effective['optimization']
    log = (output / 'server.log').read_text(errors='replace')
    expected = set(range(8))
    phases = {}
    if mode == 'attnres':
        patterns = {'prepared': r'ATTNRES_PRODUCT_PREPARED buffers=38\b',
                    'selected': r'ATTNRES_PRODUCT_SELECTED rows=(?:280|336) blocks=(?:4|8)\b'}
    elif mode in ('dspark', 'moe_front_dspark'):
        patterns = {phase: rf'DSPARK_CONTEXT_SHARD phase={phase} active=True rank=\d+ tp=8 .*local_weight_shape=\(896, 35840\) dtype=torch.bfloat16'
                    for phase in ('construct', 'loaded')}
    elif mode == 'moe_front':
        patterns = {'prepared': r'MOE_MERGED_FRONT_READY rank=\d+ active=True prepared=92 expected=92\b',
                    'selected': r'MOE_MERGED_FRONT_SELECTED rank=\d+ rows=(?:259|266|273|280|287|294|301|308|315|322|329|336)\b'}
    elif mode == 'baseline':
        patterns = {}
    else:
        raise ValueError('Unknown optimization mode: ' + mode)
    if mode == 'moe_front_dspark':
        patterns.update({'prepared': r'MOE_MERGED_FRONT_READY rank=\d+ active=True prepared=92 expected=92\b',
                         'selected': r'MOE_MERGED_FRONT_SELECTED rank=\d+ rows=(?:259|266|273|280|287|294|301|308|315|322|329|336)\b'})
    for phase, pattern in patterns.items():
        ranks = set()
        for line in log.splitlines():
            rank = re.search(r'Worker_TP(\d+)_DCP\d+', line)
            if rank and re.search(pattern, line):
                ranks.add(int(rank[1]))
        phases[phase] = sorted(ranks)
    errors = [f'{phase}: expected ranks0-7, observed {ranks}'
              for phase, ranks in phases.items() if set(ranks) != expected]
    if mode == 'baseline' and re.search(r'ATTNRES_PRODUCT_(?:PREPARED|SELECTED)|DSPARK_CONTEXT_SHARD|MOE_MERGED_FRONT_(?:READY|SELECTED)', log):
        errors.append('Candidate activation appeared in baseline')
    result = dict(status='failed' if errors else 'passed', optimization=mode,
                  phases=phases, errors=errors,
                  limits='Startup/graph activation only. Selection during capture does not identify measured replay frequency. No model-quality claim.')
    (output / 'optimization-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    result = validate(Path(sys.argv[1]))
    print(json.dumps(result))
    raise SystemExit(result['status'] != 'passed')
