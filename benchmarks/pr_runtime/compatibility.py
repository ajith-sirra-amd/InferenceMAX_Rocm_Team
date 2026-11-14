"""Build and verify explicitly derived MXFP4 runtime pairs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import runtime

BUNDLE = Path(__file__).with_name('mxfp4_compatibility')
def manifest():
    return json.loads((BUNDLE/'patch-manifest.json').read_text())

def probe_derived(image, output):
    m = manifest()
    v, a = m['changes']
    original = runtime.PROBE
    guard = "if dirty:\n    raise RuntimeError('Baked full-source checkout is dirty')"
    replacement = "if dirty.strip() != 'M vllm/v1/attention/backends/mla/indexer.py':\n    raise RuntimeError('Unexpected compatibility checkout changes: ' + dirty)"
    assert guard in original
    tail = "\nassert hashes['vllm/v1/attention/backends/mla/indexer.py'] == " + repr(v['after_sha256'])
    # Must validate before emitting the JSON record.
    check = "assert hashlib.sha256(pathlib.Path('/usr/local/lib/python3.12/dist-packages/"+a['path']+"').read_bytes()).hexdigest() == "+repr(a['after_sha256'])+"\n"
    code = original.replace(guard, replacement).replace('print(json.dumps(dict(source_sha=sha', tail+'\n'+check+'print(json.dumps(dict(source_sha=sha')
    try:
        runtime.PROBE = code
        return runtime.probe_image(image, output)
    finally:
        runtime.PROBE = original

def verify(item, output):
    m = manifest()
    assert item['compatibility']['manifest_sha256'] == hashlib.sha256((BUNDLE/'patch-manifest.json').read_bytes()).hexdigest()
    parent = runtime.probe_image(item['parent_image'], output.with_name('parent-probe.json'))
    derived = probe_derived(item['image'], output)
    assert derived['source_sha'] == parent['source_sha'] == item['source_sha']
    changed = {k for k in parent['source_sha256'] if parent['source_sha256'][k] != derived['source_sha256'].get(k)}
    assert changed == {'vllm/v1/attention/backends/mla/indexer.py'}, changed
    assert parent['source_sha256'].keys() == derived['source_sha256'].keys()
    for key in ['dependencies_sha256','native_sha256','test_sha256','vendored_source_sha256']:
        assert parent[key] == derived[key], key
    inspect = json.loads(subprocess.check_output(['docker','image','inspect',item['image']],text=True))[0]
    runtime.verify_probe(item, derived, inspect['Config']['Labels'])
    assert inspect['Id'] == item['image']
    result = dict(image=item['image'], parent_image=item['parent_image'], compatibility=item['compatibility'], **derived)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

def build(parent_path, output):
    output.mkdir(parents=True,exist_ok=False)
    m = manifest()
    pair = runtime.load_pair(parent_path,67372)
    assert pair == m['current_pair']
    records = {}
    for arm in ['baseline','candidate']:
        work = output/arm;work.mkdir()
        parent = runtime.probe_image(pair[arm]['image'],work/'parent-probe.json')
        assert parent['source_sha256'][m['changes'][0]['path']] == m['changes'][0]['before_sha256']
        for change in m['changes']:
            raw = (BUNDLE/(change['patch']+'.source-after')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == change['after_sha256']
            (work/('indexer.py' if change['path'].startswith('vllm/') else 'rowgroup.py')).write_bytes(raw)
        dockerfile = 'FROM '+pair[arm]['image']+'\nCOPY indexer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py\nCOPY indexer.py /opt/ipat/vllm/vllm/v1/attention/backends/mla/indexer.py\nCOPY rowgroup.py /usr/local/lib/python3.12/dist-packages/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_rowgroup.py\n'
        (work/'Dockerfile').write_text(dockerfile)
        subprocess.run(['docker','build','--network','none','--iidfile',str(work/'image-id.txt'),str(work)],check=True)
        item = dict(pair[arm],parent_image=pair[arm]['image'],image=(work/'image-id.txt').read_text().strip(),compatibility=dict(kind='historical_MXFP4_compatibility',manifest_sha256=hashlib.sha256((BUNDLE/'patch-manifest.json').read_bytes()).hexdigest()))
        verify(item,work/'runtime-verification.json')
        records[arm] = item
    pair.update(records)
    (output/'pair.json').write_text(json.dumps(pair,indent=2)+'\n')
    print(json.dumps(dict(status='verified_derived_pair',manifest=str(output/'pair.json'))))

if __name__ == '__main__':
    build(Path(sys.argv[1]),Path(sys.argv[2]))
