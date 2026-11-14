"""Build and verify explicitly derived MXFP4 runtime pairs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import runtime

BUNDLE = Path(__file__).with_name('mxfp4_compatibility')
def bundle(pr, variant='original'):
    assert variant in ('original','syncfree','prefill_pinned')
    name = 'mxfp4_compatibility' if pr == 67372 else 'mxfp4_compatibility_67373'
    return BUNDLE.with_name(name + ('_'+variant if variant!='original' else ''))
def manifest(pr=67372, variant='original'):
    return json.loads((bundle(pr,variant)/'patch-manifest.json').read_text())

def probe_derived(image, output, pr=67372, variant='original'):
    m = manifest(pr,variant)
    vs = [c for c in m['changes'] if c['path'].startswith('vllm/')]
    a = next(c for c in m['changes'] if c['path'].startswith('aiter/'))
    original = runtime.PROBE
    guard = "if dirty:\n    raise RuntimeError('Baked full-source checkout is dirty')"
    expected_dirty = sorted('M '+c['path'] for c in vs)
    replacement = "if sorted(line.strip() for line in dirty.splitlines()) != " + repr(expected_dirty) + ":\n    raise RuntimeError('Unexpected compatibility checkout changes: ' + dirty)"
    assert guard in original
    tail = ''.join("\nassert hashes["+repr(v['path'])+"] == "+repr(v['after_sha256']) for v in vs)
    # Must validate before emitting the JSON record.
    check = "assert hashlib.sha256(pathlib.Path('/usr/local/lib/python3.12/dist-packages/"+a['path']+"').read_bytes()).hexdigest() == "+repr(a['after_sha256'])+"\n"
    code = original.replace(guard, replacement).replace('print(json.dumps(dict(source_sha=sha', tail+'\n'+check+'print(json.dumps(dict(source_sha=sha')
    try:
        runtime.PROBE = code
        return runtime.probe_image(image, output)
    finally:
        runtime.PROBE = original

def verify(item, output):
    pr = item['compatibility'].get('pr',67372)
    variant = item['compatibility'].get('variant','original')
    m = manifest(pr,variant)
    assert item['compatibility']['manifest_sha256'] == hashlib.sha256((bundle(pr,variant)/'patch-manifest.json').read_bytes()).hexdigest()
    parent = runtime.probe_image(item['parent_image'], output.with_name('parent-probe.json'))
    derived = probe_derived(item['image'], output, pr,variant)
    assert derived['source_sha'] == parent['source_sha'] == item['source_sha']
    changed = {k for k in parent['source_sha256'] if parent['source_sha256'][k] != derived['source_sha256'].get(k)}
    assert changed == {c['path'] for c in m['changes'] if c['path'].startswith('vllm/')}, changed
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
    pr = json.loads(parent_path.read_text())['pr']
    variant = 'prefill_pinned'
    m = manifest(pr,variant)
    pair = runtime.load_pair(parent_path,pr)
    assert pair == m['current_pair']
    records = {}
    for arm in ['baseline','candidate']:
        work = output/arm;work.mkdir()
        parent = runtime.probe_image(pair[arm]['image'],work/'parent-probe.json')
        assert parent['source_sha256'][m['changes'][0]['path']] == m['changes'][0]['before_sha256']
        aiter = m['changes'][1]
        aiter_check = "import hashlib;from pathlib import Path;print(hashlib.sha256(Path('/usr/local/lib/python3.12/dist-packages/"+aiter['path']+"').read_bytes()).hexdigest())"
        actual_aiter = subprocess.check_output(['docker','run','--rm','--network','none','--entrypoint',runtime.PYTHON,pair[arm]['image'],'-c',aiter_check],text=True).strip()
        assert actual_aiter == aiter['before_sha256'], 'Parent AITER reader differs from frozen compatibility input'
        copies = []
        for index,change in enumerate(m['changes']):
            if change['path'].startswith('vllm/'):
                assert parent['source_sha256'][change['path']] == change['before_sha256']
            raw = (bundle(pr,variant)/(change['patch']+'.source-after')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == change['after_sha256']
            filename = f'frozen-{index}.py'
            (work/filename).write_bytes(raw)
            copies.append('COPY '+filename+' /usr/local/lib/python3.12/dist-packages/'+change['path']+'\n')
            if change['path'].startswith('vllm/'):
                copies.append('COPY '+filename+' /opt/ipat/vllm/'+change['path']+'\n')
        tag = 'ipat-mxfp4-parent:'+pair[arm]['image'].split(':')[1]
        subprocess.run(['docker','tag',pair[arm]['image'],tag],check=True)
        assert subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip() == pair[arm]['image']
        dockerfile = 'FROM '+tag+'\n'+''.join(copies)
        (work/'Dockerfile').write_text(dockerfile)
        subprocess.run(['docker','build','--network','none','--iidfile',str(work/'image-id.txt'),str(work)],check=True)
        item = dict(pair[arm],parent_image=pair[arm]['image'],image=(work/'image-id.txt').read_text().strip(),compatibility=dict(kind='historical_MXFP4_compatibility',pr=pr,variant=variant,manifest_sha256=hashlib.sha256((bundle(pr,variant)/'patch-manifest.json').read_bytes()).hexdigest()))
        verify(item,work/'runtime-verification.json')
        records[arm] = item
    pair.update(records)
    (output/'pair.json').write_text(json.dumps(pair,indent=2)+'\n')
    print(json.dumps(dict(status='verified_derived_pair',manifest=str(output/'pair.json'))))

if __name__ == '__main__':
    build(Path(sys.argv[1]),Path(sys.argv[2]))
