"""Run all pinned GSM8K test rows with unchanged sampled prompts and native logprobs."""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'quality'))
import screen

def assets():
    directory=Path(screen.__file__).parent
    config,fixture=screen.assets()
    provenance=json.loads((directory/'dataset-provenance.json').read_text())
    data=(directory/'gsm8k-test-full.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==provenance['source_sha256']
    raw=[json.loads(line) for line in data.splitlines()]
    rows=[dict(id=f'gsm8k-test-{i:04d}',index=i,**row) for i,row in enumerate(raw)]
    assert len(rows)==1319 and rows[:48]==fixture
    assert all(screen.answer(row['answer']) is not None for row in rows)
    config['dataset_sha256']=provenance['source_sha256']
    return config,rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoint',required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    code=screen.capture(args,dataset=assets(),request_extra=dict(logprobs=True),scope='All1319 pinned GSM8K test items, zero-shot sampled with native logprobs; first48 overlap prior screen. Not official five-shot GSM8K, broad model quality or performance.')
    rows=[json.loads(line) for line in (args.output/'responses.jsonl').read_text().splitlines()]
    tokens=0;errors=[]
    for row in rows:
        try:
            content=row['response']['choices'][0]['logprobs']['content']
            assert content
            for token in content:
                assert type(token['logprob']) in (int,float) and math.isfinite(token['logprob']) and token['logprob']<=0
                tokens+=1
        except (KeyError,TypeError,AssertionError) as e:errors.append(dict(id=row['id'],error=type(e).__name__))
    screen.save(args.output/'logprobs-validation.json',dict(status='passed' if not errors and len(rows)==1319 else 'failed',requests=len(rows),tokens=tokens,errors=errors,limits='Generated-token log probabilities; sampled continuations can differ, so these are not common-sequence teacher-forced parity.'))
    return code or int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
