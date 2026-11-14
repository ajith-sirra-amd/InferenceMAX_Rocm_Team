"""Run all pinned GSM8K test rows with unchanged sampled prompts and native logprobs."""
import argparse,hashlib,json,math,sys,re
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

def extended_answer(row,content):
    numeric=screen.answer(content)
    if numeric is not None or row['id']!='gsm8k-test-1001':return numeric
    matches=re.findall(r'(?im)^\s*####\s*(1[0-2]|[1-9]):([0-5][0-9])\s*(am|pm)\s*\.?\s*$',content)
    if not matches:return None
    hour,minute,period=matches[-1]
    # This pinned item asks for the latest start before 5pm on the same day.
    # Compare a clock only when it is explicitly PM; AM is not equivalent.
    if period.lower()!='pm':return None
    from decimal import Decimal
    return str((Decimal(hour)+Decimal(minute)/60).normalize())

def typed_answer(row,content):
    numeric=extended_answer(row,content)
    if numeric is not None or row['id']!='gsm8k-test-0806':return numeric
    # This pinned item explicitly asks for a percentage; gold70 means70%.
    matches=re.findall(r'(?im)^\s*####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*%\s*\.?\s*$',content)
    if not matches:return None
    from decimal import Decimal
    return str(Decimal(matches[-1].replace(',','')).normalize())

def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoint',required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    name=__import__('os').environ.get('IPAT_QUALITY_PROTOCOL','original2048_numeric_v1')
    assert name in ['original2048_numeric_v1','extended8192_clock_v2','extended8192_typed_v3']
    extended=name!='original2048_numeric_v1'
    extra=dict(logprobs=True)
    if extended:extra['max_tokens']=8192
    dataset=assets()
    protocol=dict(protocol=name,max_tokens=8192 if extended else 2048,time_answer_item='gsm8k-test-1001' if extended else None,percentage_answer_item='gsm8k-test-0806' if name=='extended8192_typed_v3' else None,recovery_of=43617235695 if name=='extended8192_typed_v3' else (43611263542 if extended else None),limits='No denominator, truncation, request-error, finite-logprob, counter or no-score-drop gate relaxation.')
    extractor=typed_answer if name=='extended8192_typed_v3' else (extended_answer if extended else None)
    code=screen.capture(args,dataset=dataset,request_extra=extra,answer_extractor=extractor,scope='All1319 pinned GSM8K test items, zero-shot sampled with native logprobs; protocol '+name+'. Not official five-shot GSM8K, broad model quality or performance.')
    screen.save(args.output/'protocol.json',protocol)
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
