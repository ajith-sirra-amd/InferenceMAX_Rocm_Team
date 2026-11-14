"""Compare exact-revision natural screen evidence without promoting quality."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'quality'))
import compare as quality_compare
import screen

def compare(root):
    results=[]
    _,fixture=screen.assets()
    for arm in ['baseline','candidate']:
        directory=root/arm
        loaded=json.loads((directory/'model-result.json').read_text())
        assert loaded['status']=='collected' and loaded['loaded_ranks']==8
        summary=json.loads((directory/'quality/summary.json').read_text())
        assert summary['status']=='collected' and summary['score_denominator']==48
        quality_compare.acceptance(directory/'quality',summary)
        rows=[json.loads(line) for line in (directory/'quality/responses.jsonl').read_text().splitlines()]
        assert len(rows)==48 and {r['id'] for r in rows}=={r['id'] for r in fixture}
        records={r['id']:r for r in rows}
        for row in fixture:
            observed=records[row['id']]
            assert observed['request']==screen.prompt(row,'moonshotai/Kimi-K3.1-Pro')
            assert observed['expected_answer']==screen.answer(row['answer'])
            assert observed['valid_answer'] and observed['error'] is None
            choice=observed['response']['choices'][0]
            message=choice['message']
            extracted=screen.answer(message.get('content') or message.get('reasoning_content') or '')
            assert choice['finish_reason']=='stop' and extracted is not None
            assert observed['extracted_answer']==extracted
            assert observed['correct']==(extracted==observed['expected_answer'])
        assert sum(r['correct'] for r in rows)==summary['correct']
        recipe=json.loads((directory/'recipe.json').read_text())
        normalized=copy.deepcopy(recipe)
        additional=json.loads(normalized['server_arguments']['additional-config'])
        identity=json.loads((directory/'runtime-manifest.json').read_text())
        if identity['pair']['pr']==67373:
            assert normalized['server_environment'].pop('VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE')==('1' if arm=='candidate' else '0')
            assert loaded['attnres_admission_ranks']==8
        else:
            assert additional.pop('dspark_context_projection_sharding',False)==(arm=='candidate')
        normalized['server_arguments']['additional-config']=additional
        results.append((summary,records,normalized))
    a,b=results
    assert a[2]==b[2],'Serving recipes differ beyond the tested PR flag'
    transitions=[dict(id=key,baseline_correct=a[1][key]['correct'],candidate_correct=b[1][key]['correct']) for key in a[1]]
    return dict(status='screen_collected',baseline_correct=a[0]['correct'],candidate_correct=b[0]['correct'],score_denominator=48,baseline_acceptance_length=a[0]['acceptance_length'],candidate_acceptance_length=b[0]['acceptance_length'],transitions=transitions,quality_qualified=False,limits='One matched sampled48-question screen; not full model quality, performance or unchanged exact-PR validation')

if __name__=='__main__':
    root=Path(sys.argv[1]);result=compare(root)
    (root/'model-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
