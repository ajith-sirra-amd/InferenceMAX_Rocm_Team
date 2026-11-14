"""Compare native agentic aggregates, retaining request accounting and metric scope."""
import copy,json,math,sys
from pathlib import Path

def compare(root):
    replicas=[]
    for replica in [1,2]:
        arms=[];recipes=[]
        for arm in ['baseline','candidate']:
            d=root/f'replica-{replica}'/arm
            assert json.loads((d/'performance-result.json').read_text())['status']=='collected'
            recipe=json.loads((d/'recipe.json').read_text());recipe.pop('image')
            additional=json.loads(recipe['server_arguments']['additional-config'])
            assert additional.pop('dspark_context_projection_sharding',False)==(arm=='candidate')
            recipe['server_arguments']['additional-config']=additional;recipes.append(recipe)
            check=json.loads((d/'agentic-client-start.json').read_text())
            assert check['native_plan']['replay']['duration']==3600 and check['native_plan']['replay']['concurrency']==48
            aggregate=json.loads((d/'agentic.json').read_text())
            arms.append(aggregate)
        assert recipes[0]==recipes[1],'Paired effective server recipes differ'
        replicas.append(dict(replica=replica,baseline=arms[0],candidate=arms[1]))
    return dict(status='native_agentic_pairs_collected',replicas=replicas,replicas_per_arm=2,quality_qualified=False,limits='Native fixed-time metrics and filtering retained; synthetic acceptance3.57, full accuracy and profiling overhead remain separate gates. Two replicas do not establish tight confidence intervals.')

if __name__=='__main__':
    root=Path(sys.argv[1]);result=compare(root)
    (root/'agentic-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],replicas_per_arm=result['replicas_per_arm'])))
