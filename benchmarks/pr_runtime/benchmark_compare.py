"""Validate paired fixed results and report descriptive metrics without promotion."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'fixed_length'))
from validate import validate_result

def compare(root):
    records=[];recipes=[]
    for arm in ['baseline','candidate']:
        d=root/arm
        assert json.loads((d/'performance-result.json').read_text())['status']=='collected'
        result=validate_result(json.loads((d/'baseline.json').read_text()))
        assert result['status']=='passed',result
        recipe=json.loads((d/'recipe.json').read_text());recipe.pop('image')
        additional=json.loads(recipe['server_arguments']['additional-config'])
        assert additional.pop('dspark_context_projection_sharding',False)==(arm=='candidate')
        recipe['server_arguments']['additional-config']=additional
        recipes.append(recipe);records.append(result)
    assert recipes[0]==recipes[1],'Serving recipes differ beyond PR flag/image'
    a,b=records
    return dict(status='fixed_screen_collected',baseline=a,candidate=b,output_throughput_change_percent=100*(b['output_tokens_per_second']/a['output_tokens_per_second']-1),replicas_per_arm=1,quality_qualified=False,performance_qualified=False,limits='One fixed192-request pair with synthetic acceptance3.57; not repeated agentic E2E, full accuracy, or measured profiling overhead')

if __name__=='__main__':
    root=Path(sys.argv[1]);result=compare(root)
    (root/'performance-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
