"""Apply the frozen native Kimi parser to retained raw failed-screen outputs."""
import copy,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,'/quality')
import screen
from vllm.tokenizers import get_tokenizer
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
import vllm.reasoning.kimi_k3_reasoning_parser as native

root=Path('/raw');rows=[json.loads(line) for line in (root/'quality/responses.jsonl').read_text().splitlines()]
tokenizer=get_tokenizer('/model',trust_remote_code=True)
parser=KimiK3ReasoningParser(tokenizer,chat_template_kwargs={'thinking':True})
records=[]
for row in rows:
    request=ChatCompletionRequest(**row['request'])
    message=row['response']['choices'][0]['message']
    reasoning,content=parser.extract_reasoning(message['content'],request)
    response=copy.deepcopy(row['response']);response['choices'][0]['message']['content']=content;response['choices'][0]['message']['reasoning']=reasoning
    parsed=screen.answer(content or '')
    records.append(dict(id=row['id'],raw_valid=row['valid_answer'],parsed_answer=parsed,expected=row['expected_answer'],parsed_content=content,parsed_correct=parsed==row['expected_answer']))
out=dict(status='native_parser_diagnostic',records=records,valid=sum(r['parsed_answer'] is not None for r in records),correct=sum(r['parsed_correct'] for r in records),denominator=len(rows),parser_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),raw_responses_sha256=hashlib.sha256((root/'quality/responses.jsonl').read_bytes()).hexdigest(),limits='Offline native response-parser diagnostic; does not replace original failed-screen result or qualify serving/accuracy')
Path('/logs/kimi-parser-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'}))
