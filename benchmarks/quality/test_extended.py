import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'pr_runtime'))
from full_quality import extended_answer,typed_answer

class ClockAnswerTests(unittest.TestCase):
    def test_percentage_requires_declared_percentage_item(self):
        self.assertEqual(typed_answer({'id':'gsm8k-test-0806'},'#### 70%'),'7E+1')
        self.assertEqual(typed_answer({'id':'gsm8k-test-0806'},'#### 0.70%'),'0.7')
        self.assertIsNone(typed_answer({'id':'gsm8k-test-1001'},'#### 70%'))
        self.assertIsNone(typed_answer({'id':'gsm8k-test-0806'},'70% of students'))
        self.assertIsNone(typed_answer({'id':'gsm8k-test-0806'},'#### 70% of students'))
    def test_clock_only_applies_to_declared_item(self):
        row={'id':'gsm8k-test-1001'}
        self.assertEqual(extended_answer(row,'#### 2:00 pm'),'2')
        self.assertEqual(extended_answer(row,'#### 2:30 pm'),'2.5')
        self.assertIsNone(extended_answer(row,'#### 2:00 am'))
        self.assertIsNone(extended_answer(row,'#### 2:99 pm'))
        self.assertIsNone(extended_answer({'id':'gsm8k-test-0001'},'#### 2:00 pm'))
        self.assertEqual(extended_answer(row,'#### 3'),'3')
        self.assertIsNone(extended_answer(row,'I would start at 2:00 pm'))

class BudgetProtocolTests(unittest.TestCase):
    def test_v4_changes_budget_and_retains_typed_extraction(self):
        import json, tempfile, os
        from unittest.mock import patch
        import full_quality
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            captured={}
            def capture(args, **kwargs):
                captured.update(kwargs)
                rows=[dict(id=f'gsm8k-test-{i:04d}', response={'choices':[{'logprobs':{'content':[{'logprob':-0.5}]}}]}) for i in range(1319)]
                (output/'responses.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                return 0
            with patch.dict(os.environ, {'IPAT_QUALITY_PROTOCOL':'extended32768_typed_v4'}), patch.object(sys,'argv',['full_quality','--endpoint','http://unused','--model','test','--output',folder]), patch.object(full_quality,'assets',return_value=({},[])), patch.object(full_quality.screen,'capture',side_effect=capture):
                self.assertEqual(full_quality.main(),0)
            self.assertEqual(captured['request_extra'],dict(logprobs=True,max_tokens=32768))
            self.assertIs(captured['answer_extractor'],full_quality.typed_answer)
            protocol=json.loads((output/'protocol.json').read_text())
            self.assertEqual(protocol['recovery_of'],43619543042)
            self.assertEqual(protocol['percentage_answer_item'],'gsm8k-test-0806')
            self.assertEqual(json.loads((output/'logprobs-validation.json').read_text())['requests'],1319)

if __name__=='__main__':unittest.main()
