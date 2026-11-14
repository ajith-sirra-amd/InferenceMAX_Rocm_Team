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

if __name__=='__main__':unittest.main()
