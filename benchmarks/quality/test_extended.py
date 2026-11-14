import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'pr_runtime'))
from full_quality import extended_answer

class ClockAnswerTests(unittest.TestCase):
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
