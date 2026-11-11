"""Check that cancellation cleanup cannot select an unrelated container."""
import json
from pathlib import Path
import tempfile
import unittest

import cleanup


class OwnershipTests(unittest.TestCase):
    def test_requires_both_run_label_and_saved_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / '123-1'
            run.mkdir()
            (run / 'container-transport.jsonl').write_text(json.dumps({'container': 'sa-step-123-1-owned'}) + '\n')
            container = {'Name': '/sa-step-123-1-owned', 'Config': {'Labels': {'sa.run': '123-1'}}}
            self.assertEqual(cleanup.owned(container, root), '123-1')
            container['Name'] = '/someone-elses-server'
            self.assertIsNone(cleanup.owned(container, root))
            container['Name'] = '/sa-step-123-1-owned'
            container['Config']['Labels']['sa.run'] = '124-1'
            self.assertIsNone(cleanup.owned(container, root))
            container['Config']['Labels']['sa.run'] = '../123-1'
            self.assertIsNone(cleanup.owned(container, root))

    def test_rejects_manifest_outside_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'runs'
            root.mkdir()
            other = base / 'other'
            other.mkdir()
            (other / 'container-transport.jsonl').write_text(json.dumps({'container': 'owned'}) + '\n')
            (root / '123-1').symlink_to(other, target_is_directory=True)
            container = {'Name': '/owned', 'Config': {'Labels': {'sa.run': '123-1'}}}
            self.assertIsNone(cleanup.owned(container, root))


if __name__ == '__main__':
    unittest.main()
