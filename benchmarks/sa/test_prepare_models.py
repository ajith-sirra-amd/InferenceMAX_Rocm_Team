import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from prepare_models import verify_file


class CheckpointValidationTests(unittest.TestCase):
    def test_checks_hub_lfs_and_git_objects_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'weight'
            content = b'checkpoint bytes'
            path.write_bytes(content)
            lfs = SimpleNamespace(size=len(content), rfilename=path.name,
                                  lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()))
            git = SimpleNamespace(size=len(content), rfilename=path.name, lfs=None,
                blob_id=hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest())
            for metadata in (lfs, git):
                self.assertEqual(verify_file(path, metadata)['bytes'], len(content))
            path.write_bytes(b'x' * len(content))
            for metadata in (lfs, git):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    verify_file(path, metadata)
            path.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'file size'):
                verify_file(path, lfs)
