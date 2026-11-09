"""Exercise transport adapters without a scheduler or network."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).parent


class AdapterTests(unittest.TestCase):
    def test_actions_token_and_release_bytes_keep_native_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / 'curl'
            fake.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['CAPTURE'], 'a') as f:
    f.write(json.dumps({'argv': args, 'config': sys.stdin.read()}) + '\\n')
if 'Accept: application/vnd.github+json' in args:
    print(json.dumps({'assets': [{'name': 'asset.sha256', 'url': 'https://api.github.com/assets/123'}]}))
else:
    Path(args[args.index('--output') + 1]).write_bytes(b'expected-checksum\\n')
''')
            fake.chmod(0o755)
            token = 'synthetic.header.payload=='
            env = dict(os.environ, GH_TOKEN=token, SA_REAL_CURL=str(fake), CAPTURE=str(root / 'calls'))
            target = root / 'download'
            subprocess.run([str(ROOT / 'download/curl'), '--fail', '--location',
                'https://github.com/owner/repo/releases/download/v1.2/asset.sha256',
                '--output', str(target)], check=True, env=env)
            self.assertEqual(target.read_bytes(), b'expected-checksum\n')
            calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
            self.assertIn('/releases/tags/v1.2', calls[0]['argv'][-1])
            for call in calls:
                self.assertNotIn(token, ' '.join(call['argv']))
                self.assertEqual(call['config'], 'header = ' + json.dumps('Authorization: token ' + token) + '\n')

    def test_spur_keeps_native_task_and_allocation_with_append_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / 'srun'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps({"args":sys.argv[1:],"exported":os.environ.get("EXAMPLE")}))\n')
            fake.chmod(0o755)
            log = root / 'log'
            log.write_text('previous\n')
            native = ['bash', '-c', "export PRECISION=fp4 && exec vllm serve model --tp 8"]
            subprocess.run([str(ROOT / 'spur/srun'), '--jobid', '123', '--overlap',
                '--no-container-entrypoint', '--no-container-mount-home', '--container-writable',
                '--open-mode', 'append', '--output', str(log), '--export=ALL,EXAMPLE=value',
                '--nodes', '1', '--container-image', 'registry-1.docker.io#repo/image:sha256:abc', *native], env=dict(os.environ, SA_REAL_SRUN=str(fake)), check=True)
            lines = log.read_text().splitlines()
            self.assertEqual(lines[0], 'previous')
            result = json.loads(lines[1])
            self.assertEqual(result['args'], ['--jobid', '123', '--overlap', '--nodes', '1', '--container-image', 'registry-1.docker.io/repo/image@sha256:abc', *native])
            self.assertEqual(result['exported'], 'value')


if __name__ == '__main__':
    unittest.main()
