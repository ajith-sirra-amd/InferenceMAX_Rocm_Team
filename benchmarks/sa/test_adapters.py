"""Exercise transport adapters without a scheduler or network."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).parent


class AdapterTests(unittest.TestCase):
    def test_profile_hook_keeps_server_arguments_and_bounds_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / 'rocprofv3'
            fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            native = ['serve', '/model', '--speculative-config', '{"method": "dspark"}']
            subprocess.run([str(ROOT / 'profile/vllm'), *native], check=True, env=dict(os.environ,
                PATH=str(root) + ':' + os.environ['PATH'], CAPTURE=str(root / 'args'),
                SA_PROFILE_OUTPUT=str(root), SA_PROFILE_DELAY_SECONDS='1200', SA_PROFILE_SECONDS='60'))
            args = json.loads((root / 'args').read_text())
            self.assertEqual(args[args.index('--')+1:], ['vllm', *native])
            self.assertEqual(args[args.index('--collection-period')+1], '1200:60:1')
            self.assertIn('--kernel-trace', args)
            clocks = json.loads((root / 'clock-sync.json').read_text())
            self.assertGreater(clocks['boottime_ns'], 0)
            self.assertGreaterEqual(clocks['utc_after_ns'], clocks['utc_before_ns'])

    def test_docker_transport_preserves_task_mounts_and_cleans_only_owned_container(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / 'srun'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys,os\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            docker = root / 'docker'
            docker.write_text('#!/usr/bin/env python3\nimport json,sys,os\nfrom pathlib import Path\nPath(os.environ["CLEANUP"]).write_text(json.dumps(sys.argv[1:]))\n')
            docker.chmod(0o755)
            native = ['bash', '-c', "export PRECISION=fp4 && exec vllm serve /model --tp 8\n"]
            env = dict(os.environ, SA_REAL_SRUN=str(fake), SA_OUTPUT=str(root),
                       SLURM_JOB_ID='123',
                       CAPTURE=str(root / 'args'), CLEANUP=str(root / 'cleanup'),
                       PATH=str(root) + ':' + os.environ['PATH'])
            subprocess.run([str(ROOT / 'spur/srun'), '--quiet', '--jobid', '123', '--nodes=1',
                '--container-image=registry-1.docker.io#repo/image:sha256:abc',
                '--container-remap-root', '--container-writable', '--no-container-entrypoint',
                '--container-mounts', '/weights:/model,/data:/logs', *native], env=env, check=True)
            args = json.loads((root / 'args').read_text())
            self.assertEqual(args[:6], ['--jobid', '123', '--nodes=1', '--overlap', 'docker', 'run'])
            image_index = args.index('registry-1.docker.io/repo/image@sha256:abc')
            self.assertEqual(args[image_index-2:image_index], ['--entrypoint', native[0]])
            self.assertEqual(args[image_index+1:], native[1:])
            self.assertIn('type=bind,src=/weights,dst=/model,readonly', args)
            self.assertIn('type=bind,src=/data,dst=/logs', args)
            name = args[args.index('--name') + 1]
            self.assertEqual(json.loads((root / 'cleanup').read_text()), ['rm', '-f', name])

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



if __name__ == '__main__':
    unittest.main()
