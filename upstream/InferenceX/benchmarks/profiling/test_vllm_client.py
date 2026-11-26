"""Verify native client timestamp hooks around actual request iteration."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vllm_client import measured_requests


class NativeClientTests(unittest.TestCase):
    def test_warmup_excluded_and_requests_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            phase = Path(directory) / "measurement.json"
            calls = []

            async def external_request(value):
                calls.append((value, phase.exists()))
                return value * 2

            async def external_iterator(values):
                for value in values:
                    yield value

            module = SimpleNamespace(
                get_request=external_iterator,
                ASYNC_REQUEST_FUNCS={"fixture": external_request},
            )

            async def run():
                with measured_requests(module, phase):
                    await module.ASYNC_REQUEST_FUNCS["fixture"](0)
                    outputs = []
                    async for value in module.get_request([1, 2, 3]):
                        outputs.append(
                            await module.ASYNC_REQUEST_FUNCS["fixture"](value)
                        )
                return outputs

            self.assertEqual(asyncio.run(run()), [2, 4, 6])
            self.assertEqual(calls, [(0, False), (1, True), (2, True), (3, True)])
            interval = json.loads(phase.read_text())
            self.assertGreaterEqual(interval["end_unix"], interval["start_unix"])
            self.assertIs(module.get_request, external_iterator)
            self.assertIs(module.ASYNC_REQUEST_FUNCS["fixture"], external_request)
