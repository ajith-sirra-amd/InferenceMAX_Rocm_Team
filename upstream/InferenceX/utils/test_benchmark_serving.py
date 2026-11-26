"""Exercise client result persistence without loading a model or serving requests."""

import asyncio
import json
from argparse import Namespace
from unittest.mock import AsyncMock

import pytest
from infx.bench_serving import benchmark_serving as client


class Tokenizer:
    vocab_size = 128

    def encode(self, text, **kwargs):
        return [ord(character) - 256 for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(256 + int(token)) for token in tokens)

    def __call__(self, text, **kwargs):
        return Namespace(input_ids=self.encode(text))


@pytest.mark.parametrize("enabled", [False, True])
def test_measurement_marker_excludes_warmup(tmp_path, monkeypatch, enabled):
    marker = tmp_path / "phase.json"
    if enabled:
        monkeypatch.setenv("INFERENCEX_PROFILE_PHASE_FILE", str(marker))
    else:
        monkeypatch.delenv("INFERENCEX_PROFILE_PHASE_FILE", raising=False)
    observations = []

    async def request(**kwargs):
        observations.append(marker.exists())
        return client.RequestFuncOutput(
            success=True,
            generated_text="ok",
            latency=0.03,
            ttft=0.01,
            itl=[0.02],
            output_tokens=2,
            prompt_len=3,
        )

    monkeypatch.setitem(client.ASYNC_REQUEST_FUNCS, "fixture", request)
    result = asyncio.run(
        client.benchmark(
            backend="fixture",
            api_url="http://unused",
            base_url="http://unused",
            model_id="fixture",
            model_name="fixture",
            tokenizer=Tokenizer(),
            input_requests=[
                ("one", 3, 2, None),
                ("two", 3, 2, None),
                ("six", 3, 2, None),
            ],
            logprobs=None,
            best_of=1,
            request_rate=float("inf"),
            burstiness=1,
            disable_tqdm=True,
            num_warmups=2,
            profile=False,
            selected_percentile_metrics=["ttft"],
            selected_percentiles=[99.0],
            ignore_eos=True,
            goodput_config_dict={},
            max_concurrency=2,
            lora_modules=None,
        )
    )
    assert observations == [False, False, enabled, enabled, enabled]
    assert result["completed"] == 3
    if enabled:
        assert (
            json.loads(marker.read_text())["start_unix"]
            == result["benchmark_start_time_unix"]
        )
        assert not (tmp_path / "phase.json.tmp").exists()


def test_random_prefix_is_shared_and_added_to_input_length():
    requests = client.sample_random_requests(
        prefix_len=6,
        input_len=8,
        output_len=4,
        num_prompts=3,
        range_ratio=1,
        tokenizer=Tokenizer(),
        num_workers=1,
    )
    assert len(requests) == 3
    assert len({prompt[:6] for prompt, *_ in requests}) == 1
    assert all(
        len(prompt) == input_len == 14 and output_len == 4
        for prompt, input_len, output_len, *_ in requests
    )


@pytest.mark.parametrize(
    "requested,completed,status",
    [
        (100, 100, "passed"),
        (100, 95, "passed"),
        (100, 94, "failed"),
        (100, 0, "failed"),
        (0, 0, "failed"),
        (100, 101, "failed"),
        (100, -1, "failed"),
        (100, None, "failed"),
    ],
)
@pytest.mark.parametrize("save_result", [True, False])
def test_client_preserves_outcome_before_failure(
    tmp_path,
    monkeypatch,
    requested,
    completed,
    status,
    save_result,
):
    args = Namespace(
        seed=0,
        backend="vllm",
        model="fixture",
        served_model_name=None,
        tokenizer=None,
        tokenizer_mode="auto",
        base_url="http://unused",
        endpoint="/completions",
        trust_remote_code=False,
        dataset_name="random",
        random_prefix_len=0,
        random_input_len=1,
        random_output_len=1,
        num_prompts=requested,
        random_range_ratio=1,
        use_chat_template=False,
        dsv4=False,
        random_num_workers=1,
        goodput=None,
        logprobs=None,
        best_of=1,
        request_rate=float("inf"),
        burstiness=1,
        disable_tqdm=True,
        num_warmups=0,
        profile=False,
        percentile_metrics="ttft",
        metric_percentiles="99",
        ignore_eos=False,
        max_concurrency=1,
        lora_modules=None,
        save_result=save_result,
        metadata=None,
        save_detailed=True,
        result_filename="result.json",
        result_dir=str(tmp_path),
    )
    metrics = {
        name: 0
        for name in (
            "median_ttft_ms",
            "mean_ttft_ms",
            "std_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "std_tpot_ms",
            "p99_tpot_ms",
            "median_itl_ms",
            "mean_itl_ms",
            "std_itl_ms",
            "p99_itl_ms",
        )
    }
    raw = {**metrics, "completed": completed, "errors": ["retained diagnostic"]}
    monkeypatch.setattr(client, "_load_tokenizer", lambda *a, **kw: object())
    monkeypatch.setattr(client, "sample_random_requests", lambda **kw: [])
    monkeypatch.setattr(client, "benchmark", AsyncMock(return_value=raw))
    # Avoid changing the test process's global GC state.
    monkeypatch.setattr(client.gc, "freeze", lambda: None)
    invalid = (
        requested == 0 or completed is None or completed < 0 or completed > requested
    )

    if status == "failed":
        message = "invalid request counts" if invalid else "request failure rate"
        with pytest.raises(SystemExit, match=message):
            client.main(args)
    else:
        client.main(args)

    path = tmp_path / "result.json"
    if not save_result:
        assert not path.exists()
        return
    saved = json.loads(path.read_text())
    assert saved["num_prompts"] == requested
    assert saved["completed"] == completed
    assert saved["errors"] == ["retained diagnostic"]
    outcome = saved["benchmark_outcome"]
    assert outcome["status"] == status
    assert outcome["requested"] == requested
    assert outcome["completed"] == completed
    if invalid:
        assert "0 <= completed <= requested > 0" in outcome["error"]
        assert "failed" not in outcome  # Invalid counts cannot define a failure rate.
    else:
        assert "error" not in outcome
        assert outcome["failed"] == {100: 0, 95: 5, 94: 6, 0: 100}[completed]
        assert outcome["max_failure_rate"] == 0.05
