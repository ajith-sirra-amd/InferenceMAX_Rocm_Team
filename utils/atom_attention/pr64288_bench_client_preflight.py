"""Exercise real benchmark argument parsing, tokenizer and dataset generation."""
import json
import os
from pathlib import Path

import pr64288_matched_serve as serving
from vllm.benchmarks.serve import add_cli_args
from vllm.benchmarks.datasets import get_samples
from vllm.tokenizers import get_tokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
add_cli_args(parser)
report = []
tokenizer = None
for case in serving.CASES:
    command = serving.bench_command("preflight", case)
    args = parser.parse_args(list(map(str, command[3:])))
    if tokenizer is None:
        tokenizer = get_tokenizer(
            args.tokenizer, tokenizer_mode=args.tokenizer_mode,
            trust_remote_code=args.trust_remote_code)
    # The short-query check uses its complete request set: random-prefix
    # generation also depends on the number of sampled requests.
    args.num_prompts = case[4] if case[0] == "short_query_c8" else 2
    samples = get_samples(args, tokenizer)
    if len(samples) != args.num_prompts or any(not s.prompt or s.prompt_len <= 0 for s in samples):
        raise RuntimeError("Benchmark dataset did not produce usable prompts")
    entry = {"case": case[0], "sample_lengths": [s.prompt_len for s in samples],
             "tokenizer": type(tokenizer).__name__}
    if case[0] == "short_query_c8":
        encoded = [tokenizer.encode(sample.prompt, add_special_tokens=False) for sample in samples]
        common = 0
        for tokens in zip(*encoded):
            if len(set(tokens)) != 1:
                break
            common += 1
        alignment = 12288
        cached = common // alignment * alignment
        remaining = [len(tokens) - cached for tokens in encoded]
        if cached < 8192 or not all(0 < length < 512 for length in remaining):
            raise RuntimeError("Generated short-query prompts cannot satisfy the actual cache alignment")
        entry.update(common_prefix_tokens=common, cache_alignment=alignment,
                     aligned_cached_tokens=cached, expected_query_tokens=remaining)
    report.append(entry)
Path(os.environ["REVIEW_EVIDENCE"], "benchmark-client-preflight.json").write_text(json.dumps(report, indent=2))
print("BENCHMARK_CLIENT_PREFLIGHT", json.dumps(report), flush=True)
