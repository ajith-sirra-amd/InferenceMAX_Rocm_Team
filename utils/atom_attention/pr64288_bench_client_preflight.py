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
    # Two concrete samples per workload shape exercise the same dataset path.
    args.num_prompts = 2
    samples = get_samples(args, tokenizer)
    if len(samples) != 2 or any(not s.prompt or s.prompt_len <= 0 for s in samples):
        raise RuntimeError("Benchmark dataset did not produce usable prompts")
    report.append({"case": case[0], "sample_lengths": [s.prompt_len for s in samples],
                   "tokenizer": type(tokenizer).__name__})
Path(os.environ["REVIEW_EVIDENCE"], "benchmark-client-preflight.json").write_text(json.dumps(report, indent=2))
print("BENCHMARK_CLIENT_PREFLIGHT", json.dumps(report), flush=True)
