#!/usr/bin/env python3
"""Exercise the actual native tokenizer and chat-template sampler before model load."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import traceback


def preflight(model, input_tokens, output_tokens):
    import numpy as np
    from infx.bench_serving.benchmark_serving import _load_tokenizer, sample_random_requests

    # The existing run_benchmark_serving helper uses the native CLI's seed 0.
    random.seed(0)
    np.random.seed(0)
    tokenizer = _load_tokenizer(model, tokenizer_mode='kimi_k3', trust_remote_code=True)
    requests = sample_random_requests(
        prefix_len=0, input_len=input_tokens, output_len=output_tokens, num_prompts=4,
        range_ratio=1.0, tokenizer=tokenizer, use_chat_template=True,
        tokenizer_id=model, tokenizer_mode='kimi_k3', trust_remote_code=True, num_workers=1,
    )
    records = [dict(input_tokens=length, output_tokens=output,
                    independently_encoded_tokens=len(tokenizer.encode(prompt, add_special_tokens=False)),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
               for prompt, length, output, _ in requests]
    passed = len(records) == 4 and all(
        row['input_tokens'] == row['independently_encoded_tokens'] == input_tokens
        and row['output_tokens'] == output_tokens for row in records)
    return dict(status='passed' if passed else 'failed', tokenizer_mode='kimi_k3',
                tokenizer_class=type(tokenizer).__name__, use_chat_template=True,
                random_range_ratio=1, random_seed=0, requested_input_tokens=input_tokens,
                requested_output_tokens=output_tokens, requests=records,
                scope='Native client tokenizer and sampler only; no model load or server request.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/model')
    parser.add_argument('--input-tokens', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=Path('/logs/client-preflight.json'))
    args = parser.parse_args()
    try:
        result = preflight(args.model, args.input_tokens, args.output_tokens)
    except Exception as error:
        result = dict(status='failed', error=type(error).__name__ + ': ' + str(error),
                      traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
