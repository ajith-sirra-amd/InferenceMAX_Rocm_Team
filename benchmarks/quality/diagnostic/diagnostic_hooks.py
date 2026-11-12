"""Observe two live calls per stage; no sampler, graph or model output changes."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path(os.environ.get("KIMI_DIAGNOSTIC_OUTPUT", "/logs/diagnostic"))
COUNTS = {}
INSTALLED = False


def scalar(value):
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return str(value)
    return value


def tensor_info(value, values=False):
    if value is None:
        return None
    value = value.detach()
    result = dict(shape=list(value.shape), dtype=str(value.dtype),
                  stride=list(value.stride()), data_ptr=value.data_ptr(),
                  numel=value.numel())
    if value.numel():
        result.update(min=scalar(value.min().item()), max=scalar(value.max().item()))
        if value.is_floating_point():
            result.update(finite=int(torch.isfinite(value).sum().item()),
                          nan=int(torch.isnan(value).sum().item()),
                          positive_inf=int(torch.isposinf(value).sum().item()),
                          negative_inf=int(torch.isneginf(value).sum().item()))
            if value.ndim >= 2:
                rows = value.reshape(-1, value.shape[-1])
                result['finite_count_per_row'] = torch.isfinite(rows).sum(-1).cpu().tolist()[:1024]
                result['positive_finite_count_per_row'] = (torch.isfinite(rows) & (rows > 0)).sum(-1).cpu().tolist()[:1024]
                result['all_masked_rows'] = int(torch.isneginf(rows).all(-1).sum().item())
        if values:
            result['values'] = [scalar(v) for v in value.reshape(-1)[:1024].cpu().tolist()]
            result['values_truncated'] = value.numel() > 1024
    return result


def rank():
    from vllm.distributed.parallel_state import get_tp_group
    return get_tp_group().rank_in_group


def begin(kind):
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        stage = (ROOT / 'stage.txt').read_text().strip()
    except FileNotFoundError:
        return None
    if stage not in ('c1-greedy', 'c1-sampled', 'c48-sampled'):
        return None
    key = (stage, kind)
    number = COUNTS.get(key, 0)
    if number >= 2:
        return None
    COUNTS[key] = number + 1
    return dict(stage=stage, kind=kind, call=number + 1, rank=rank(),
                pid=os.getpid(), unix_time=time.time())


def write(record):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"rank-{record['rank']}-{record['stage']}-{record['kind']}-{record['call']}.json"
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def cache_info(cache, idx, draft_ids=None):
    if cache is None:
        return None
    result = dict(storage=tensor_info(cache.values[:0]),
                  row_mapping=tensor_info(cache.row_mapping, True))
    rows = idx.long() if cache.row_mapping is None else cache.row_mapping[idx.long()].long()
    owned = rows >= 0
    result['request_rows'] = tensor_info(rows, True)
    if cache.row_mapping is not None:
        result['inverse_row_mapping'] = tensor_info(cache.inverse_row_mapping, True)
    # Only local owned rows are valid. Save distributions' summaries, not logits.
    local = cache.values[rows[owned]]
    result['owned_logits'] = tensor_info(local)
    if draft_ids is not None and owned.any().item():
        ids = draft_ids[owned].long()
        valid = (ids >= 0) & (ids < local.shape[-1])
        selected = local.gather(-1, ids.clamp(0, local.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
        result['selected_token_logits'] = tensor_info(selected, True)
        result['selected_ids_valid'] = bool(valid.all().item())
    return result


def install():
    global INSTALLED
    if INSTALLED:
        return
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.batch_shard import BatchSharder
    from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
    originals = dict(propose=DFlashSpeculator.propose, verify=RejectionSampler._verify,
                     sample=GPUModelRunner.sample, shard=BatchSharder.shard_sampler_inputs)
    signatures = {k: inspect.signature(v) for k, v in originals.items()}

    @functools.wraps(originals['propose'])
    def propose(self, *args, **kwargs):
        bound = signatures['propose'].bind(self, *args, **kwargs)
        if bound.arguments.get('dummy_run', False):
            return originals['propose'](self, *args, **kwargs)
        record = begin('proposal')
        output = originals['propose'](self, *args, **kwargs)
        if record is not None:
            try:
                batch = bound.arguments['input_batch']
                nr = batch.num_reqs
                record.update(model_class=f'{type(self.model).__module__}.{type(self.model).__qualname__}',
                              model_sample_sequential=hasattr(self.model, 'sample_sequential'),
                              use_draft_logits=self.use_draft_logits,
                              graph_count=len(self.query_cudagraph_manager.graphs),
                              last_graph_replay=getattr(self.query_cudagraph_manager, '_diagnostic_replay', None),
                              request_ids=batch.req_ids,
                              num_reqs=nr, input_token_ids=tensor_info(batch.input_ids, True),
                              input_positions=tensor_info(batch.positions, True),
                              idx_mapping=tensor_info(batch.idx_mapping, True),
                              draft_token_ids=tensor_info(output, True),
                              sample_indices=tensor_info(self.sample_indices[:nr*self.num_speculative_steps], True),
                              sample_positions=tensor_info(self.sample_pos[:nr*self.num_speculative_steps], True),
                              sample_idx_mapping=tensor_info(self.sample_idx_mapping[:nr*self.num_speculative_steps], True),
                              temperature=tensor_info(self.temperature[batch.idx_mapping.long()], True),
                              draft_cache=cache_info(self.draft_logits, batch.idx_mapping, output))
            except Exception as error:
                record['diagnostic_error'] = f'{type(error).__name__}: {error}'
            write(record)
        return output

    @functools.wraps(originals['verify'])
    def verify(self, *args, **kwargs):
        bound = signatures['verify'].bind(self, *args, **kwargs).arguments
        record = begin('verification')
        if record is not None:
            try:
                record.update(block_verification=self.use_block_verification,
                              synthetic_rates=tensor_info(self.synthetic_conditional_rates, True),
                              raw_target_logits=tensor_info(bound['logits']),
                              gathered_draft_token_ids=tensor_info(bound['draft_sampled'], True),
                              verified_draft_token_ids=tensor_info(bound['verify_draft_sampled'], True),
                              positions=tensor_info(bound['pos'], True),
                              idx_mapping=tensor_info(bound['idx_mapping'], True),
                              cu_num_logits=tensor_info(bound['cu_num_logits'], True),
                              draft_cache=cache_info(bound['draft_logits'], bound['idx_mapping']))
            except Exception as error:
                record['diagnostic_error'] = f'{type(error).__name__}: {error}'
        result = originals['verify'](self, *args, **kwargs)
        if record is not None:
            try:
                processed, sampled, num_sampled = result
                record.update(processed_target_logits=tensor_info(processed),
                              sampled_token_ids=tensor_info(sampled, True),
                              num_sampled=tensor_info(num_sampled, True))
            except Exception as error:
                record['diagnostic_error'] = f'{type(error).__name__}: {error}'
            write(record)
        return result

    DFlashSpeculator.propose = propose
    RejectionSampler._verify = verify

    @functools.wraps(originals['sample'])
    def sample(self, *args, **kwargs):
        record = begin('sampler')
        output = originals['sample'](self, *args, **kwargs)
        if record is not None:
            try:
                batch = signatures['sample'].bind(self, *args, **kwargs).arguments['input_batch']
                custom = self.model_state.custom_sampler
                record.update(batch_sharder=type(self.batch_sharder).__qualname__ if self.batch_sharder is not None else None,
                              sampler_class=f'{type(self.sampler).__module__}.{type(self.sampler).__qualname__}',
                              rejection_sampler_class=f'{type(self.rejection_sampler).__module__}.{type(self.rejection_sampler).__qualname__}',
                              custom_sampler_method=custom.__func__.__qualname__,
                              model_state_class=f'{type(self.model_state).__module__}.{type(self.model_state).__qualname__}',
                              request_ids=batch.req_ids,
                              global_idx_mapping=tensor_info(batch.idx_mapping, True),
                              global_logits_indices=tensor_info(batch.logits_indices, True),
                              sampled_token_ids=tensor_info(output[0].sampled_token_ids, True),
                              num_sampled=tensor_info(output[1], True), num_rejected=tensor_info(output[2], True))
            except Exception as error:
                record['diagnostic_error'] = f'{type(error).__name__}: {error}'
            write(record)
        return output

    @functools.wraps(originals['shard'])
    def shard(self, *args, **kwargs):
        record = begin('ownership')
        output = originals['shard'](self, *args, **kwargs)
        if record is not None:
            try:
                local, sorted_indices, _, metadata = output
                record.update(tp_rank=self.tp_rank, tp_size=self.tp_size,
                              request_ids=local.req_ids,
                              local_idx_mapping=tensor_info(local.idx_mapping, True),
                              local_cu_num_logits=tensor_info(local.cu_num_logits, True),
                              local_logits_indices=tensor_info(local.logits_indices, True),
                              sorted_logits_indices=tensor_info(sorted_indices, True),
                              gathered_src_indices=tensor_info(metadata.gathered_src_indices, True),
                              num_logits_per_rank=metadata.num_logits_per_rank,
                              num_local_reqs=metadata.num_local_reqs)
            except Exception as error:
                record['diagnostic_error'] = f'{type(error).__name__}: {error}'
            write(record)
        return output

    GPUModelRunner.sample = sample
    BatchSharder.shard_sampler_inputs = shard
    replay_original = DFlashCudaGraphManager.run_fullgraph

    @functools.wraps(replay_original)
    def replay(self, desc):
        result = replay_original(self, desc)
        # Python metadata only; record the real selected graph without re-running it.
        record = begin('draft-graph')
        if record is not None:
            record.update(mode=str(desc.cg_mode), num_tokens=desc.num_tokens,
                          num_reqs=desc.num_reqs)
            self._diagnostic_replay = record
            write(record)
        return result

    DFlashCudaGraphManager.run_fullgraph = replay
    # Attest the actual imported modules before any worker data capture.
    sources = {}
    for cls in (DFlashSpeculator, DSparkSpeculator, RejectionSampler, GPUModelRunner, BatchSharder):
        path = Path(inspect.getfile(cls))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    root = Path(inspect.getfile(DSparkSpeculator)).parents[5]
    for relative in ('v1/worker/gpu/spec_decode/speculator.py',
                     'v1/worker/gpu/spec_decode/draft_logits.py',
                     'v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
                     'v1/worker/gpu/spec_decode/dflash/cudagraph.py',
                     'v1/worker/gpu/sample/gumbel.py',
                     'models/kimi_k3/nvidia/dspark_mla.py'):
        path = root / relative
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = json.loads(Path(__file__).with_name('source-pins.json').read_text())
    observed = {str(Path(path).relative_to(root)): digest for path, digest in sources.items()}
    if observed != expected:
        raise RuntimeError('Diagnostic imported source hashes differ from pinned image source')
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f'hooks-{os.getpid()}.json').write_text(json.dumps(dict(
        pid=os.getpid(), source_sha256=sources,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mode='observe first two live calls per scenario; original outputs returned; no eager replay',
        hooks=['DFlashSpeculator.propose', 'RejectionSampler._verify',
               'GPUModelRunner.sample', 'BatchSharder.shard_sampler_inputs',
               'DFlashCudaGraphManager.run_fullgraph'],
    ), indent=2) + '\n')
    INSTALLED = True
