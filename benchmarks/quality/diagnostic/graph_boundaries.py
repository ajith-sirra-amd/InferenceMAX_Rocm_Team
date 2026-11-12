"""Bounded per-row observations recorded into the native DSpark CUDA graph.

Integration: install once before model construction. In the existing outer
proposal observer, call ticket = begin_proposal(speculator, bound.arguments)
before the original call and end_proposal(speculator, ticket) afterward.
Only end_proposal reads device data on the CPU. Native outputs are unchanged.
"""
from __future__ import annotations

import functools
import inspect
import os

import torch


MAX_GRAPH_REQUESTS = int(os.environ.get('KIMI_GRAPH_BOUNDARY_MAX_REQS', '2'))
INSTALLED = False
_ACTIVE = None
_GRAPHS = {}
_LABELS = {}


def _summary(value):
    rows = value.detach().reshape(-1, value.shape[-1])
    return torch.stack([
        torch.isfinite(rows).sum(-1), torch.isnan(rows).sum(-1),
        torch.isposinf(rows).sum(-1), torch.isneginf(rows).sum(-1),
    ], dim=-1)


def _record(name, value, domain='query'):
    if (_ACTIVE is None or not _ACTIVE.get('enabled', True)
            or not isinstance(value, torch.Tensor)):
        return
    if not value.is_floating_point() or value.ndim < 1:
        return
    occurrence = _ACTIVE['counts'].get(name, 0)
    _ACTIVE['counts'][name] = occurrence + 1
    _ACTIVE['records'].append(dict(
        name=name, occurrence=occurrence, domain=domain,
        row_offset=_ACTIVE.get('context_offset', 0) if domain == 'context' else 0,
        shape=list(value.shape), dtype=str(value.dtype),
        counts=_summary(value)))


def _bucket(mode):
    return dict(mode=mode, records=[], counts={}, context_offset=0)


def begin_proposal(speculator, arguments):
    """Observe target inputs outside capture; return an opaque proposal ticket."""
    global _ACTIVE
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('begin_proposal must run outside CUDA capture')
    if _ACTIVE is not None:
        raise RuntimeError('Nested graph-boundary proposal observation')
    batch = arguments['input_batch']
    ticket = _bucket('eager_before_graph')
    ticket.update(num_reqs=batch.num_reqs, enabled=batch.num_reqs <= MAX_GRAPH_REQUESTS,
                  query_per_req=speculator.num_query_per_req,
                  target_live=(~batch.is_padding[:batch.num_tokens]).clone(),
                  num_target_rows=batch.num_tokens,
                  graph_key=None)
    speculator._graph_boundary_ticket = ticket
    _ACTIVE = ticket
    for index, hidden in enumerate(arguments.get('aux_hidden_states') or []):
        _record(f'target.aux.{index}', hidden[:batch.num_tokens], 'target')
    return ticket


def _serialize(bucket, ticket):
    target_live = ticket['target_live'].cpu().tolist()
    result = []
    for record in bucket['records']:
        counts = record['counts'].cpu().tolist()
        domain, offset = record['domain'], record['row_offset']
        if domain in ('target', 'context'):
            live = target_live[offset:offset + len(counts)]
            live += [False] * (len(counts) - len(live))
        else:
            limit = ticket['num_reqs']
            if domain == 'query':
                limit *= ticket['query_per_req']
            live = [row < limit for row in range(len(counts))]
        output = {k: v for k, v in record.items() if k != 'counts'}
        output.update(mode=bucket['mode'], row_is_live=live,
                      columns=['finite', 'nan', 'positive_inf', 'negative_inf'],
                      counts_per_row=counts,
                      live_nan=sum(row[1] for row, valid in zip(counts, live) if valid),
                      padded_nan=sum(row[1] for row, valid in zip(counts, live) if not valid))
        result.append(output)
    return result


def cancel_proposal(speculator):
    """Clear an interrupted outer observation without reading device data."""
    global _ACTIVE
    _ACTIVE = None
    speculator._graph_boundary_ticket = None


def end_proposal(speculator, ticket):
    """Read only the graph that this proposal actually replayed, after it ends."""
    cancel_proposal(speculator)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('end_proposal must run outside CUDA capture')
    key = ticket['graph_key']
    graph = _GRAPHS.get(key)
    records = _serialize(ticket, ticket)
    if graph is not None:
        records.extend(_serialize(graph, ticket))
    required = ['context.staged_aux', 'context.projection.output',
                'context.norm.output', 'context.kv_projection.output',
                'draft.final_norm.output', 'draft.base_logits']
    required += [f'draft.layer.{i}.{part}.output'
                 for i in range(5) for part in ('attention', 'mlp')]
    required += [f'draft.summed_logits.{i}' for i in range(6)]
    names = {r['name'] for r in records if r['mode'] == 'native_graph_replay'}
    missing = [name for name in required if name not in names] if graph is not None else []
    return dict(
        status=('incomplete' if missing else 'collected') if graph is not None else 'graph_not_instrumented',
        graph_num_reqs=key[1] if key is not None else None,
        live_num_reqs=ticket['num_reqs'], max_instrumented_graph_reqs=MAX_GRAPH_REQUESTS,
        missing_required_boundaries=missing, records=records,
        first_live_nan=next((r['name'] for r in records if r['live_nan']), None),
        limits='Device summaries replay in the native graph; no eager rerun. '
               'Only bounded small graphs instrumented; larger graphs have no inner coverage. '
               'Live rows follow request padding, not per-rank cache residency. '
               'Finiteness does not prove numerical correctness; observations alter timing.')


def _module_observer(module, name, domain='query', inputs=False):
    if inputs:
        def before(_module, args, kwargs):
            value = kwargs.get('hidden_states', args[0] if args else None)
            _record(name + '.input', value, domain)
        module.register_forward_pre_hook(before, with_kwargs=True)

    def after(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        _record(name + '.output', value, domain)
    module.register_forward_hook(after)


def install():
    """Attach observations before model construction; does not initialize GPUs."""
    global INSTALLED
    if INSTALLED:
        return
    if MAX_GRAPH_REQUESTS < 1:
        raise ValueError('KIMI_GRAPH_BOUNDARY_MAX_REQS must be positive')
    from vllm.models.kimi_k3.nvidia import dspark_mla
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

    initialize = dspark_mla.K3DSparkModel.__init__

    @functools.wraps(initialize)
    def model_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        _module_observer(self.context_proj, 'context.projection', 'context', True)
        _module_observer(self.context_norm, 'context.norm', 'context', True)
        _module_observer(self.context_kv_proj, 'context.kv_projection', 'context', True)
        for index, layer in enumerate(self.layers):
            prefix = f'draft.layer.{index}'
            _module_observer(layer.self_attn, prefix + '.attention', inputs=True)
            _module_observer(layer.mlp, prefix + '.mlp', inputs=True)
            _module_observer(layer.input_layernorm, prefix + '.input_norm', inputs=True)
            _LABELS[id(layer.input_layernorm)] = prefix + '.input_norm'
            _LABELS[id(layer.post_attention_layernorm)] = prefix + '.post_attention_norm'
        _LABELS[id(self.final_norm)] = 'draft.final_norm'
    dspark_mla.K3DSparkModel.__init__ = model_init

    fused_norm = dspark_mla.fused_allreduce_rms_norm

    @functools.wraps(fused_norm)
    def norm(hidden, residual, norm_layer, *args, **kwargs):
        name = _LABELS.get(id(norm_layer), 'draft.unlabelled_norm')
        _record(name + '.input', hidden)
        _record(name + '.residual_input', residual)
        result = fused_norm(hidden, residual, norm_layer, *args, **kwargs)
        _record(name + '.output', result[0])
        _record(name + '.residual_output', result[1])
        return result
    dspark_mla.fused_allreduce_rms_norm = norm

    for method, name, domain in (
        ('compute_draft_logits', 'draft.base_logits', 'query'),
        ('markov_embed', 'draft.markov_embedding', 'request'),
        ('markov_bias', 'draft.markov_bias', 'request'),
    ):
        original = getattr(dspark_mla.K3DSparkForCausalLM, method)
        def wrap(function, label, rows):
            @functools.wraps(function)
            def observed(self, *args, **kwargs):
                result = function(self, *args, **kwargs)
                _record(label, result, rows)
                return result
            return observed
        setattr(dspark_mla.K3DSparkForCausalLM, method, wrap(original, name, domain))

    sample = DSparkSpeculator._sample_logits

    @functools.wraps(sample)
    def sample_logits(self, logits, idx_map, sample_pos, step):
        _record(f'draft.summed_logits.{step}', logits, 'request')
        return sample(self, logits, idx_map, sample_pos, step)
    DSparkSpeculator._sample_logits = sample_logits

    prepare = DFlashSpeculator._prepare_graph_forward

    @functools.wraps(prepare)
    def prepare_graph(self, num_reqs, attn_metadata):
        global _ACTIVE
        if torch.cuda.is_current_stream_capturing():
            _ACTIVE = None
            if num_reqs <= MAX_GRAPH_REQUESTS:
                _ACTIVE = _bucket('native_graph_replay')
                _GRAPHS[(id(self.query_cudagraph_manager), num_reqs)] = _ACTIVE
                count = self._num_graph_context_tokens(num_reqs)
                if self._aux_staging is not None:
                    _record('context.staged_aux', self._aux_staging[:count], 'context')
        return prepare(self, num_reqs, attn_metadata)
    DFlashSpeculator._prepare_graph_forward = prepare_graph

    generate = DSparkSpeculator._generate_draft

    @functools.wraps(generate)
    def generate_draft(self, *args, **kwargs):
        global _ACTIVE
        capturing = torch.cuda.is_current_stream_capturing()
        try:
            return generate(self, *args, **kwargs)
        finally:
            if capturing:
                _ACTIVE = None
    DSparkSpeculator._generate_draft = generate_draft

    store = DFlashSpeculator._store_hidden_states
    store_signature = inspect.signature(store)

    @functools.wraps(store)
    def store_hidden(self, *args, **kwargs):
        bound = store_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        bucket = _ACTIVE
        if bucket is not None:
            bucket['context_offset'] = (
                min(bound.arguments['num_tokens'], bound.arguments['num_graph_tokens'])
                if self._aux_staging is not None else 0)
        try:
            return store(self, *args, **kwargs)
        finally:
            if bucket is not None:
                bucket['context_offset'] = 0
    DFlashSpeculator._store_hidden_states = store_hidden

    precompute = DFlashSpeculator._precompute_context_kv

    @functools.wraps(precompute)
    def precompute_context(self, start, end, *args, **kwargs):
        bucket = _ACTIVE
        previous = bucket['context_offset'] if bucket is not None else 0
        if bucket is not None:
            bucket['context_offset'] = start
        try:
            return precompute(self, start, end, *args, **kwargs)
        finally:
            if bucket is not None:
                bucket['context_offset'] = previous
    DFlashSpeculator._precompute_context_kv = precompute_context

    replay = DFlashCudaGraphManager.run_fullgraph

    @functools.wraps(replay)
    def run_graph(self, desc):
        if _ACTIVE is not None and _ACTIVE['mode'] == 'eager_before_graph':
            _ACTIVE['graph_key'] = (id(self), desc.num_reqs)
        return replay(self, desc)
    DFlashCudaGraphManager.run_fullgraph = run_graph
    INSTALLED = True
