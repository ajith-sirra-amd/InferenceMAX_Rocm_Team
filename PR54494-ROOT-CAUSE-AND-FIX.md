# #54494 (DCP query replication): real root cause, and the real fix

**Written:** 2026-10-09. Status: fix committed (`aa652bbb`), applies cleanly
and compiles against the exact pinned vLLM commit, dispatched for its first
real GPU test (`37946632603`, isolated conc=96, all other patches off,
against the 14,077 tok/s/GPU clean baseline). **Not yet GPU-verified** —
this doc will be updated once that result lands.

---

## The simple version

`#54494` is an upstream vLLM PR that lets each GPU skip an expensive
cross-GPU "sync up" step during decode, by giving each GPU its own full
copy of a small piece of data instead of asking the other GPUs for it every
single step. That sync-up step is one of our biggest measured costs
(21.3% of total wall-clock), so removing it is a real, meaningful win if it
works.

Our first attempt at adapting this PR to our setup had a real bug: it took
a weight tensor (`W_K`) that's used by **every** decode step, and
overwrote it with a bigger, combined version meant only for the **new**
optimized steps. Every other decode step then got handed the wrong-sized
weight. That mismatch froze the server for about 40 minutes before a
10-minute network timeout finally killed it — a crash, not graceful
degradation.

The fix: instead of overwriting the original weight, keep it untouched and
create a **second, separate** copy for the new optimized path, then switch
between the two depending on which kind of step is running. This is
exactly what the upstream PR's own authors already did correctly for a
different model configuration — our case (FP4-quantized models) was the
one configuration they explicitly left unfinished, with a comment saying
so. We finished it, using their own working pattern as the template rather
than guessing.

---

## The real root cause

### What our first patch did (broken)

In `vllm/model_executor/layers/attention/mla_attention.py`, inside
`process_weights_after_loading` (the one-time weight-setup code that runs
when the model loads), the FP4-quantized MLA attention path builds a
weight called `W_K` (used to project queries during decode). Our first
patch added this, inside the FP4-specific setup branch:

```python
if self.dcp_q_replicate:
    replace_parameter(
        self,
        "W_K",
        get_dcp_group().all_gather(self.W_K.contiguous(), dim=0),
        prefer_copy=True,
    )
```

`replace_parameter(self, "W_K", ...)` **overwrites** `self.W_K` in place.
After this runs, `self.W_K` permanently holds the bigger, all-gathered
(combined-across-GPUs) version — there is no separate copy of the original
left anywhere.

### Why that's wrong

`self.W_K` isn't only used by the new query-replicated decode path. It's
read unconditionally by the decode projection code, every single decode
step, regardless of whether that step is using query replication:

```python
if self.is_aiter_triton_fp4_bmm_enabled:
    mqa_ql_nope = rocm_aiter_ops.batched_gemm_a16wfp4(
        mqa_q_nope,
        self.W_K,       # <-- always this rank's W_K, no branch on replication state
        self.W_K_scale,
        ...
    )
```

Once `self.W_K` is globally overwritten with the bigger, combined shape,
every decode call — replicated or not — feeds a shape-mismatched
computation into this kernel. That's a correctness bug waiting to
manifest, and distributed collective code tends to manifest shape bugs as
hangs rather than clean crashes: one GPU's tensor shapes and schedule stop
agreeing with the others', some GPUs wait on a collective operation that
others never issue the same way, and the whole group eventually times out
together.

### What the logs actually showed

The first real test (conc=96, isolated, run `37930016104`) ran cleanly for
about 40 minutes, then the server's own periodic status logging (which
normally prints every ~10 seconds) simply stopped — while the HTTP server
itself kept responding to unrelated health-check traffic (so the process
hadn't crashed outright, just the request-processing loop had stalled).
Exactly 10 minutes later — the default PyTorch NCCL collective timeout —
all 8 GPUs' internal watchdogs fired simultaneously:

```
[Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=87505,
OpType=_ALLGATHER_BASE, ...) ran for 600014 milliseconds before timing out.
terminate called after throwing an instance of 'c10::DistBackendError'
```

This matches the shape-mismatch theory: a collective operation (`_ALLGATHER_BASE`)
that some ranks called and others didn't (or called with different
expectations), stalling forever until the watchdog gave up.

### Why upstream had already flagged this

The same file has this guard, which our first patch's hand-adaptation
**deleted** in order to make the patch "work" on our FP4 model:

```python
if self.dcp_q_replicate:
    ...
    if (
        self.is_aiter_triton_fp4_bmm_enabled
        or self.is_aiter_triton_fp8_bmm_enabled
    ):
        raise NotImplementedError(
            "DCP query replication is not implemented for the aiter "
            "FP4/FP8 MLA BMM paths."
        )
```

This isn't a defensive placeholder — it's the upstream author telling us,
explicitly, that this exact configuration was never implemented. Deleting
the guard didn't implement the missing feature; it just let us run code
that was never finished for our configuration.

---

## The real fix

The same file already has a **working** implementation of this exact idea
for a different (non-FP4-BMM) code path. It never overwrites the original
weight — it creates a new, separate parameter for the replicated case:

```python
# working pattern, already in upstream, for the non-BMM path
replace_parameter(self, "W_UK_T", W_UK.permute(1, 2, 0), prefer_copy=True)
if self.dcp_q_replicate:
    replace_parameter(
        self,
        "W_UK_T_dcp_qrep",          # <-- a NEW, separate parameter
        get_dcp_group().all_gather(self.W_UK_T.contiguous(), dim=0),
        prefer_copy=True,
    )
```

And at its decode call site, it branches between the two by a flag
(`qrep_decode`) that's already set earlier in the same function depending
on whether this particular step is using the replicated path:

```python
W_UK_T = self.W_UK_T_dcp_qrep if qrep_decode else self.W_UK_T
```

Our fix mirrors this exactly for the FP4/FP8 BMM paths, which never had
it:

**1. Weight setup** — keep the original, add a separate replicated copy:
```python
if self.dcp_q_replicate:
    replace_parameter(
        self, "W_K_dcp_qrep",
        get_dcp_group().all_gather(self.W_K.contiguous(), dim=0),
        prefer_copy=True,
    )
    replace_parameter(
        self, "W_K_scale_dcp_qrep",
        get_dcp_group().all_gather(self.W_K_scale.contiguous(), dim=0),
        prefer_copy=True,
    )
```
(`self.W_K` / `self.W_K_scale` themselves are never touched.)

**2. Decode call site** — branch on `qrep_decode`, same pattern as the
proven working case:
```python
bmm_W_K = self.W_K_dcp_qrep if qrep_decode else self.W_K
bmm_W_K_scale = self.W_K_scale_dcp_qrep if qrep_decode else self.W_K_scale
mqa_ql_nope = rocm_aiter_ops.batched_gemm_a16wfp4(
    mqa_q_nope, bmm_W_K, bmm_W_K_scale, ...
)
```

**3. The `NotImplementedError` guard stays removed** — but now it's
actually safe, because the feature is implemented, not bypassed.

One deliberate scope decision: only `W_K`/`W_K_scale` get a replicated
copy, not `W_V`/`W_V_scale`. This mirrors the working non-BMM path exactly
— it never creates a `W_UV`-replicated variant either, which means the
V-side (output) projection genuinely doesn't need one; only the K-side
query-projection weight does.

---

## How this was verified (and what's still open)

Verified so far, without using any GPU (all of this is plain text/Python,
done on the sandbox node per the standing no-GPU-on-this-node rule):

1. **Fetched the exact pinned vLLM commit** (`81198e97ba7eee2a22540caaa756b7fdddcb4d93`
   — what our current image actually runs) directly from GitHub, not a
   remembered or assumed version. Confirmed line-by-line that the bug
   theory matches the real installed code.
2. **Applied the patch for real** with `patch -p1` against that exact
   source tree (not just checked hunk-header arithmetic, which caught two
   earlier bugs in this same patch file but doesn't prove the *logic* is
   right).
3. **`py_compile`'d all three changed files** — no syntax errors.
4. Manually re-read the patched output to confirm indentation and logic
   landed exactly as intended.

**Not yet verified:** whether this is actually correct under real
distributed execution — real shapes, real collectives across 8 GPUs, real
CUDA graphs. That can only be confirmed by an actual run, which is in
flight now (`37946632603`). GSM8K accuracy gate still applies before
trusting any throughput number from it, same as every other
numerics-affecting change this campaign.

## Next steps

1. Pull the result of `37946632603` — does the server survive serving
   traffic, or does it hit a new failure? If it survives, what's the real
   throughput number vs. the 14,077 baseline?
2. If clean, GSM8K-gate it (0.995 threshold, standing rule).
3. Only then consider stacking it with the other known-working patches
   (`#59069`/`#59070`/`#59693`/`#59966`/`#54625`) — not before, so any
   regression or win is attributable to this change alone.
