# #54494 (DCP query replication): real root cause, and the real fix

**Written:** 2026-10-09. **Update (same day):** the fix below (`aa652bbb`)
was GPU-tested (`37946632603`) and **did not resolve the hang**. The exact
same `_ALLGATHER_BASE` collective timeout recurred, with the exact same
tensor shapes, at a similar point in serving. The fix itself was real and
necessary (confirmed by direct source reading, not guessed), but it was not
sufficient — there is a second, still-unidentified issue. See "Second
attempt: same failure, different understanding" at the bottom for the
current, honest state of this investigation. Treat everything below the
original fix description as superseded by that section.

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

## Second attempt: same failure, different understanding

Run `37946632603` (the fixed patch, isolated conc=96) ran for ~50 minutes,
then hit the **exact same symptom** as the original broken patch:

```
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=83881,
OpType=_ALLGATHER_BASE, NumelIn=4755456, NumelOut=38043648, Timeout(ms)=600000)
```

Identical op type, identical tensor sizes (`38043648 / 4755456 = 8.0`
exactly — an 8-way gather) to the first failure. The `W_K` shape-mismatch
bug fixed above was real and is still a correct fix for a real problem, but
it is clearly not the (or not the only) cause of this hang.

### Where this collective actually comes from

`vllm/v1/attention/ops/dcp.py`'s `MLADCPManager` has exactly one all-gather
matching this shape and purpose:

```python
def _gather_query(self, query: torch.Tensor) -> torch.Tensor:
    query = self.group.all_gather(query, dim=1)
```

This is wired up as `self.query_gather`, and the decode call site in
`mla_attention.py` only invokes it when **not** using the replicated path:

```python
if not qrep_decode:
    mqa_q = self.dcp_manager.query_gather(mqa_q)
```

This is the exact collective `#54494` exists to *remove*. Its shape (an
8-way gather, matching our DCP=8 config) lines up precisely with both
crashes. The implication: **`qrep_decode` is not `True` for every decode
step** while the feature is enabled — something causes some steps (or some
ranks' view of a step) to fall back to this old collective path, and when
that happens asymmetrically across the 8 ranks (some call it, some don't),
the group hangs until the watchdog fires 600 seconds later.

`qrep_decode` is set from `q_dcp_replicated is not None` — a parameter
threaded all the way from a custom op (`unified_mla_attention_with_output`)
whose actual caller lives in the Kimi-K3 model's own per-layer forward
code (not yet read — this is the next concrete lead, not a conclusion).
Something in that call path must be deciding, per-step or per-rank,
whether to pass the replicated query or fall back to `None` — and that
decision isn't staying consistent across all 8 ranks.

### Why guessing a third fix blind is the wrong move here

Two source-reading-based fixes have now both produced a plausible,
verified-correct-on-paper patch that still failed identically on real
hardware. That's a sign the remaining bug lives in a part of the system
that's hard to reason about from source alone — likely a race or a
data-dependent branch that diverges by rank. Continuing to patch from
static reading risks a third confident-but-wrong attempt, each costing a
real GPU dispatch and ~50 minutes before failing.

### Proposed next step (not yet started)

Build a fast, direct reproduction instead of a full agentic replay:

1. Start the server with the patch applied (same as today), but skip the
   full aiperf/agentic harness entirely.
2. Fire a small, fixed batch of direct `curl` requests against the running
   server — enough to exercise decode under load, but small enough to get
   a result in minutes, not ~50 minutes of warmup.
3. Add temporary debug logging around `qrep_decode` (per rank, per step) so
   the actual divergence — if the hang reproduces — is directly observable
   in `server.log`, rather than inferred from a cold NCCL timeout after the
   fact.

This turns "guess a fix from reading source" into "watch the actual
divergence happen," which is the right tool for a bug that's survived two
source-level fixes already. Not yet built — proposing this before spending
another full GPU dispatch on an unverified guess.

## Third attempt: a smoke test, and a different kind of answer

Built `SMOKE_TEST=1` (launcher mode: skip the ~1hr agentic replay, fire a
bounded, direct curl-based load test against the live server instead) and
`VLLM_DEBUG_QREP=1` (temporary logging of `qrep_decode` per rank per decode
step, via `patches/debug-qrep-logging.diff`). Goal: observe the divergence
directly instead of guessing a third source-level fix blind.

**Side note on tooling:** the debug-logging patch mysteriously failed to
apply via classic `patch -p1` despite being byte-verified correct (checked
context, checked arithmetic, even widened the context block) — `git apply`
accepted the identical hunk cleanly on the first try. Not worth chasing
further; switched that one patch's application mechanism to `git apply`.

### Result: clean run, no divergence — useful negative evidence

Dispatched with `APPLY_PR54494=1`, `APPLY_DEBUG_QREP=1`,
`VLLM_DEBUG_QREP=1`, `SMOKE_TEST=1` (32 concurrent requests, 480s window).
The job appeared to hang for ~50 minutes past when the smoke window should
have closed — turned out to be a **separate bug in the smoke-test's own
cleanup code** (bare `wait` with no arguments waits for *all* background
jobs in the shell, including the vLLM server process itself, backgrounded
much earlier in the script — not anything to do with `#54494`). Cancelled
the run and recovered `server.log` from the uploaded artifact (upload
happens before the job finalizes, so cancellation didn't lose the data).

The actual data: **93,312 logged decode calls, across all 8 ranks, for the
full 8-minute window — `qrep_decode=True` every single time. Zero
divergence. Zero collective errors. Zero crashes.** The server ran
completely clean through the whole smoke window.

This is useful negative evidence, not a failure: it rules out "decode
under any load" as the trigger. Both real deadlocks happened **~40-50
minutes** into serving, specifically right as requests started backing up
on **external KV-cache fetches** (`Deferred: 26 reqs, KV fetch ... in
progress`, `Waiting: 120+`). A 32-concurrent, 8-minute smoke test never
builds that kind of queue or DRAM-offload pressure. The remaining
hypothesis — `qrep_decode` diverging across ranks — still stands, but the
trigger condition looks tied to **sustained runtime and/or KV-offload
queue pressure**, not simple concurrent decode.

## Status

- `#54494`'s `W_K` shape-mismatch bug: **fixed, verified correct, but
  insufficient on its own**.
- The real hang: **not yet root-caused**. Lead (`_gather_query` firing
  asymmetrically) still stands, but a clean 93K-call smoke test under
  light/short load found zero divergence — the trigger needs sustained
  runtime and/or real KV-offload queue pressure to reproduce, which a
  quick smoke test doesn't build up.
- Found and not yet fixed: the smoke-test harness's own cleanup code has a
  `wait`-without-arguments bug that hangs the script after the real smoke
  window finishes (harmless to results since the artifact still uploads,
  but wastes ~50 minutes of runner time per run until fixed).
- `#54494` remains **disabled by default** (`APPLY_PR54494=0`) until the
  divergence bug is resolved. Do not re-enable or re-dispatch without a
  new finding.
- Next options, not yet decided: (a) fix the `wait` bug and run a longer/
  heavier smoke test that deliberately manufactures KV-offload pressure,
  or (b) trace directly into the Kimi-K3 model's per-layer forward code
  (the real caller of `q_dcp_replicated`) to find the divergence condition
  by reading rather than reproducing, or (c) deprioritize this and focus
  on the idle/decode-launch-rate lever instead (28.2% of wall, still
  completely unexplored).
