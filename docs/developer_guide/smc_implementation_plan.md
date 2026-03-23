# SMC Speculative Decoding — Implementation Plan

This document describes the strongest SMC integration that is practical in the
current SGLang architecture and the implementation that should be treated as the
baseline going forward.

The key decision is:

- The scheduler-visible request remains the parent `Req`.
- SMC particles remain worker-local/internal.
- Each particle owns two internal execution requests:
  - a target-side `Req`
  - a draft-side `Req`

This is intentionally not the original overlap-first design. SMC still keeps
particles worker-local rather than making them first-class scheduler rows.
However, the implementation now supports the overlap/spec-v2 scheduler path by
keeping the parent request scheduler-visible and relaying only parent-level
future payloads (`last_token_ids`, `new_seq_lens`) through the overlap future
map.

## Current Status

Implemented in the codebase:

- `SMC` is a first-class speculative algorithm in `spec_info.py`.
- `server_args.py` exposes `--smc-n-particles`, `--smc-gamma`,
  `--smc-draft-temperature`, `--smc-target-temperature`,
  `--smc-resample-threshold`, and `--smc-resample-method`.
- SMC supports the overlap/spec-v2 scheduler path when
  `SGLANG_ENABLE_SPEC_V2=True`.
- After prefill, the scheduler attaches SMC state to the parent request and
  spawns internal particle requests.
- Each particle now owns separate target-side and draft-side internal `Req`s.
- `SMCWorkerV2` executes SMC decode as a worker-local loop built on ordinary
  extend batches.
- In overlap mode, `SMCWorkerV2` consumes `ModelWorkerBatch` and returns an
  `SMCDraftInput` future payload so the regular overlap scheduler can keep
  running parent-visible decode batches.
- Resampling reuses committed KV prefixes by copying block-table rows and
  bumping allocator refcounts.
- Parent decode collapse happens through `GenerationBatchResult.smc_parent_updates`.
- There is a registered server-launched SMC overlap smoke test under
  `test/registered/spec/test_smc_speculative_decoding.py`.

Explicitly deferred:

- paged-cache support (`page_size > 1`)
- DP attention
- disaggregation
- constrained decoding
- stop-string / stop-regex handling
- parent `return_logprob`
- parent `return_hidden_states`
- parent `return_routed_experts`
- scheduler-visible particle rows

## Design Invariants

The implementation is organized around these invariants.

1. The parent request is the API/lifecycle unit.

- The scheduler only tracks the parent request.
- Streaming, finish handling, retraction, and final cleanup stay parent-shaped.
- The worker returns parent-aligned `SMCParentUpdate`s instead of particle token
  streams.

2. Target state and draft state are separate.

- Each particle has a target-side internal `Req` and a draft-side internal
  `Req`.
- This avoids corrupting `kv_committed_len`, `kv_allocated_len`, and
  `req_to_token_pool` when the draft model advances farther than the target
  model between scoring/resampling points.

3. The committed prefix is the only aliasable prefix.

- Resampling aliases `req_to_token_pool` rows only up to
  `kv_committed_len`.
- Uncommitted speculative tail is never shared across particles.
- Cleanup decrements refcounts over the full internal allocation owned by each
  particle request.

4. Each unfinished particle always ends the step with exactly one logical
   pending token.

- Draft stepping commits the previous pending token and appends a new one.
- Target scoring commits through the scored prefix and then rolls back the final
  pending token for unfinished particles.
- This keeps the next step aligned with SGLang’s standard causal-LM extend
  semantics.

## Runtime Architecture

### Prefill

1. The parent request runs through the normal target prefill path.
2. After prefill, the scheduler validates that the request shape is compatible
   with SMC.
3. The scheduler allocates internal request rows for:
   - `n_particles` target-side particle requests
   - `n_particles` draft-side particle requests
4. Each internal request copies the parent’s committed prefix block table and
   starts from the same visible output state as the parent.

Why prefill stays parent-visible:

- parent streaming/output behavior already exists
- tree-cache ownership already exists on the parent
- internal particle requests should not leak into filtering, merging, or
  retraction policy

### Decode Step

For each parent request in the running batch:

1. Gather active particles from `parent_req.smc_state`.
2. Run up to `gamma` draft one-token extend passes on the draft-side particle
   requests.
3. Record:
   - drafted token IDs per particle
   - exact draft-side logprobs from the actual sampler path
4. Run one target extend pass on the target-side particle requests with
   `return_logprob=True` so the worker gets exact target-side input-token
   logprobs for the drafted suffix.
5. Update particle log weights with:

```text
log_weight += target_logprob - draft_logprob
```

6. For unfinished particles, roll back the final scored token on the target side
   so the particle still ends with one logical pending token.
7. If ESS falls below the configured threshold, resample active particles by:
   - snapshotting source particle state
   - aliasing committed block-table prefixes into destination particles
   - resetting active particle weights to zero
8. Select the current best particle and emit a parent update.

### Overlap Path

When `SGLANG_ENABLE_SPEC_V2=True`:

1. Prefill still runs on the parent-visible target request.
2. The overlap scheduler stores an `SMCDraftInput` future payload containing:
   - `last_token_ids`
   - `new_seq_lens`
3. On decode, the scheduler keeps batching only parent requests.
4. `SMCWorkerV2` ignores EAGLE-style draft/verify row expansion and instead
   runs the worker-local particle loop directly on the forward stream.
5. The worker returns:
   - parent-aligned `smc_parent_updates`
   - a fresh `SMCDraftInput` for the next overlap step

This gives real overlap-scheduler support without forcing SMC particles to
become scheduler rows.

### Parent Update Path

`SMCWorkerV2` returns:

- `smc_parent_updates[i].best_output_ids` for every parent request
- `smc_parent_updates[i].done` when all particles for that parent are terminal
- final finish metadata from the best particle when `done=True`

The scheduler-side decode processor:

- updates the parent request’s visible output on every SMC step
- streams from the parent request as usual
- tears down internal particle state only when the parent is terminal or
  retracted

## Why This Is Stronger Than Scheduler-Visible Particles Right Now

Making particles first-class scheduler rows would require broad changes to:

- `ScheduleBatch.filter_batch()`
- `ScheduleBatch.merge_batch()`
- decode retraction policy
- stream output accounting
- finish handling
- batch/result ownership

That is a large scheduler rewrite.

The parent-visible / particle-internal model keeps the invasive changes inside:

- SMC request-state initialization
- SMC worker execution
- parent decode collapse
- internal cleanup

This gives a working SMC path with much smaller surface area and preserves the
existing scheduler contract.

## Files That Matter

Implemented / updated around this design:

- `python/sglang/srt/speculative/smc_worker_v2.py`
- `python/sglang/srt/speculative/smc_info.py`
- `python/sglang/srt/speculative/spec_info.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/utils.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/allocator.py`

## Remaining Work

### Phase 2: Harden The Overlap Path

- Add more unit coverage for:
  - worker-local resampling snapshots
  - dual-request particle cleanup
  - parent update monotonicity
- Add a stronger correctness test for overlap SMC beyond server smoke.
- Audit all cache free paths for refcount-safe reuse under SMC.

### Phase 3: Feature Parity

- support stop strings / stop regex
- support parent hidden-state / routed-expert outputs
- support parent logprob return
- support paged allocation

### Phase 4: Optional Particle-Row Rewrite

If SMC ever needs deeper integration than the current parent-visible overlap
path, the next project is a scheduler-visible particle-row rewrite. That would
be a separate architectural change, not a prerequisite for the current
implementation.
