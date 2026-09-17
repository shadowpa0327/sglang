# Prefix compression

See [Architecture](ARCHITECTURE.md) for component boundaries and execution
flows. SGLang owns request/cache mechanics; a Python plugin owns only the
numerical round trip:

```text
plugin.compress(X) -> opaque payload -> plugin.decompress(...) -> X'
```

The same unit enters compression and leaves decompression. Plugins do not split
payloads, expose shared tensors, or reconstruct subranges.

## Compression units

| `compression_unit` | Write trigger | Stored entry | Hit policy | Eviction |
| --- | --- | --- | --- | --- |
| `block` | Completed blocks during request progress | One fixed-size block | Longest consecutive block prefix | Block LRU |
| `request` | `cache_finished_req()` only | One opaque restorable request span | Exact complete-input identity only | Whole-request LRU |

Request mode is named for its lifecycle trigger. It is not a special “prefill
finished” hook: an evaluator may submit a preparation request with one output
token, and SGLang invokes compression when that request is finalized.

```text
block mode                         request mode

chunk progress                     unfinished chunk (hybrid only)
   |                                  |
   v                                  +--> retain deepest recurrent checkpoint
gather one completed block
   |                               request finalization
   v                                  |
compress/store block                  v
                                   gather restorable span once
                                      |
                                      v
                                   compress/store opaque request
```

In request mode the complete original input, `extra_key`, and `cache_salt` form
the identity. A request that merely shares a prefix cannot hit. The payload can
cover a shorter restorable span—for example, the deepest valid hybrid recurrent
checkpoint—but identity still uses the complete input.

## Files

- `linker.py`: request lifecycle, exact identity, gather/scatter, and store routing.
- `pre_rope.py`: optional de-rotation before compression and re-rotation after
  whole-unit decompression.
- `kvcompress.store.BlockStore`: fixed-layout block storage.
- `kvcompress.request_store.RequestStore`: variable-size opaque request storage.
- `kvcompress.api`: the single plugin interface.
- `audit.py`: block-mode provenance audit. Request mode currently rejects
  `audit=true`.

## Request-mode flow

```text
cache_finished_req(req)
    |
    v
CompressionLinker.on_request_progress(..., finished=True)
    |
    +-- exact key = hash(all original input IDs, extra_key, cache_salt)
    +-- choose complete page-aligned restorable span
    +-- gather logical slots once through req_to_token + index_select
    +-- optionally de-rotate all keys
    v
plugin.compress(full span)
    |
    v
RequestStore.insert(exact key, opaque payload, optional recurrent state)

next request admission
    |
    +-- exact complete-input key miss -> no compressed hit
    +-- exact key hit -> allocate the entire stored span
    v
plugin.decompress(full payload)
    |
    +-- optionally re-rotate all keys
    +-- scatter full K/V span once
    +-- restore recurrent checkpoint
```

The store budget is a logical resident-byte limit, not a custom GPU arena:

```text
entry_bytes = owned payload tensor bytes + recurrent-state tensor bytes

while used_bytes + entry_bytes > store_bytes:
    evict the least-recently-used complete request
```

Payloads larger than the budget are compressed but not inserted. Plugin-global
tensors shared across requests remain separate and are reported as
`global_bytes`.

## Configuration

Block mode:

```python
prefix_compression_config=json.dumps({
    "plugin": "/abs/path/algorithm.py",
    "parameters": {},
    "cache_mode": "compressed",
    "compression_unit": "block",
    "block_pages": 32,
    "store_bytes": 10 << 30,
    "metrics_path": "/abs/path/events.jsonl",
    "audit": False,
})
```

Request mode:

```python
prefix_compression_config=json.dumps({
    "plugin": "/abs/path/algorithm.py",
    "parameters": {},
    "cache_mode": "compressed",
    "compression_unit": "request",
    "store_bytes": 10 << 30,
    "metrics_path": "/abs/path/events.jsonl",
})
```

`compression_unit` defaults to `block`. The old `scope` setting is rejected;
there is no `prompt` alias. `block_pages` applies only to block mode.

## Status and events

`GET /prefix_compression/status` reports `compression_unit` and the selected
store description. `restoration_boundary_tokens` is the block/chunk common
boundary in block mode, the chunk/page common checkpoint boundary for hybrid
request mode, and one page for non-hybrid request mode. Block stores report
`stored_blocks`; request stores report `stored_requests`, logical
used/free/peak bytes, whole-request evictions, and cumulative compression byte
counters.

Request mode emits one `compress` event per stored request, exact-hit `lookup`
events, one `decompress` event per restored request, and whole-request `evict`
events. Block event shapes remain unchanged.

## Current limits

One GPU, BF16/FP16 ordinary full-attention KV, synchronous compression, and no
LoRA/speculative decoding/ReplaySSM/int8 recurrent checkpoints. Request mode
does not support partial-prefix hits or byte-level audit. Its dynamic tensor
allocations honor the logical store budget but do not preallocate a physical
GPU arena. A hybrid request also needs an unfinished chunk boundary before its
final prefill chunk; a short or unchunked hybrid request has no earlier
restorable recurrent state and emits `compress_skipped` instead of being stored.
Requests with raw input embeddings or positional embedding overrides are also
skipped because token IDs alone are not an exact KV identity for them.
