# Block-granular prefix compression

See [Architecture](ARCHITECTURE.md) for Mermaid diagrams, execution flows,
memory ownership, and mode controls.

SGLang serves; a Python plugin owns the numerics. The unit of everything on the
token axis is the **block**: `block_pages` consecutive pages of one prompt.

```text
prompt (tokens)
|-- blk0 --|-- blk1 --|-- blk2 --|- tail -|
   stored     stored     stored    never

block = 1 key = 1 codec call = 1 payload (+ 1 raw recurrent state)
```

| | rule |
| --- | --- |
| key | chained hash of the block's last page (`get_hash_str`), so equal token prefixes give equal keys |
| trigger | the request's own prefill: every chunk end stores each newly completed block |
| KV | `plugin.compress` on `[block_pages, layers, page_tokens, kv_heads, dim]` key/value |
| Mamba (hybrid) | the live recurrent + conv state at the block end, copied raw |
| hit | whole blocks from the prompt start; stop at the first missing block |
| restore | decode the hit blocks into request-private slots, copy the last block's state |
| tail | never stored, always recomputed |
| native reuse | off in `compressed` mode; the radix tree is a chunk cache |

There is no partial addressing inside a block, no object shared between
blocks, and nothing tied to radix nodes: node splits, evictions and
write-through never touch the store.

## Files

- `linker.py` - `CompressionLinker` (external-linker backend `compression`) and
  `NativePoolAdapter` (slots <-> plugin tensors, recurrent state copies).
- `kvcompress.store` - engine-independent `BlockStore`.
- `kvcompress.api` - `KVCompressionPlugin` / `CompressedPayload`; `kvcompress.loader`
  loads external Python codec files. Local `store.py` and `plugin.py` are import shims.
- `pre_rope.py` - optional key space: de-rotate keys with the model's own RoPE
  table before `compress`, re-rotate after `decompress`.
- `audit.py` - opt-in byte-level provenance (source round trip, poisoned source
  slots, exact scatter, attention-read hashes, no prefix recomputation).

Codec implementations live in workspace `codecs/` (identity, row INT8, packed
INT4/INT2, xKV cross-layer SVD, template); example paths are compatibility symlinks.

## Hooks into SGLang

```text
scheduler ---- chunk done ----> cache_unfinished_req
          ---- finish -------> cache_finished_req
                                  |  on_request_progress(req, processed)
                                  v
                          CompressionLinker: store new blocks
admission ---- match_prefix ---> linker.match -> lookup(block keys)
          ---- init_load_back -> _load_private -> load / start_layer_wise_loading
```

`processed` counts prompt tokens whose KV sits in the request's slots. For a
hybrid model a block is stored only when a chunk ends exactly at the block end
and no decode step has run, because that is the only moment its recurrent state
exists; `block_pages * page_size` must therefore be a multiple of
`chunked_prefill_size`. The linker also publishes `prefill_boundary_tokens`
(= `block_pages * page_size`), and with chunked prefill the scheduler's
`PrefillAdder` caps every chunk of every request at the next absolute multiple
of it (`len(prefix_indices) + extend`), so a batch that shares its chunk budget,
or a request resuming after a restored prefix, still ends a chunk on every block
end. This holds in every mode and for full-attention models too, so chunk
boundaries are aligned across separate cache-mode runs.

## Configuration

```python
engine = sglang.Engine(
    model_path=...,
    page_size=256,
    chunked_prefill_size=8192,
    enable_unified_cache_external_linker=True,
    unified_cache_external_linker_backend="compression",
    prefix_compression_config=json.dumps({
        "plugin": "/abs/path/algorithm.py",   # or "identity" / "int8"
        "parameters": {},
        "cache_mode": "compressed",          # fixed for this server lifetime
        "block_pages": 32,                    # 8192 tokens at page 256
        "key_space": "auto",                  # follow the plugin; or pre_rope / post_rope
        "metrics_path": "/abs/path/events.jsonl",
        "audit": False,
    }),
)
```

The persistent HTTP server exposes:

| endpoint | effect |
| --- | --- |
| `GET /prefix_compression/status` | resolved codec identity, mode, block size, idle/write state |
| `POST /prefix_compression/control` with `{"action":"reset"}` | idle-only run reset, preserve fixed mode, restore writes |
| same endpoint with `frozen` / `unfreeze` | idle-only global compressed storage write control |
| `GET /prefix_compression/events/{request_id}` | collect and consume events for that request, including shared batch events |

The evaluator supplies `rid` on `/generate` and joins the response/events to run,
session and case IDs. It never slices a shared event file to attribute requests.
Legacy in-process mode RPCs can reset only the already configured mode; switching
modes requires another server. Opening or closing a frontend session has no engine effect.

The asynchronous evaluator and optional server runner live in `kv-compress/src/kvcompress`.
Fixed-prefix runs prepare sources, await completion, freeze once, score targets,
and restore writes. Growing protocols do not freeze between tasks.

## Events

`compress` (per block: key, tokens, payload/metadata/state bytes, ratio),
`block_skipped_no_state`, `lookup` (hit pages), `private_restore`, `decompress`
(blocks, tokens, latency), `load_batch`, `trial_mode`, `frozen`, and the
`audit_*` records when auditing.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python test/manual/compression/smoke.py \
  --model /path/to/Llama-3.1-8B-Instruct --output /tmp/smoke --plugin identity --mixed --overlap
# hybrid: add --hybrid and point --model at Qwen3.5-9B
```

Unit tests: `test/registered/unit/mem_cache/test_compression_*.py`,
`test_xkv_pre_rope.py`, `test_packed_quant_plugin.py`.

## Scope

One GPU, BF16/FP16 `MHATokenToKVPool` (the full-attention pool of a hybrid
model), synchronous compression on the scheduler thread, blocks completed
during decode are not stored. Not covered: TP/PP, LoRA, speculative decoding,
ReplaySSM ring state, int8 recurrent checkpoints, quantized active KV.
