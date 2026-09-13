# SGLang prefix compression architecture

The compression backend has two storage paths behind one numerical plugin API.
The paths share native-pool gathering, optional RoPE conversion, and the
external-linker wrapper; they do not share storage semantics.

## Components

```text
Scheduler / UnifiedRadixCache
             |
             v
UnifiedCacheLinkerWrapper
  - passes the Req to request-aware lookup
  - allocates private restore destinations
             |
             v
CompressionLinker
  |                         |
  | unit=block              | unit=request
  v                         v
BlockStore                  RequestStore
fixed slots                 variable opaque payloads
block hash lookup           exact full-input lookup
block LRU                   whole-request LRU
  |                         |
  +------------+------------+
               v
       one KVCompressionPlugin API
               |
               v
NativePoolAdapter <----> native KV and recurrent pools
```

The plugin contract is unit-symmetric:

```text
compress(X) -> payload
decompress(payload) -> reconstruction shaped like X
```

`X` is one block in block mode and one complete restorable request span in
request mode. The plugin never sees block keys, request identities, eviction,
payload partitions, or partial reconstruction.

## Block path

```text
cache_unfinished_req / cache_finished_req
    |
    v
find newly completed block boundaries
    |
    v
for each missing block hash
    gather -> optional de-RoPE -> compress -> fixed BlockStore slot

lookup: consecutive block hashes
restore: decode each block -> scatter each block -> final checkpoint
```

One block has `block_pages * page_size` tokens. Prefix sharing and recurrent
checkpoint availability determine the longest restorable sequence of blocks.
This path preserves the original block-level lookup and eviction behavior.

## Request write path

```text
unfinished callback
    |
    +-- non-hybrid: no work
    +-- hybrid: retain deepest page-aligned recurrent checkpoint only

cache_finished_req
    |
    v
key = SHA256(version, complete original token IDs, extra_key, cache_salt)
    |
    v
choose restorable_end
    +-- non-hybrid: largest complete page before the final input token
    +-- hybrid: deepest retained checkpoint not beyond that limit
    |
    v
req_to_token[request row, 0:restorable_end]
    |
    v
one gather across every full-attention layer
    |
    v
optional de-RoPE -> plugin.compress(full span)
    |
    v
RequestStore.insert(key, opaque payload, checkpoint)
```

The final token is excluded because SGLang keeps at least one input token for
logit computation during normal prefix matching. Generated tokens never enter
the stored artifact. Consequently, a hybrid request needs an earlier unfinished
chunk checkpoint. If no such checkpoint exists (for example, a short unchunked
prompt), request compression is skipped because the final recurrent state is
newer than the largest restorable KV prefix.

## Request lookup and restore

```text
match_prefix(req)
    |
    v
UnifiedCacheLinkerWrapper.match(key, req, native_result)
    |
    v
CompressionLinker.lookup_request(req, transfers)
    |
    +-- recompute exact complete-input identity
    +-- require one live RequestStore entry
    +-- require the transfer to cover the whole stored span
    v
report exactly stored_pages (or zero)
    |
    +-- pin the matched request entry against LRU eviction
    |
    v
allocate private destination slots
    |
    v
RequestStore.reconstruct(exact key)       one plugin call
    |
    v
optional re-RoPE -> scatter entire span   one linker operation
    |
    +-- unpin the entry after reconstruction or cancellation
    |
    v
restore request checkpoint, then run suffix prefill/decode
```

The base linker API keeps `lookup(rid, transfers)` for existing backends and
adds a default `lookup_request(req, transfers)` adapter. Existing backends use
the default; request compression overrides it because exact identity requires
the complete `Req`.

There are no prefix-hash probes in request mode. A longer, shorter, or
same-prefix/different-tail input is a miss. The stored payload may cover less
than the full input only when that exact request's deepest restorable boundary
is earlier.

## Memory ownership

| Owner | Block mode | Request mode |
| --- | --- | --- |
| Native pools | Active request K/V and recurrent state | Same |
| Compression store | Preallocated fixed block slots | Owned variable-size payload tensors |
| Codec globals | `global_bytes`, once per server/rank | Same |
| Reconstruction | One or more block outputs | One complete stored request output |

RequestStore maintains a logical byte budget:

```text
resident request bytes = payload tensor bytes + recurrent checkpoint bytes
used_bytes <= store_bytes
```

It evicts complete LRU entries until a new request fits. CPU metadata is
reported but does not consume the GPU tensor budget. `global_bytes` stays
outside `store_bytes` because it is shared across all requests by the plugin.

## RoPE boundary

When `key_space=pre_rope`, the linker de-rotates exactly the selected input unit.
`NativeRoPEDecodePlugin` stores that unit's `start_position` and layer IDs in its
ordinary payload metadata. Decompression reconstructs the same complete unit
and reapplies RoPE once. No context rebinding or post-compression partitioning
is required.

## Controls and observability

`cache_mode=compressed` disables native cross-request reuse and restores into
request-private slots. `frozen` stops new writes while retaining existing
entries. Reset clears either store and all linker-side progress/hit maps.

Request-mode status exposes exact logical budget usage and cumulative encoded
bytes. Request-mode events are request-granular; block-mode events remain
block-granular. Byte-level provenance audit remains block-only for now.
