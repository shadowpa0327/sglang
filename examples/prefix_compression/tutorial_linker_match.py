"""Standalone tutorial: how ``UnifiedCacheLinkerWrapper.match`` is realized.

Pure Python, no sglang or torch imports.  Run it and read the printed trace:

    python examples/prefix_compression/tutorial_linker_match.py

What this mirrors (real file: python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py)

    real code                                        here
    -----------------------------------------------  ---------------------------------
    RadixKey.page_aligned / child_key                page_aligned / child_key
    utils.get_hash_str (C++ backed)                  get_hash_str
    utils.compute_node_hash_values                   compute_node_hash_values
    utils.split_node_hash_value                      split_node_hash_value
    UnifiedTreeNode (multi-page key, hash list)      TreeNode
    UnifiedTreeCore._add_new_node (hash if enabled)  RadixTree._add_new_node
    UnifiedTreeCore._split_node (slice, no rehash)   RadixTree._split_node
    UnifiedTreeCore.match_prefix (splits mid-node)   RadixTree.match_prefix
    UnifiedTreeCore.backfill_missing_hash_values     RadixTree.backfill_missing_hash_values
    UnifiedCacheLinkerWrapper.match                  LinkerWrapper.match
    UnifiedCacheLinkerWrapper._tail_hashes           LinkerWrapper._tail_hashes
    UnifiedCacheLinkerWrapper._sync_restorable       LinkerWrapper._sync_restorable_prefix
    TreeComponent.build_external_linker_transfer     FullComponent / MambaComponent
    UnifiedCacheLinker.lookup                        ToyBackend.lookup (same loop as
                                                      CompressionLinker.lookup)
    ExternalCacheHitMarker                           ExternalCacheHitMarker

Simplifications, on purpose:

* No locks, no eviction policy, no load-back.  Offload only records keys in
  the backend so that a later match can find them.
* "Ranks" are just a list of backends; the all-reduce MIN becomes a Python
  AND over 0/1 masks.
* Slots are integers handed out by a counter, never freed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Optional, Sequence

PAGE_SIZE = 4


# --------------------------------------------------------------------------- #
# 1. Keys and hashing.  Real: RadixKey (radix_cache.py) and mem_cache/utils.py
# --------------------------------------------------------------------------- #
def page_aligned(tokens: Sequence[int]) -> tuple[int, ...]:
    """Real: RadixKey.page_aligned -> key[: len // P * P]."""
    return tuple(tokens[: len(tokens) // PAGE_SIZE * PAGE_SIZE])


def child_key(tokens: Sequence[int], offset: int = 0) -> tuple[int, ...]:
    """Real: RadixKey.child_key -> the FIRST page only, used as the dict key."""
    return tuple(tokens[offset : offset + PAGE_SIZE])


def get_hash_str(
    token_ids: Sequence[int], prior_hash: Optional[str], page_size: int
) -> list[str]:
    """One SHA-256 hex digest per page, chained: page N folds in page N-1's digest.

    ``prior_hash`` is the digest of the page *before* the first page here.  It is
    None only when ``token_ids`` starts at the head of the sequence.
    """
    assert len(token_ids) % page_size == 0
    hashes = []
    prior = bytes.fromhex(prior_hash) if prior_hash else b""
    for start in range(0, len(token_ids), page_size):
        page = token_ids[start : start + page_size]
        digest = hashlib.sha256(prior + bytes(str(list(page)), "utf-8")).digest()
        hashes.append(digest.hex())
        prior = digest
    return hashes


def compute_node_hash_values(node: "TreeNode") -> list[str]:
    """Real: utils.compute_node_hash_values.  Chain from the parent's LAST hash.

    If the parent has no hashes, the chain silently restarts here.  That is the
    aliasing bug backfill_missing_hash_values exists to prevent.
    """
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]
    return get_hash_str(node.key, parent_hash, PAGE_SIZE)


def split_node_hash_value(
    child_hash_value: Optional[list[str]], split_len: int
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Real: utils.split_node_hash_value.  Slice the list; never rehash."""
    if child_hash_value is None:
        return None, None
    split_pages = split_len // PAGE_SIZE
    return child_hash_value[:split_pages], child_hash_value[split_pages:]


def short(h: Optional[str]) -> str:
    return h[:8] if h else "None"


def shorts(hs: Optional[list[str]]) -> str:
    return "None" if hs is None else "[" + " ".join(short(h) for h in hs) + "]"


# --------------------------------------------------------------------------- #
# 2. The radix tree.  A node holds ONE OR MORE pages.  Real: UnifiedTreeCore
# --------------------------------------------------------------------------- #
@dataclass
class TreeNode:
    key: tuple[int, ...]  # page-aligned tokens, possibly many pages
    value: list[int]  # GPU slots, len(value) == len(key)
    hash_value: Optional[list[str]]  # one hash per page; None = never hashed
    parent: Optional["TreeNode"]
    children: dict[tuple[int, ...], "TreeNode"] = field(default_factory=dict)
    external_cache_stored: bool = False
    id: int = 0

    def get_last_hash_value(self) -> Optional[str]:
        """Real: UnifiedTreeNode.get_last_hash_value -> hash_value[-1] or None."""
        if not self.hash_value:
            return None
        return self.hash_value[-1]

    def __repr__(self) -> str:
        if not self.key:
            return "root"
        return f"node#{self.id}[{self.key[0]}..{self.key[-1]}]"


class MatchResult(NamedTuple):
    """Field names copied from sglang.srt.mem_cache.base_prefix_cache.MatchResult."""

    device_indices: list[int]
    last_device_node: TreeNode
    last_host_node: TreeNode
    best_match_node: TreeNode
    host_hit_length: int = 0
    mamba_host_hit_length: int = 0


class RadixTree:
    def __init__(self) -> None:
        # Real: root_node.hash_value = [] seeds every chain.
        self.root = TreeNode(key=(), value=[], hash_value=[], parent=None, id=0)
        self._next_id = 1
        self._next_slot = 0
        # Real: UnifiedTreeCore.enable_external_cache_linker, set True by the
        # wrapper's constructor.  Hashing costs SHA per page, so it is gated.
        self.enable_external_cache_linker = False

    # -- helpers -------------------------------------------------------------
    def _alloc(self, n: int) -> list[int]:
        slots = list(range(self._next_slot, self._next_slot + n))
        self._next_slot += n
        return slots

    def get_last_hash_value(self, node: TreeNode) -> Optional[str]:
        return node.get_last_hash_value()

    @staticmethod
    def _match_len(node_key: tuple[int, ...], query: tuple[int, ...]) -> int:
        """Real: RadixKey.match.  Leading tokens in common, page-rounded down."""
        n = 0
        limit = min(len(node_key), len(query))
        while n < limit and node_key[n] == query[n]:
            n += 1
        return n // PAGE_SIZE * PAGE_SIZE

    # -- node creation: hash the WHOLE node once, only if enabled -------------
    def _add_new_node(self, parent: TreeNode, key: tuple[int, ...]) -> TreeNode:
        """Real: UnifiedTreeCore._add_new_node (unified_tree_core.py:1230)."""
        node = TreeNode(
            key=key,
            value=self._alloc(len(key)),
            hash_value=None,
            parent=parent,
            id=self._next_id,
        )
        self._next_id += 1
        parent.children[child_key(key)] = node
        if self.enable_external_cache_linker:
            node.hash_value = compute_node_hash_values(node)
        return node

    # -- split: prefix half becomes a new parent; hashes are sliced -----------
    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        """Real: UnifiedTreeCore._split_node (unified_tree_core.py:1176).

            parent -> child            =>   parent -> new_node -> child
                      [p1 p2 p3 p4]                   [p1 p2]     [p3 p4]
        """
        assert 0 < split_len < len(child.key) and split_len % PAGE_SIZE == 0
        new_node = TreeNode(
            key=child.key[:split_len],
            value=child.value[:split_len],
            hash_value=None,
            parent=child.parent,
            id=self._next_id,
        )
        self._next_id += 1
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len
        )
        new_node.external_cache_stored = child.external_cache_stored
        new_node.children = {child_key(child.key, split_len): child}
        new_node.parent.children[child_key(new_node.key)] = new_node
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        say(f"split {new_node} | {child}   hashes {shorts(new_node.hash_value)} | {shorts(child.hash_value)}")
        return new_node

    # -- insert --------------------------------------------------------------
    def insert(self, tokens: Sequence[int]) -> TreeNode:
        """Real: UnifiedTreeCore.begin_insert + walk/commit/tail steps."""
        key = page_aligned(tokens)  # Real: key = key.page_aligned(page_size)
        node, pos = self.root, 0
        while pos < len(key):
            breakpoint()
            child = node.children.get(child_key(key, pos))
            if child is None:
                node = self._add_new_node(node, key[pos:])
                break
            m = self._match_len(child.key, key[pos:])
            if m < len(child.key):
                child = self._split_node(child, m)
            node, pos = child, pos + m
        return node

    # -- match: walk while the next page exists on device ----------------------
    def match_prefix(self, tokens: Sequence[int]) -> MatchResult:
        """Real: UnifiedTreeCore.match_prefix.  Splits a node if the match ends
        mid-node, so last_device_node always ends exactly where the hit ends."""
        key = tuple(tokens)
        node, pos = self.root, 0
        indices: list[int] = []
        while pos + PAGE_SIZE <= len(key):
            child = node.children.get(child_key(key, pos))
            if child is None:
                break
            m = self._match_len(child.key, key[pos:])
            if m == 0:
                break
            if m < len(child.key):
                child = self._split_node(child, m)
            indices.extend(child.value)
            node, pos = child, pos + m
        return MatchResult(
            device_indices=indices,
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )

    # -- misc ----------------------------------------------------------------
    def evict_subtree(self, node: TreeNode) -> None:
        assert node.parent is not None
        del node.parent.children[child_key(node.key)]

    def backfill_missing_hash_values(self) -> int:
        """Real: UnifiedTreeCore.backfill_missing_hash_values (line 577).
        Parent-before-child, so every node hashes against a filled parent."""
        filled = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and node.hash_value is None:
                node.hash_value = compute_node_hash_values(node)
                filled += 1
            stack.extend(node.children.values())
        return filled

    def path(self, node: TreeNode) -> list[TreeNode]:
        out = []
        while node.parent is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))

    def dump(self) -> None:
        def rec(node: TreeNode, depth: int) -> None:
            for child in node.children.values():
                stored = " stored" if child.external_cache_stored else ""
                say(f"{'   ' * depth}{child}  pages={len(child.key) // PAGE_SIZE}  hashes={shorts(child.hash_value)}{stored}")
                rec(child, depth + 1)

        say("tree:")
        rec(self.root, 1)


# --------------------------------------------------------------------------- #
# 3. Transfers: what components hand to the backend.
#    Real: sglang.srt.mem_cache.hicache_storage.PoolTransfer / PoolHitPolicy
# --------------------------------------------------------------------------- #
class HitPolicy(str, Enum):
    ALL_PAGES = "all_pages"  # contiguous prefix: every page up to N must exist
    TRAILING_PAGES = "trailing"  # only the page *at* boundary N must exist


class Phase(str, Enum):
    LOOKUP = "lookup"
    OFFLOAD = "offload"


@dataclass
class PoolTransfer:
    name: str
    keys: list[str]
    hit_policy: HitPolicy = HitPolicy.ALL_PAGES


class FullComponent:
    """Real: FullComponent.build_external_linker_transfer (full_component.py:436)."""

    name = "kv"

    def build_external_linker_transfer(
        self, phase: Phase, node: Optional[TreeNode], keys: Optional[list[str]]
    ) -> Optional[PoolTransfer]:
        if phase == Phase.OFFLOAD:
            if node is None or not node.hash_value:
                return None
            return PoolTransfer(self.name, list(node.hash_value))  # every page
        if not keys:
            return None
        return PoolTransfer(self.name, list(keys), HitPolicy.ALL_PAGES)


class MambaComponent:
    """Real: MambaComponent.build_external_linker_transfer (mamba_component.py:660).

    One recurrent-state checkpoint per NODE, keyed by the node's last page.  So
    checkpoints only exist at node boundaries, which makes the restorable set
    sparse.
    """

    name = "mamba"

    def build_external_linker_transfer(
        self, phase: Phase, node: Optional[TreeNode], keys: Optional[list[str]]
    ) -> Optional[PoolTransfer]:
        if phase == Phase.OFFLOAD:
            if node is None or not node.hash_value:
                return None
            return PoolTransfer(self.name, [node.hash_value[-1]], HitPolicy.TRAILING_PAGES)
        if not keys:
            return None
        return PoolTransfer(self.name, list(keys), HitPolicy.TRAILING_PAGES)


# --------------------------------------------------------------------------- #
# 4. The backend ("warehouse").  Knows only pool names and page hashes.
#    Real: UnifiedCacheLinker ABC; lookup loop copied from CompressionLinker.lookup
# --------------------------------------------------------------------------- #
class ToyBackend:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.store: dict[str, set[str]] = {}  # pool name -> set of page hashes

    def has(self, pool: str, key: str) -> bool:
        return key in self.store.get(pool, ())

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        for t in transfers:
            self.store.setdefault(t.name, set()).update(t.keys)
        return True

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every tail length (in pages) restorable on THIS rank.

        Same loop as CompressionLinker.lookup: walk the full-KV keys in order,
        stop at the first missing KV page, and accept a length only when every
        other pool also satisfies its policy at that exact boundary.
        """
        full = next(t for t in transfers if t.name == "kv")
        result = []
        for end, key in enumerate(full.keys, 1):
            if not self.has("kv", key):
                break
            if all(
                t.name == "kv"
                or (
                    t.hit_policy == HitPolicy.TRAILING_PAGES
                    and self.has(t.name, t.keys[end - 1])
                )
                for t in transfers
            ):
                result.append(end)
        return result


# --------------------------------------------------------------------------- #
# 5. The wrapper ("librarian").  This is the part you asked about.
#    Real: UnifiedCacheLinkerWrapper.match / _tail_hashes / _sync_restorable_prefix
# --------------------------------------------------------------------------- #
class ExternalCacheHitMarker(NamedTuple):
    """What match found; consumed later by load_back (not covered here)."""

    prefix_key: list[int]  # device prefix + restorable tail
    tail_hashes: list[str]  # hashes of the tail only, page 0 = first uncached page
    device_hit_len: int


class LinkerWrapper:
    def __init__(
        self, tree: RadixTree, components: list, backends: list[ToyBackend]
    ) -> None:
        self.tree = tree
        self.components = components
        self.backends = backends  # one per "rank"
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Real: cache.tree_core.enable_external_cache_linker = True
        tree.enable_external_cache_linker = True

    # ---- match -------------------------------------------------------------
    def match(self, rid: str, key: list[int], result: MatchResult) -> MatchResult:
        page = PAGE_SIZE
        device_hit_len = len(result.device_indices)
        say(f"tree walk: {device_hit_len} tokens on GPU, stopped at {result.last_device_node}")

        # (a) Device already covers the whole key: nothing to ask.
        if device_hit_len >= len(key):
            say("device covers the whole key -> return unchanged")
            return result

        # (b) Hash the page-aligned tail, anchored on the device node's last hash.
        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        if not tail_hashes:
            say("no hashable tail -> return unchanged")
            return result
        say(f"tail hashes ({len(tail_hashes)} pages): {shorts(tail_hashes)}")

        # (c) Every component must be able to describe a lookup, else no hit.
        lookup_transfers = []
        for component in self.components:
            transfer = component.build_external_linker_transfer(Phase.LOOKUP, None, tail_hashes)
            if transfer is None:
                say(f"component {component.name} cannot build a lookup -> return unchanged")
                return result
            lookup_transfers.append(transfer)
        by_pool = {t.name: t for t in lookup_transfers}
        say(f"lookup pools: {[(t.name, t.hit_policy.value) for t in lookup_transfers]}")

        # (d) Ask every rank's backend, then intersect the sparse sets.
        per_rank = [b.lookup(rid, lookup_transfers) for b in self.backends]
        hit_pages = self._sync_restorable_prefix(per_rank, num_pages=len(tail_hashes))
        if hit_pages == 0:
            say("no common restorable length -> return unchanged")
            return result
        hit_tokens = hit_pages * page

        # (e) Record the promise and rewrite the result the scheduler sees.
        mamba_host_hit_length = 1 if "mamba" in by_pool else 0
        self.hit_markers[rid] = ExternalCacheHitMarker(
            prefix_key=list(key[: device_hit_len + hit_tokens]),
            tail_hashes=tail_hashes[:hit_pages],
            device_hit_len=device_hit_len,
        )
        say(f"HIT: {hit_tokens} tokens restorable -> host_hit_length={hit_tokens}, marker stored")
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            mamba_host_hit_length=max(result.mamba_host_hit_length, mamba_host_hit_length),
        )

    def _tail_hashes(
        self, key: list[int], result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page hashes of the device-uncached tail, chained from the anchor."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.tree.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, producing keys that can never match the store.
                say(f"device node {result.last_device_node} has NO hash -> no anchor -> give up")
                return []
        say(f"anchor hash: {short(last_hash)}")
        tail_len = (len(key) - device_hit_len) // PAGE_SIZE * PAGE_SIZE
        if tail_len == 0:
            return []
        dropped = len(key) - device_hit_len - tail_len
        if dropped:
            say(f"dropping {dropped} unaligned trailing tokens")
        return get_hash_str(key[device_hit_len : device_hit_len + tail_len], last_hash, PAGE_SIZE)

    def _sync_restorable_prefix(self, per_rank: list[list[int]], *, num_pages: int) -> int:
        """Intersect per-rank sets of restorable lengths; return the longest common one.

        Real code builds a 0/1 mask and all-reduces it with MIN across ranks.
        MIN on a 0/1 mask is AND, so the reduction is a set intersection.
        """
        masks = []
        for rank, pages_list in enumerate(per_rank):
            mask = [0] * (num_pages + 1)
            for pages in pages_list:
                if 0 < pages <= num_pages:
                    mask[pages] = 1
            masks.append(mask)
            say(f"rank {rank} restorable lengths {pages_list} -> mask {mask}")
        common = [min(col) for col in zip(*masks)]  # MIN == AND
        say(f"intersection (MIN)                       -> mask {common}")
        hits = [i for i, v in enumerate(common) if v]
        return hits[-1] if hits else 0

    # ---- offload (minimal, just so the store has something) ----------------
    def offload_nodes(self, nodes: list[TreeNode]) -> None:
        """Real: offload_nodes -> _offload_node, minus locks and async acks."""
        for node in nodes:
            if node.external_cache_stored:
                continue
            transfers = [
                c.build_external_linker_transfer(Phase.OFFLOAD, node, None)
                for c in self.components
            ]
            transfers = [t for t in transfers if t is not None]
            if not transfers:
                say(f"offload {node}: nothing to send (no hashes)")
                continue
            for backend in self.backends:
                backend.offload(transfers)
            node.external_cache_stored = True
            say(f"offload {node}: {[(t.name, len(t.keys)) for t in transfers]} keys")


# --------------------------------------------------------------------------- #
# 6. Scenarios
# --------------------------------------------------------------------------- #
_INDENT = "    "


def say(msg: str) -> None:
    print(_INDENT + msg)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def show_result(r: MatchResult) -> None:
    say(
        f"MatchResult(device_indices={len(r.device_indices)} tok, "
        f"host_hit_length={r.host_hit_length}, "
        f"mamba_host_hit_length={r.mamba_host_hit_length}, "
        f"last_device_node={r.last_device_node})"
    )


def run_match(wrapper: LinkerWrapper, rid: str, key: list[int]) -> MatchResult:
    print(f"  request {rid}: key=[{key[0]}..{key[-1]}] ({len(key)} tok)")
    result = wrapper.tree.match_prefix(key)
    result = wrapper.match(rid, key, result)
    show_result(result)
    return result


def scenario_cold_miss() -> None:
    banner("Scenario 1: cold miss (empty tree, empty store)")
    tree = RadixTree()
    wrapper = LinkerWrapper(tree, [FullComponent()], [ToyBackend(0)])
    run_match(wrapper, "r1", list(range(12)))


def scenario_full_external_hit() -> None:
    banner("Scenario 2: one 3-page node, offloaded, then evicted from GPU")
    tree = RadixTree()
    wrapper = LinkerWrapper(tree, [FullComponent()], [ToyBackend(0)])

    tokens = list(range(12))
    leaf = tree.insert(tokens)
    tree.dump()
    wrapper.offload_nodes(tree.path(leaf))
    tree.evict_subtree(tree.path(leaf)[0])
    say("evicted the whole branch from GPU")

    run_match(wrapper, "r2", tokens)


def scenario_split_and_anchor() -> None:
    banner("Scenario 3: a second request SPLITS the node; hashes are sliced, not rehashed")
    tree = RadixTree()
    wrapper = LinkerWrapper(tree, [FullComponent()], [ToyBackend(0)])

    a = list(range(24))  # 6 pages, one node
    leaf_a = tree.insert(a)
    original_hashes = list(leaf_a.hash_value)
    wrapper.offload_nodes([leaf_a])
    tree.dump()

    print("  -- insert B = same first 2 pages + 4 different pages --")
    b = a[:8] + list(range(100, 116))
    leaf_b = tree.insert(b)
    tree.dump()
    prefix_node = tree.path(leaf_b)[0]  # the new [0..7] node shared by A and B
    # A split keeps the ORIGINAL object as the suffix half and creates a new
    # object for the prefix half (same as the real _split_node).  So leaf_a is
    # now the [8..23] node, reachable from prefix_node by its first page.
    a_tail = leaf_a
    assert prefix_node.children[child_key(a, 8)] is a_tail
    say(f"A's tail hashes after split == original[2:] ? {a_tail.hash_value == original_hashes[2:]}")

    tree.evict_subtree(a_tail)
    say("evicted A's tail [8..23] from GPU; it is still in the store")
    tree.dump()

    print("  -- correct: tail hashed from the GPU node's anchor --")
    run_match(wrapper, "r3", a)

    print("  -- what would happen WITHOUT the anchor (hash tail as if from head) --")
    wrong = get_hash_str(a[8:], None, PAGE_SIZE)
    say(f"un-anchored tail hashes: {shorts(wrong)}")
    say(f"stored page-3 hash:      {short(original_hashes[2])}")
    say(f"backend.lookup would see: {wrapper.backends[0].lookup('x', [PoolTransfer('kv', wrong)])}")


def scenario_unaligned_tail() -> None:
    banner("Scenario 4: key length not page aligned -> trailing partial page ignored")
    tree = RadixTree()
    wrapper = LinkerWrapper(tree, [FullComponent()], [ToyBackend(0)])
    tokens = list(range(12))
    leaf = tree.insert(tokens)
    wrapper.offload_nodes(tree.path(leaf))
    tree.evict_subtree(tree.path(leaf)[0])
    run_match(wrapper, "r4", tokens + [99, 98])  # 14 tokens, 3.5 pages


def scenario_sparse_mamba() -> None:
    banner("Scenario 5: hybrid model -> Mamba checkpoints live at NODE boundaries only")
    tree = RadixTree()
    backend = ToyBackend(0)
    wrapper = LinkerWrapper(tree, [FullComponent(), MambaComponent()], [backend])

    tokens = list(range(24))
    # Three requests of growing length create three nodes: 3 + 2 + 1 pages.
    for n in (12, 20, 24):
        tree.insert(tokens[:n])
    tree.dump()
    leaf = tree.insert(tokens)
    path = tree.path(leaf)
    wrapper.offload_nodes(path)
    say("mamba checkpoints landed at page boundaries 3, 5, 6")
    backend.store["mamba"].discard(path[-1].hash_value[-1])
    say("pretend the page-6 checkpoint was never persisted -> checkpoints at 3 and 5")
    tree.evict_subtree(path[0])

    run_match(wrapper, "r5", tokens)
    say("length 6 has KV but no checkpoint, so the longest valid length is 5")


def scenario_multi_rank_intersection() -> None:
    banner("Scenario 6: two ranks keep different checkpoints -> why MIN over masks")
    tree = RadixTree()
    rank0, rank1 = ToyBackend(0), ToyBackend(1)
    wrapper = LinkerWrapper(tree, [FullComponent(), MambaComponent()], [rank0, rank1])

    tokens = list(range(24))
    for n in (12, 16, 20, 24):  # boundaries at pages 3, 4, 5, 6
        tree.insert(tokens[:n])
    leaf = tree.insert(tokens)
    path = tree.path(leaf)
    wrapper.offload_nodes(path)
    # Per-rank store eviction left different checkpoints behind.
    rank0.store["mamba"] = {path[0].hash_value[-1], path[2].hash_value[-1]}  # {3, 5}
    rank1.store["mamba"] = {path[1].hash_value[-1]}  # {4}
    say("rank0 keeps checkpoints at {3,5};  rank1 keeps {4}")
    tree.evict_subtree(path[0])

    run_match(wrapper, "r6", tokens)
    say("reducing per-rank MAXIMA would give MIN(5, 4) = 4, which rank0 cannot restore")
    say("intersecting the SETS gives {} -> 0, which is the only safe answer")


def scenario_device_covers_key() -> None:
    banner("Scenario 7: device already covers the whole key -> early exit")
    tree = RadixTree()
    wrapper = LinkerWrapper(tree, [FullComponent()], [ToyBackend(0)])
    tokens = list(range(8))
    tree.insert(tokens)
    run_match(wrapper, "r7", tokens)


def scenario_backfill() -> None:
    banner("Scenario 8: node built BEFORE hashing was enabled -> no anchor -> backfill")
    tree = RadixTree()
    tokens = list(range(24))
    tree.insert(tokens[:8])  # linker not attached yet: hash_value stays None
    tree.dump()

    backend = ToyBackend(0)
    wrapper = LinkerWrapper(tree, [FullComponent()], [backend])  # enables hashing
    # Another server instance already offloaded the full 6-page prompt.
    backend.store["kv"] = set(get_hash_str(tokens, None, PAGE_SIZE))
    say("store holds all 6 pages of the prompt (offloaded elsewhere)")

    print("  -- before backfill: the guard refuses to hash an un-anchored tail --")
    run_match(wrapper, "r8a", tokens)

    filled = tree.backfill_missing_hash_values()
    say(f"backfill_missing_hash_values() filled {filled} node(s)")
    tree.dump()

    print("  -- after backfill: same query now finds the tail --")
    run_match(wrapper, "r8b", tokens)


if __name__ == "__main__":
    scenario_cold_miss()
    scenario_full_external_hit()
    scenario_split_and_anchor()
    scenario_unaligned_tail()
    scenario_sparse_mamba()
    scenario_multi_rank_intersection()
    scenario_device_covers_key()
    scenario_backfill()
    print()
