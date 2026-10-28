"""Download prompt sources from Hugging Face as regeneration-ready conversation JSONL.

Draft models train best on answers written by their own target model, so a prompt
source only supplies the system and user turns; the target regenerates every
assistant turn (regen_driver.py). See draft_data_regen/PROMPT_SOURCES.md for what
each source contains.

    # what is available
    python draft_data_regen/scripts/download_prompt_sources.py --list

    # the Qwen3.8-Flash-Next thinking blend: PerfectBlend + OpenCodeInstruct (+ gaps)
    python draft_data_regen/scripts/download_prompt_sources.py \
        --sources perfectblend opencodeinstruct aya nemotron-if \
        --sample opencodeinstruct=250000 aya=60000 \
        --decontaminate-against ./eval_datasets \
        --output-dir ./cache/dataset/prompts

    # exactly the prompts (order, reasoning effort) of a reference run, from its lock
    python draft_data_regen/scripts/download_prompt_sources.py \
        --lock draft_data_regen/locks/qwen38-flash-next-think-stage1 \
        --output-dir ./cache/dataset/prompts

    # raw Open-SWE-Traces sessions written by one trajectory model (used as is)
    python draft_data_regen/scripts/download_prompt_sources.py --swe-traces qwen38_flash_next \
        --output-dir ./cache/dataset/prompts

Each source is written to <output-dir>/<source>_prompts.jsonl in a random order
(seeded), one {"id", "source", "conversations"[, "language"]} row per source
conversation, so regenerating any prefix of the file regenerates a random sample.
Rows pass the regenerators' input check: optional leading system message, then
alternating user/assistant turns with non-empty content. Assistant turns of
multi-turn sources are kept (the regenerator ignores and replaces them); a
source that ships user turns only gets a placeholder between them. Rows are
dropped when they are invalid, carry tool definitions (the regenerators do not
pass tools), duplicate an earlier prompt of the same source, or share a 13-word
sequence with any string in the --decontaminate-against files.

<output-dir>/manifest.json records, per source, the dataset revision that was
read, the row counts and the drop reasons. Pass it back with --revisions-from to
download exactly the same revisions again.

--lock DIR rebuilds a reference run's prompt files instead: DIR/lock.json pins each
source's dataset revision and DIR/prompts.jsonl.gz lists (source, id,
reasoning_effort) in run order. The rows come out in that order with their
"reasoning_effort", which regen_driver.py uses as is; nothing is sampled or filtered.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regen_common import validate_conversation  # noqa: E402

PLACEHOLDER = "[assistant turn to be regenerated]"


# --------------------------------------------------------------------------- sources


def _messages(msgs, role_key="role", content_key="content", role_map=None):
    out = []
    for m in msgs:
        role = m.get(role_key)
        role = (role_map or {}).get(role, role)
        out.append({"role": role, "content": m.get(content_key)})
    return out


def conv_sharegpt(row, idx):
    roles = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "chatgpt": "assistant",
        "system": "system",
    }
    return (
        row.get("id", idx),
        _messages(row["conversations"], "from", "value", roles),
        {},
    )


def conv_messages(row, idx):
    return (
        row.get("prompt_id") or row.get("uuid") or row.get("id") or idx,
        _messages(row["messages"]),
        {},
    )


def conv_opencodeinstruct(row, idx):
    return row.get("id") or idx, [{"role": "user", "content": row["input"]}], {}


def conv_aya(row, idx):
    rid = hashlib.sha1(row["inputs"].encode()).hexdigest()[:16]
    return (
        rid,
        [{"role": "user", "content": row["inputs"]}],
        {"language": row.get("language")},
    )


def conv_structured(row, idx):
    params = row["responses_create_params"]
    if params.get("tools"):
        return idx, None, {"drop": "has tools"}
    return idx, _messages(params["input"]), {}


def conv_magicoder(row, idx):
    return idx, [{"role": "user", "content": row["instruction"]}], {}


def conv_codealpaca(row, idx):
    text = row["instruction"] + (
        f"\n\n{row['input']}" if (row.get("input") or "").strip() else ""
    )
    return idx, [{"role": "user", "content": text}], {}


# key -> (HF repo, how to load, converter, license, one-line description)
SOURCES: dict[str, dict[str, Any]] = {
    "perfectblend": dict(
        repo="mlabonne/open-perfectblend",
        load={"split": "train"},
        conv=conv_sharegpt,
        id_from="index",
        license="Apache-2.0",
        rows="1.42M",
        desc="General chat, math, code and instruction following (PerfectBlend mix); 14% multi-turn",
    ),
    "ultrachat": dict(
        repo="HuggingFaceH4/ultrachat_200k",
        load={"split": "train_sft"},
        conv=conv_messages,
        id_from="prompt_id",
        license="MIT",
        rows="208K",
        desc="Multi-turn assistant chat (3-4 user turns typical)",
    ),
    "opencodeinstruct": dict(
        repo="nvidia/OpenCodeInstruct",
        load={"split": "train"},
        conv=conv_opencodeinstruct,
        id_from="id",
        license="CC-BY-4.0",
        rows="5.0M",
        desc="Single-turn coding tasks (mostly Python), synthetic",
    ),
    "aya": dict(
        repo="CohereLabs/aya_dataset",
        load={"split": "train"},
        conv=conv_aya,
        license="Apache-2.0",
        rows="202K",
        desc="Human-written multilingual prompts, 65 languages",
    ),
    "nemotron-if": dict(
        repo="nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
        load={"jsonl": "data/instruction_following.jsonl"},
        conv=conv_messages,
        license="CC-BY-4.0 / ODC-By",
        rows="249K",
        desc="Verifiable instruction following (format/length constraints); ~60% multi-turn",
    ),
    "nemotron-chat": dict(
        repo="nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
        load={"jsonl": "data/chat.jsonl"},
        conv=conv_messages,
        license="CC-BY-4.0 / ODC-By",
        rows="637K (~55% without prompts)",
        desc="Multi-turn open chat; lmsys/WildChat-seeded rows ship without prompts and are dropped "
        "unless restored with the dataset's prepare_chat_prompts.py",
    ),
    "nemotron-structured": dict(
        repo="nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        # The subsets have different schemas, so each is loaded on its own.
        load={
            "each_data_files": [
                "direct_generation/*.parquet",
                "diversified_tasks/*.parquet",
            ],
            "split": "train",
        },
        conv=conv_structured,
        license="CC-BY-4.0",
        rows="54K",
        desc="Structured output (JSON/XML/YAML/TOML to a schema); the tool-calling subset is not "
        "downloaded (the regenerators do not pass tools)",
    ),
    "magicoder": dict(
        repo="ise-uiuc/Magicoder-Evol-Instruct-110K",
        load={"split": "train"},
        conv=conv_magicoder,
        license="Apache-2.0",
        rows="111K",
        desc="Evol-Instruct coding prompts (~1% overlap HumanEval: decontaminate)",
    ),
    "codealpaca": dict(
        repo="sahil2801/CodeAlpaca-20k",
        load={"split": "train"},
        conv=conv_codealpaca,
        license="CC-BY-4.0",
        rows="20K",
        desc="Short coding instructions",
    ),
}
# The DaoCloud/Qwen3.8-27B-Drafter-SFT prompt set is deliberately not offered: do not
# download it. Build mixes from the original sources above.

SWE_TRACES_REPO = "nvidia/Open-SWE-Traces"
# trajectory model directory -> (dataset config, sessions)
SWE_TRACE_MODELS = {
    "qwen38_flash_next": ("v2.0", 489109),  # xhigh 168,824, medium 151,689, low 168,596
    "qwen38_27b": ("v1.2", 235396),
    "qwen36_27b": ("v1.1", 212931),
    "deepseek_v4_flash_0731": ("v1.1", 50295),
    "minimax_m25": ("v1.0", 90422),
    "qwen35_122b": ("v1.0", 60797),
}


# --------------------------------------------------------------------------- loading


class JsonlRows:
    """Random access to a (possibly large) JSONL file through line offsets."""

    def __init__(self, path: str):
        self.path = path
        self.offsets = []
        with open(path, "rb") as f:
            pos = 0
            for line in f:
                if line.strip():
                    self.offsets.append(pos)
                pos += len(line)
        self.f = open(path, "rb")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        self.f.seek(self.offsets[i])
        return json.loads(self.f.readline())


def load_rows(spec: dict[str, Any], revision: str):
    load = dict(spec["load"])
    if "jsonl" in load:
        from huggingface_hub import hf_hub_download

        return JsonlRows(
            hf_hub_download(
                spec["repo"], load["jsonl"], repo_type="dataset", revision=revision
            )
        )
    from datasets import load_dataset

    if "each_data_files" in load:
        files = load.pop("each_data_files")
        return [
            row
            for pattern in files
            for row in load_dataset(
                spec["repo"], revision=revision, data_files=pattern, **load
            )
        ]
    return load_dataset(spec["repo"], revision=revision, **load)


def iterate(rows, order: list[int]) -> Iterator[tuple[int, dict]]:
    """(source index, row) in the given order; HF datasets are read in batches."""
    if hasattr(rows, "select"):
        j = 0
        for batch in rows.select(order).iter(batch_size=1000):
            keys = list(batch)
            for vals in zip(*(batch[k] for k in keys)):
                yield order[j], dict(zip(keys, vals))
                j += 1
    else:
        for i in order:
            yield i, rows[i]


def resolve_revision(repo: str, pinned: dict[str, str]) -> str:
    if repo in pinned:
        return pinned[repo]
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo).sha


# --------------------------------------------------------------------------- cleaning


def normalize(msgs: list[dict]) -> list[dict] | None:
    """Merge consecutive same-role turns, put placeholders between user turns, drop a
    trailing assistant turn; None when a prompt turn is missing."""
    out: list[dict] = []
    for m in msgs:
        role, content = m["role"], m["content"]
        if role not in ("system", "user", "assistant"):
            return None
        if not isinstance(content, str) or not content.strip():
            if role == "assistant":
                content = PLACEHOLDER
            elif role == "system":
                continue  # an empty system prompt is no system prompt
            else:
                return None
        if role == "system" and out:
            role = (
                "user"  # a mid-conversation system turn becomes part of the next prompt
            )
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
            continue
        if role == "user" and out and out[-1]["role"] == "user":
            out.append({"role": "assistant", "content": PLACEHOLDER})
        out.append({"role": role, "content": content})
    while out and out[-1]["role"] == "assistant":
        out.pop()
    return out


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Decontaminator:
    """Drops prompts sharing a 13-word sequence with any string of the eval files, or (for
    eval strings shorter than 13 words but at least 4) equal to one after normalization.
    """

    N = 13

    def __init__(self, paths: list[str]):
        self.grams: dict[int, int] = {}
        self.short: dict[str, int] = {}
        self.names: list[str] = []
        files = []
        for p in paths:
            pp = Path(p)
            files += (
                sorted(pp.rglob("*.jsonl")) + sorted(pp.rglob("*.json"))
                if pp.is_dir()
                else [pp]
            )
        for path in files:
            idx = len(self.names)
            self.names.append(path.stem)
            for obj in self._objects(path):
                for text in self._texts(obj):
                    g = self._grams(text)
                    for h in g:
                        self.grams.setdefault(h, idx)
                    w = words(text)
                    if not g and len(w) >= 4:
                        self.short.setdefault(" ".join(w), idx)
        print(
            f"decontamination: {len(files)} eval files, {len(self.grams)} 13-grams, "
            f"{len(self.short)} short texts",
            flush=True,
        )

    @staticmethod
    def _objects(path: Path) -> Iterator[Any]:
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            yield json.load(open(path, encoding="utf-8"))

    def _texts(self, obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from self._texts(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from self._texts(v)

    def _grams(self, text: str) -> set[int]:
        w = words(text)
        return {hash(" ".join(w[i : i + self.N])) for i in range(len(w) - self.N + 1)}

    def hit(self, msgs: list[dict]) -> str | None:
        for m in msgs:
            if m["role"] == "assistant":
                continue
            for h in self._grams(m["content"]):
                if h in self.grams:
                    return self.names[self.grams[h]]
            key = " ".join(words(m["content"]))
            if key in self.short:
                return self.names[self.short[key]]
        return None


# --------------------------------------------------------------------------- main


def build_source(
    key: str,
    revision: str,
    out_path: Path,
    sample: int | None,
    seed: int,
    decon: Decontaminator | None,
    dedup: bool,
) -> dict[str, Any]:
    spec = SOURCES[key]
    t0 = time.time()
    rows = load_rows(spec, revision)
    order = list(range(len(rows)))
    random.Random(f"{seed}-{key}").shuffle(order)
    conv: Callable = spec["conv"]
    kept, drops, seen = 0, Counter(), set()
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for n, (i, raw) in enumerate(iterate(rows, order)):
            if sample is not None and kept >= sample:
                break
            rid, msgs, extra = conv(raw, i)
            if msgs is None:
                drops[extra.get("drop", "unconvertible")] += 1
                continue
            msgs = normalize(msgs)
            if not msgs:
                drops["missing prompt text (withheld or empty)"] += 1
                continue
            bad = validate_conversation(msgs, error_style="regeneration")
            if bad:
                drops[bad] += 1
                continue
            if dedup:
                h = hashlib.sha1(
                    "\x00".join(
                        m["content"] for m in msgs if m["role"] != "assistant"
                    ).encode()
                ).digest()
                if h in seen:
                    drops["duplicate prompt"] += 1
                    continue
                seen.add(h)
            if decon is not None:
                why = decon.hit(msgs)
                if why:
                    drops[f"eval overlap: {why}"] += 1
                    continue
            row = {"id": f"{key}-{rid}", "source": key, "conversations": msgs}
            if extra.get("language"):
                row["language"] = extra["language"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if (n + 1) % 200000 == 0:
                print(f"  {key}: {n + 1}/{len(rows)} read, {kept} kept", flush=True)
    os.replace(tmp, out_path)
    stats = {
        "repo": spec["repo"],
        "revision": revision,
        "load": spec["load"],
        "license": spec["license"],
        "rows_in_source": len(rows),
        "kept": kept,
        "dropped": dict(drops),
        "sample": sample,
        "seed": seed,
        "output": str(out_path),
        "seconds": round(time.time() - t0),
    }
    print(
        f"{key}: kept {kept} of {len(rows)} ({dict(drops)}) -> {out_path}", flush=True
    )
    return stats


def row_positions(key: str, rows) -> dict[str, int]:
    """Row index of every id the source's converter writes (without the source prefix)."""
    spec = SOURCES[key]
    col = spec.get("id_from")
    if col == "index":
        return {str(i): i for i in range(len(rows))}
    if col and hasattr(rows, "select"):
        return {str(v): i for i, v in enumerate(rows[col])}
    return {
        str(spec["conv"](raw, i)[0]): i
        for i, raw in iterate(rows, list(range(len(rows))))
    }


def build_from_lock(lock_dir: Path, out_dir: Path, overwrite: bool) -> dict[str, Any]:
    """Rebuild the exact prompt files of a reference run: the lock lists (source, id,
    reasoning_effort) in run order and pins each source's dataset revision."""
    lock = json.load(open(lock_dir / "lock.json"))
    items: dict[str, list[tuple[str, str | None]]] = {}
    with gzip.open(lock_dir / "prompts.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.setdefault(r["source"], []).append(
                (r["id"], r.get("reasoning_effort"))
            )
    manifest = {"lock": lock["name"]}
    for key, wanted in items.items():
        out_path = out_dir / f"{key}_prompts.jsonl"
        src = lock["sources"][key]
        if out_path.exists() and not overwrite:
            print(f"{key}: {out_path} exists; skipping (use --overwrite to rebuild)")
            continue
        t0 = time.time()
        rows = load_rows(SOURCES[key], src["revision"])
        pos = row_positions(key, rows)
        missing = [rid for rid, _ in wanted if rid not in pos]
        if missing:
            raise SystemExit(
                f"{key}: {len(missing)} locked ids not in {src['repo']}@{src['revision']}, "
                f"e.g. {missing[:3]}"
            )
        order = [pos[rid] for rid, _ in wanted]
        kept, drops = 0, Counter()
        tmp = out_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for (rid, effort), (i, raw) in zip(wanted, iterate(rows, order)):
                _, msgs, extra = SOURCES[key]["conv"](raw, i)
                msgs = normalize(msgs) if msgs is not None else None
                bad = (
                    validate_conversation(msgs, error_style="regeneration")
                    if msgs
                    else "missing prompt text"
                )
                if bad:
                    drops[bad] += 1
                    continue
                row = {"id": f"{key}-{rid}", "source": key, "conversations": msgs}
                if effort:
                    row["reasoning_effort"] = effort
                if extra.get("language"):
                    row["language"] = extra["language"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
        os.replace(tmp, out_path)
        manifest[key] = {
            "repo": src["repo"],
            "revision": src["revision"],
            "locked": len(wanted),
            "kept": kept,
            "dropped": dict(drops),
            "output": str(out_path),
            "seconds": round(time.time() - t0),
        }
        print(
            f"{key}: {kept} of {len(wanted)} locked prompts ({dict(drops)}) -> {out_path}",
            flush=True,
        )
    return manifest


def download_swe_traces(
    models: list[str], out_dir: Path, pinned: dict[str, str]
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    unknown = [m for m in models if m not in SWE_TRACE_MODELS]
    if unknown:
        raise SystemExit(
            f"unknown Open-SWE-Traces models {unknown}; choose from {list(SWE_TRACE_MODELS)}"
        )
    revision = resolve_revision(SWE_TRACES_REPO, pinned)
    target = out_dir / "open_swe_traces"
    snapshot_download(
        SWE_TRACES_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=target,
        allow_patterns=["README.md", "data_distribution.md"]
        + [f"data/*/{m}/**" for m in models],
    )
    print(f"Open-SWE-Traces {models} -> {target}", flush=True)
    return {
        "repo": SWE_TRACES_REPO,
        "revision": revision,
        "models": models,
        "output": str(target),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--list", action="store_true", help="print the available sources and exit"
    )
    ap.add_argument(
        "--sources", nargs="*", default=[], choices=sorted(SOURCES), metavar="SOURCE"
    )
    ap.add_argument("--output-dir", type=Path, default=Path("cache/dataset/prompts"))
    ap.add_argument(
        "--sample",
        nargs="*",
        default=[],
        metavar="SOURCE=N",
        help="keep at most N rows of a source (a random sample; default: all)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--decontaminate-against",
        nargs="*",
        default=[],
        metavar="PATH",
        help="eval JSONL/JSON files or directories of them; every string in them is checked",
    )
    ap.add_argument(
        "--no-dedup", action="store_true", help="keep exact duplicate prompts"
    )
    ap.add_argument(
        "--swe-traces",
        nargs="*",
        default=[],
        metavar="MODEL",
        help=f"also download raw Open-SWE-Traces sessions of these trajectory models: {list(SWE_TRACE_MODELS)}",
    )
    ap.add_argument(
        "--revisions-from",
        type=Path,
        help="manifest.json of an earlier download to pin revisions",
    )
    ap.add_argument(
        "--lock",
        type=Path,
        help="rebuild a reference run's exact prompt files from a lock directory "
        "(lock.json + prompts.jsonl.gz); --sources, --sample and decontamination are ignored",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild sources whose output already exists",
    )
    a = ap.parse_args(argv)

    if a.lock:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_from_lock(a.lock, a.output_dir, a.overwrite)
        json.dump(manifest, open(a.output_dir / "manifest.json", "w"), indent=1)
        if not a.swe_traces:
            return
    if a.list or not (a.sources or a.swe_traces or a.lock):
        print(f"{'source':<20} {'rows':<30} {'license':<22} repo / contents")
        for k, s in SOURCES.items():
            print(
                f"{k:<20} {s['rows']:<30} {s['license'][:22]:<22} {s['repo']}\n{'':<74}{s['desc']}"
            )
        print(
            f"\n--swe-traces MODEL: raw agent sessions from {SWE_TRACES_REPO} (used as recorded, see build_swe_trace_rows.py)"
        )
        for m, (cfg, n) in SWE_TRACE_MODELS.items():
            print(f"  {m:<24} config {cfg}, {n:,} sessions")
        return

    samples = {}
    for item in a.sample:
        k, _, v = item.partition("=")
        if k not in a.sources or not v.isdigit():
            ap.error(f"--sample {item!r}: expected SOURCE=N for one of --sources")
        samples[k] = int(v)
    pinned = {}
    if a.revisions_from:
        old = json.load(open(a.revisions_from))
        pinned = {
            v["repo"]: v["revision"]
            for v in old.values()
            if isinstance(v, dict) and "revision" in v
        }

    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output_dir / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}
    decon = Decontaminator(a.decontaminate_against) if a.decontaminate_against else None
    for key in a.sources:
        out = a.output_dir / f"{key}_prompts.jsonl"
        if out.exists() and not a.overwrite:
            print(f"{key}: {out} exists; skipping (use --overwrite to rebuild)")
            continue
        stats = build_source(
            key,
            resolve_revision(SOURCES[key]["repo"], pinned),
            out,
            samples.get(key),
            a.seed,
            decon,
            not a.no_dedup,
        )
        stats["decontaminated_against"] = a.decontaminate_against
        manifest[key] = stats
        json.dump(manifest, open(manifest_path, "w"), indent=1)
    if a.swe_traces:
        manifest["open-swe-traces"] = download_swe_traces(
            a.swe_traces, a.output_dir, pinned
        )
        json.dump(manifest, open(manifest_path, "w"), indent=1)


if __name__ == "__main__":
    main()
