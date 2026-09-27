"""`nanotoken bench`: time nanotoken against HuggingFace tokenizers on a corpus."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from nanotoken import JsonlFileSource, TextFileSource, Tokenizer


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _read_docs(path: Path, limit: int, separator: str | None, jsonl_field: str | None) -> list[str]:
    with open(path, "rb") as f:
        data = f.read(limit)
    if jsonl_field is not None:
        import json

        lines = data.split(b"\n")
        if len(data) == limit:
            lines = lines[:-1]
        return [json.loads(line)[jsonl_field] for line in lines if line.strip()]
    text = data.decode("utf-8", errors="ignore")
    if separator is None:
        return [text]
    docs = text.split(separator)
    if len(data) == limit and len(docs) > 1:
        docs = docs[:-1]
    if docs and docs[-1] == "":
        docs = docs[:-1]
    return docs


def _line(name: str, seconds: float, nbytes: int, ntokens: int) -> str:
    mb = nbytes / 1e6
    return (
        f"{name:>10}: {seconds:8.3f} s | {mb:10.2f} MB at {mb / seconds:9.2f} MB/s | "
        f"{ntokens / 1e6:8.2f} Mtok at {ntokens / 1e6 / seconds:8.2f} Mtok/s"
    )


def bench(args: argparse.Namespace) -> int:
    path = Path(args.file)
    tok = Tokenizer(args.model)
    if args.jsonl_field is not None:
        source = JsonlFileSource(path, field=args.jsonl_field)
    else:
        source = TextFileSource(path, separator=args.separator)
    total = path.stat().st_size
    print(f"{'cpu':>10}: {_cpu_name()}, {os.cpu_count()} cores")

    ours = tok.encode_files(source)
    best = float("inf")
    for _ in range(args.repeat):
        start = time.perf_counter()
        ours = tok.encode_files(source, add_special_tokens=not args.no_special_tokens)
        best = min(best, time.perf_counter() - start)
    print(_line("nanotoken", best, total, ours.num_tokens))

    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError:
        print("install `tokenizers` to compare against HuggingFace")
        return 0
    limit = min(total, int(args.hf_mb * 1e6))
    docs = _read_docs(path, limit, args.separator, args.jsonl_field)
    nbytes = sum(len(d.encode()) for d in docs)
    hf = HFTokenizer.from_str(tok.to_json())
    hf.no_truncation()
    hf.no_padding()
    start = time.perf_counter()
    encodings = hf.encode_batch_fast(docs, add_special_tokens=not args.no_special_tokens)
    hf_seconds = time.perf_counter() - start
    hf_tokens = sum(len(e.ids) for e in encodings)
    print(_line("hf", hf_seconds, nbytes, hf_tokens))
    print(f"nanotoken is {(total / best) / (nbytes / hf_seconds):.1f}x faster than hf")

    if args.validate:
        mine = tok.encode_batch(docs, add_special_tokens=not args.no_special_tokens)
        bad = [i for i, e in enumerate(encodings) if not np.array_equal(mine[i], np.asarray(e.ids, dtype=np.uint32))]
        if bad:
            print(f"validation FAILED: {len(bad)} of {len(docs)} documents differ (first: {bad[0]})")
            return 1
        print(f"validation OK: {len(docs)} documents match")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanotoken")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench", help="time encoding a file with nanotoken vs HuggingFace tokenizers")
    b.add_argument("model", help="Hub repo id, or path to tokenizer.json / vocab.txt / a directory")
    b.add_argument("file", help="text or JSONL corpus")
    b.add_argument("--separator", help="document separator for text files, e.g. '<|endoftext|>'")
    b.add_argument("--jsonl-field", help="treat the file as JSONL and read documents from this field")
    b.add_argument("--hf-mb", type=float, default=100.0, help="MB of the file to encode with HF (default 100)")
    b.add_argument("--repeat", type=int, default=3, help="nanotoken timing runs; the best is reported")
    b.add_argument("--validate", action="store_true", help="check ids match HF on the compared subset")
    b.add_argument("--no-special-tokens", action="store_true", help="encode without [CLS]/[SEP]")
    args = parser.parse_args(argv)
    return bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
