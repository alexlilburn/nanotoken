"""Encode a corpus straight from disk: files are memory-mapped and split into
documents in Rust, in parallel."""

import sys
import time
from pathlib import Path

import nanotoken as nt

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("owt_valid.txt")
tok = nt.Tokenizer("google-bert/bert-base-uncased")

source = nt.TextFileSource(path, separator="<|endoftext|>")  # or nt.JsonlFileSource(path, field="text")
start = time.perf_counter()
tokens = tok.encode_files(source)
seconds = time.perf_counter() - start

size = path.stat().st_size
print(f"{len(tokens)} documents, {tokens.num_tokens} tokens in {seconds:.2f}s ({size / seconds / 1e6:.0f} MB/s)")
print("first document:", tokens[0][:16], "...")
