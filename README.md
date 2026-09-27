# nanotoken

**WordPiece tokenization at GB/s, with exactly the same output as HuggingFace `tokenizers`.**

nanotoken does one thing: WordPiece, the tokenizer behind BERT and the encoder/embedding models built on it (bert-*, DistilBERT, MiniLM, BGE, E5, and many more). It is a Rust core with Python bindings. It reads HuggingFace `tokenizer.json` files and can drop in for a transformers tokenizer.

> **Standing on the shoulders of [gigatoken](https://github.com/marcelroed/gigatoken).**
> nanotoken exists because of Marcel Rød's excellent gigatoken, which tokenizes BPE and SentencePiece at tens of GB/s and lists WordPiece as not yet supported. nanotoken fills that gap by applying gigatoken's ideas to WordPiece:
> - persistent per-worker caches of pre-token encodings,
> - coarse parallel work units, with huge documents split at boundaries that provably don't change the output,
> - mmap'd file sources read entirely in Rust,
> - a compatibility wrapper that stands in for transformers.
>
> If you tokenize BPE or SentencePiece models, use gigatoken. See [Acknowledgements](#acknowledgements).

## Performance

Encoding the OpenWebText validation sample ([`owt_valid.txt`](https://huggingface.co/datasets/stanford-cs336/owt-sample), 290 MB) with the `BAAI/bge-small-en-v1.5` tokenizer, which is uncased BERT WordPiece. Measured on an Apple M3 laptop (8 cores: 4 performance + 4 efficiency):

| | throughput | vs HF |
|---|---:|---:|
| nanotoken `encode_files` | **1,447 MB/s** (317 M tokens/s) | **56×** |
| nanotoken `encode_batch(list[str])` | 1,277 MB/s | 50× |
| HF `tokenizers` `encode_batch_fast` (multithreaded Rust) | 25.7 MB/s | 1× |

All 20,177 documents in the compared 100 MB produce identical ids.

As a drop-in for `transformers` (`tokenizer(texts, padding=True, truncation=True, return_tensors="pt")` on OWT sentences, 138 characters on average):

| batch size | transformers | nanotoken `as_hf()` | speedup |
|---:|---:|---:|---:|
| 32 | 1.86 ms | 0.035 ms | 53× |
| 256 | 24.3 ms | 0.120 ms | 203× |

Reproduce with the built-in benchmark:

```bash
curl -LO https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
nanotoken bench BAAI/bge-small-en-v1.5 owt_valid.txt --separator "<|endoftext|>" --validate
```

## Installation

```bash
pip install nanotoken
```

Prebuilt abi3 wheels cover Python 3.10+ on Linux (x86_64, aarch64), macOS (Apple Silicon, Intel) and Windows (x64). Other platforms build from the source distribution, which needs a Rust toolchain.

## Usage

### Native API

```python
import nanotoken as nt

tok = nt.Tokenizer("BAAI/bge-small-en-v1.5")  # Hub repo id, tokenizer.json, vocab.txt, a directory,
                                               # a tokenizers.Tokenizer or a transformers tokenizer
tok.encode("Hello, world!")                    # array([ 101, 7592, 1010, 2088,  999,  102], dtype=uint32)
tok.tokenize("unaffable")                      # ['una', '##ffa', '##ble']

batch = tok.encode_batch(texts)                # RaggedIds: one flat uint32 buffer + int64 row offsets
batch[0], len(batch), batch.lengths, batch.to_list()

inputs = tok.encode_batch_padded(texts, max_length=512)   # {'input_ids', 'token_type_ids', 'attention_mask'}
                                                          # int64 (rows x width) numpy arrays, model-ready
tok.decode(batch[0], skip_special_tokens=True)
```

Batch methods release the GIL and fan out across a thread pool. Inside multiprocessing workers and forked children they automatically run on the calling thread, because a thread pool inherited across `fork()` deadlocks. Pass `parallel=True/False` to override. The output is identical either way.

### Drop-in for transformers

```python
from transformers import AutoTokenizer
import nanotoken as nt

tokenizer = nt.Tokenizer(AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")).as_hf()
# or: nt.Tokenizer("sentence-transformers/all-MiniLM-L6-v2").as_hf()

batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
```

`as_hf()` supports:
- `__call__`, `encode`, `encode_plus`, `batch_encode_plus`, `tokenize` and `pad`
- `decode` and `batch_decode`
- `convert_ids_to_tokens`, `convert_tokens_to_ids` and `convert_tokens_to_string`
- `get_vocab`, `get_added_vocab` and `len()`
- the special-token attributes (`cls_token_id`, `pad_token`, `all_special_ids`, ...)
- `model_max_length`, `padding_side` and `truncation_side`

It accepts every padding and truncation strategy, `pad_to_multiple_of`, `return_special_tokens_mask`, `return_length`, and `return_tensors` of `None`, `"np"` or `"pt"`. The test suite compares it with transformers over a grid of 2,400 calls. Sentence pairs, offset mappings, overflowing tokens and pre-tokenized input raise `NotImplementedError`.

### Files

```python
tokens = tok.encode_files(nt.TextFileSource("corpus.txt", separator="<|endoftext|>"))
tokens = tok.encode_files(nt.JsonlFileSource(["a.jsonl", "b.jsonl"], field="text"))
tokens = tok.encode_files(["doc1.txt", "doc2.txt"])  # each file is one document
```

Files are memory-mapped and split into documents in Rust, with no per-document Python objects:
- A text file becomes `data.split(separator)`, with a trailing empty document dropped.
- A JSONL file yields one document per non-blank line.
- Invalid UTF-8 is replaced with U+FFFD.

## What is supported

nanotoken implements the WordPiece subset of `tokenizer.json`. Loading a tokenizer that uses anything else raises `nanotoken.UnsupportedTokenizerError` rather than silently tokenizing differently.

| component | supported |
|---|---|
| model | `WordPiece`, including untyped legacy files: `unk_token`, `continuing_subword_prefix`, `max_input_chars_per_word` |
| normalizer | `BertNormalizer`, `NFC`, `NFD`, `NFKC`, `NFKD`, `Lowercase`, `StripAccents`, `Sequence`, none |
| pre_tokenizer | `BertPreTokenizer`, `Whitespace`, `WhitespaceSplit`, `Punctuation`, `Digits`, `Sequence`, none |
| post_processor | `BertProcessing`, `RobertaProcessing`, `TemplateProcessing` (single-sequence template) |
| decoder | `WordPiece` (with cleanup), none |
| added tokens | special and non-special, with `lstrip`, `rstrip`, `single_word` and `normalized` |

Parity with `tokenizers` 0.23.2 is tested on these corpora:
- 38 adversarial strings covering accents, combining marks, CJK, emoji and ZWJ sequences, control and zero-width characters, exotic whitespace, Greek final sigma, Turkish dotted I, ligatures, overlong words and embedded special tokens,
- 400 seeded random-Unicode documents,
- their concatenation.

The tests run these corpora against:
- `google-bert/bert-base-{uncased,cased,multilingual-cased,chinese}`
- `distilbert/distilbert-base-uncased`
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`
- `intfloat/e5-base-v2`
- 11 synthetic normalizer and pre-tokenizer combinations
- every added-token flag

## How it works

1. **Added tokens first.** Special tokens are matched on the raw text with a leftmost-longest Aho–Corasick automaton. This follows HF's `lstrip`/`rstrip`/`single_word` rules exactly.
2. **ASCII whitespace is a hard boundary.** Every supported pre-tokenizer removes whitespace, and every supported normalizer is char-local and keeps `' ' \t \n \r` as whitespace. So the tokens of a whitespace-delimited chunk depend only on that chunk's bytes. (This is not true of `\x0b`/`\x0c`, which BERT's `clean_text` *deletes*, joining their neighbours; nanotoken treats them accordingly.) Three things follow from this:
   - **Two-level cache.** Chunks such as `"world,"` are cached by their raw bytes, and normalized pre-tokens such as `"world"` in a second level. Keys of up to 15 bytes are packed into a `u128` and stored in a linear-probing table with 32-byte slots. The scan finds the next 64 chunks and prefetches their slots before looking any of them up, so tail-of-Zipf DRAM misses overlap. Caches belong to pooled workers and persist across calls.
   - **ASCII fast path.** On a cache miss, an ASCII chunk is normalized through a 128-entry table derived from the Unicode normalizer itself, so the two cannot disagree. For BERT, splitting happens in the same pass. Anything non-ASCII goes through the full Unicode pipeline, which uses the same Unicode tables as HF.
   - **Safe parallel splits.** Batches are grouped into work units of at least 1 MiB. A huge document is split at whitespace that no added-token match touches, and the tokens are reassembled identically.
3. **WordPiece** runs greedy longest-match-first over two byte tries (word-initial pieces and `##` continuations), and replaces the whole word with `[UNK]` if any part fails to match, as HF does.
4. **Rows are assembled in Rust.** Special tokens, truncation, padding and masks are written in parallel straight into the numpy buffers.

## Limitations

- Only single sequences are supported. `text_pair`, `return_offsets_mapping`, `return_overflowing_tokens` and `is_split_into_words` raise `NotImplementedError`.
- If `max_length` is smaller than the number of special tokens, HF keeps one whole *word*. nanotoken raises `ValueError` instead.
- There is no WordPiece training, and no reading of compressed (`.gz`/`.zst`) files.
- The caches hold up to 2²⁰ entries per level per thread, up to about 130 MiB per thread when full. Tune this with `nanotoken.set_cache_entries(n)` or the `NANOTOKEN_CACHE_ENTRIES` environment variable.
- CJK-heavy text has long whitespace-free chunks that bypass the chunk cache, so it is slower than Latin-script text.

## Development

```bash
uv venv && uv pip install maturin numpy pytest ruff tokenizers transformers torch
maturin develop --release
pytest                      # add -m "not network" to skip Hub downloads
cargo test --no-default-features
NANOTOKEN_BENCH_FILE=owt_valid.txt cargo bench --no-default-features --bench encode
```

## Acknowledgements

- **[gigatoken](https://github.com/marcelroed/gigatoken)** by Marcel Rød. nanotoken's design comes directly from it:
  - pooled workers with persistent pre-token caches and packed keys,
  - prefetched cache probes,
  - chunked parallel encoding that splits oversized documents at output-preserving boundaries,
  - memory-mapped `TextFileSource`/`JsonlFileSource` inputs,
  - fork-aware parallelism defaults,
  - the `as_hf()` compatibility wrapper,
  - the `bench --validate` CLI.

  No code was copied; the ideas are gigatoken's. If you use nanotoken in research, please cite gigatoken:

  ```bibtex
  @software{roed2026gigatoken,
    author = {Marcel R{\o}d},
    title = {{G}igatoken: SIMD and Cache Hierarchies for 1000x Faster Byte-Pair Encoding Tokenization on Modern CPUs},
    url = {https://github.com/marcelroed/gigatoken},
    year = {2026},
  }
  ```
- **[HuggingFace tokenizers](https://github.com/huggingface/tokenizers)** is the reference implementation that nanotoken matches.

## License

MIT
