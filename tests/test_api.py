from __future__ import annotations

import json
import multiprocessing
import sys

import numpy as np
import pytest
from conftest import BERT_UNCASED, adversarial, fuzz
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, normalizers

import nanotoken as nt


@pytest.fixture(scope="module")
def tok():
    return nt.Tokenizer(BERT_UNCASED)


def test_ragged_ids(tok):
    texts = ["Hello world", "", "a b c", "unaffable"]
    r = tok.encode_batch(texts)
    assert len(r) == 4
    assert r.ids.dtype == np.uint32 and r.offsets.dtype == np.int64
    assert r[0].tolist() == [101, 7592, 2088, 102]
    assert r[-1].tolist() == tok.encode("unaffable").tolist()
    assert [row.tolist() for row in r] == r.to_list()
    assert r.lengths.tolist() == [4, 2, 5, 5]
    assert r[1:3].to_list() == r.to_list()[1:3]
    assert r[3:1].to_list() == []
    assert r.num_tokens == 16
    with pytest.raises(IndexError):
        r[4]
    assert tok.encode_batch_list(texts) == r.to_list()


def test_inputs(tok):
    texts = ["Hello world", "naïve café"]
    assert tok.encode_batch(t for t in texts).to_list() == tok.encode_batch(texts).to_list()
    assert tok.encode_batch([t.encode() for t in texts]).to_list() == tok.encode_batch(texts).to_list()
    assert tok.encode_batch(tuple(texts)).to_list() == tok.encode_batch(texts).to_list()
    assert tok.encode(b"Hello world").tolist() == tok.encode("Hello world").tolist()
    with pytest.raises(TypeError, match="list of documents"):
        tok.encode_batch("not a list")
    with pytest.raises(TypeError, match="str or bytes"):
        tok.encode_batch(["ok", 3])
    assert len(tok.encode_batch([])) == 0


def test_truncation(tok):
    text = "one two three four five six seven"
    right = tok.encode_batch([text], max_length=5).to_list()[0]
    left = tok.encode_batch([text], max_length=5, truncation_side="left").to_list()[0]
    assert right == [101, 2028, 2048, 2093, 102]
    assert left == [101] + tok.encode(text, add_special_tokens=False).tolist()[-3:] + [102]
    assert tok.encode_batch([text], max_length=3, add_special_tokens=False).to_list() == [[2028, 2048, 2093]]
    with pytest.raises(ValueError, match="no room"):
        tok.encode_batch([text], max_length=1)


def test_padded(tok):
    texts = ["Hello world", "a much longer sentence than the first one", ""]
    out = tok.encode_batch_padded(texts)
    assert list(out) == ["input_ids", "token_type_ids", "attention_mask"]
    ids, mask = out["input_ids"], out["attention_mask"]
    assert ids.dtype == np.int64 and ids.shape == (3, 10)
    assert ids[0].tolist() == [101, 7592, 2088, 102] + [0] * 6
    assert mask.sum(axis=1).tolist() == [4, 10, 2]
    left = tok.encode_batch_padded(texts, padding_side="left", max_length=8, padding="max_length")
    assert left["input_ids"].shape == (3, 8)
    assert left["input_ids"][0].tolist() == [0] * 4 + [101, 7592, 2088, 102]
    assert left["input_ids"][1, -1] == 102
    multiple = tok.encode_batch_padded(texts, pad_to_multiple_of=16, return_special_tokens_mask=True)
    assert multiple["input_ids"].shape == (3, 16)
    assert multiple["special_tokens_mask"][0].tolist() == [1, 0, 0, 1] + [1] * 12
    with pytest.raises(ValueError, match="longer than max_length"):
        tok.encode_batch_padded(texts, max_length=5, padding="max_length", truncation=False)


def test_decode(tok):
    ids = tok.encode("Hello, world! It's [MASK] time.")
    assert tok.decode(ids) == "[CLS] hello, world! it ' s [MASK] time. [SEP]"
    assert tok.decode(ids, skip_special_tokens=True) == "hello, world! it ' s time."
    assert tok.decode(ids.astype(np.int64).tolist()) == tok.decode(ids)
    assert tok.decode(np.asarray(ids, dtype=np.int32)) == tok.decode(ids)
    assert tok.decode_batch([ids, ids[:3]]) == [tok.decode(ids), tok.decode(ids[:3])]
    assert tok.decode([99999999]) == ""
    with pytest.raises(ValueError, match="invalid token id"):
        tok.decode([-1])


def test_vocab_and_lookup(tok):
    assert tok.vocab_size == 30522
    assert tok.token_to_id("[CLS]") == 101 and tok.id_to_token(101) == "[CLS]"
    assert tok.token_to_id("definitely-not-a-token") is None
    assert tok.id_to_token(-5) is None
    assert tok.get_vocab()["hello"] == 7592
    assert tok.tokenize("unaffable!") == ["una", "##ffa", "##ble", "!"]
    assert tok.pad_token_id == 0


def test_loading_paths(tmp_path, tok):
    hf = HFTokenizer.from_file(str(BERT_UNCASED))
    text = "Loading from many sources works."
    reference = tok.encode(text).tolist()
    assert nt.Tokenizer(hf).encode(text).tolist() == reference
    assert nt.Tokenizer(tok).encode(text).tolist() == reference
    (tmp_path / "tokenizer.json").write_text(BERT_UNCASED.read_text())
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 128, "pad_token": "[PAD]"}))
    from_dir = nt.Tokenizer(tmp_path)
    assert from_dir.encode(text).tolist() == reference
    assert from_dir.config["model_max_length"] == 128
    assert from_dir.as_hf().model_max_length == 128
    vocab = [t for t, _ in sorted(hf.get_vocab(with_added_tokens=False).items(), key=lambda kv: kv[1])]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    assert nt.Tokenizer(tmp_path / "vocab.txt").encode(text).tolist() == reference
    assert (
        nt.Tokenizer.from_vocab(tmp_path / "vocab.txt").encode_batch(adversarial()).to_list()
        == tok.encode_batch(adversarial()).to_list()
    )
    (tmp_path / "tokenizer.json").unlink()
    assert nt.Tokenizer(tmp_path).encode(text).tolist() == reference
    with pytest.raises(FileNotFoundError):
        nt.Tokenizer(tmp_path / "missing" / "tokenizer.json")
    with pytest.raises(TypeError):
        nt.Tokenizer(12345)


def test_unsupported():
    bpe = HFTokenizer(models.BPE())
    with pytest.raises(nt.UnsupportedTokenizerError, match="only implements WordPiece"):
        nt.Tokenizer(bpe)
    hf = HFTokenizer.from_file(str(BERT_UNCASED))
    hf.normalizer = normalizers.Replace("a", "b")
    with pytest.raises(nt.UnsupportedTokenizerError, match="Replace"):
        nt.Tokenizer(hf)
    assert issubclass(nt.UnsupportedTokenizerError, ValueError)


def test_parallel_matches_serial(tok):
    docs = (adversarial() + fuzz(500)) * 60
    assert sum(len(d.encode()) for d in docs) > 2 << 20
    parallel = tok.encode_batch(docs, parallel=True)
    serial = tok.encode_batch(docs, parallel=False)
    assert np.array_equal(parallel.ids, serial.ids) and np.array_equal(parallel.offsets, serial.offsets)
    padded = tok.encode_batch_padded(docs, max_length=64, padding="max_length", parallel=True)
    assert np.array_equal(
        padded["input_ids"],
        tok.encode_batch_padded(docs, max_length=64, padding="max_length", parallel=False)["input_ids"],
    )
    tok.clear_cache()
    assert np.array_equal(tok.encode_batch(docs).ids, parallel.ids)


def _child_encode(queue):
    tok = nt.Tokenizer(BERT_UNCASED)
    docs = (adversarial() + fuzz(200)) * 60
    queue.put(tok.encode_batch(docs).num_tokens)


@pytest.mark.skipif(sys.platform == "win32", reason="fork start method")
def test_fork_after_parallel_use(tok):
    docs = (adversarial() + fuzz(200)) * 60
    expected = tok.encode_batch(docs, parallel=True).num_tokens
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child_encode, args=(queue,))
    proc.start()
    proc.join(timeout=60)
    assert proc.exitcode == 0, "forked child hung or crashed"
    assert queue.get(timeout=5) == expected
