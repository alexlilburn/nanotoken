from __future__ import annotations

import json

import pytest
from conftest import BERT_UNCASED, adversarial, fuzz

import nanotoken as nt


@pytest.fixture(scope="module")
def tok():
    return nt.Tokenizer(BERT_UNCASED)


def docs_for_files(n_repeat: int = 1) -> list[str]:
    base = [d for d in adversarial() + fuzz(300) if "<|endoftext|>" not in d]
    return base * n_repeat


def expected(tok, docs):
    return tok.encode_batch(docs, parallel=False).to_list()


@pytest.mark.parametrize("separator", ["<|endoftext|>", "\n\n", "\n"])
@pytest.mark.parametrize("repeat", [1, 60])
def test_text_separator_matches_split(tok, tmp_path, separator, repeat):
    docs = docs_for_files(repeat)
    data = separator.join(docs).encode()
    path = tmp_path / "corpus.txt"
    path.write_bytes(data + separator.encode())
    split = data.decode().split(separator)
    got = tok.encode_files(nt.TextFileSource(path, separator=separator))
    assert got.to_list() == expected(tok, split)
    assert tok.encode_files(nt.TextFileSource([path], separator=separator), parallel=False).to_list() == got.to_list()


def test_jsonl(tok, tmp_path):
    docs = docs_for_files(40)
    lines = []
    for i, doc in enumerate(docs):
        lines.append(
            json.dumps({"id": i, "text": doc, "meta": {"nested": [1, {"text": "no"}]}}, ensure_ascii=i % 2 == 0)
        )
        if i % 50 == 0:
            lines.append("   ")
    path = tmp_path / "corpus.jsonl"
    path.write_text("\r\n".join(lines) + "\n", encoding="utf-8")
    got = tok.encode_files(nt.JsonlFileSource(path))
    assert got.to_list() == expected(tok, docs)


def test_jsonl_errors(tok, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "ok"}\n{"body": "no text"}\n')
    with pytest.raises(ValueError, match="no string field"):
        tok.encode_files(nt.JsonlFileSource(path))
    path.write_text('{"text": "ok"}\n{"text": \n')
    with pytest.raises(ValueError, match="invalid JSONL"):
        tok.encode_files(nt.JsonlFileSource(path))
    with pytest.raises(ValueError, match="no string field"):
        tok.encode_files(nt.JsonlFileSource(path, field="missing"))


def test_whole_files_and_fragments(tok, tmp_path):
    small = tmp_path / "small.txt"
    small.write_text("Hello there, [MASK] world.")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    big_doc = " [SEP] ".join(docs_for_files(30))
    big = tmp_path / "big.txt"
    big.write_text(big_doc, encoding="utf-8")
    got = tok.encode_files([small, empty, big])
    assert got.to_list() == expected(tok, ["Hello there, [MASK] world.", big_doc])


def test_invalid_utf8_is_replaced(tok, tmp_path):
    raw = b"caf\xc3\xa9 \xff\xfe broken \xe2\x82 end<|endoftext|>second \xc3"
    path = tmp_path / "bad.txt"
    path.write_bytes(raw)
    got = tok.encode_files(nt.TextFileSource(path, separator="<|endoftext|>"))
    docs = [d.decode("utf-8", "replace") for d in raw.split(b"<|endoftext|>")]
    assert got.to_list() == expected(tok, docs)


def test_missing_file(tok, tmp_path):
    with pytest.raises(ValueError, match="No such file"):
        tok.encode_files(tmp_path / "nope.txt")


def test_source_repr_and_validation(tmp_path):
    assert "separator" in repr(nt.TextFileSource(tmp_path / "a.txt", separator=b"\n"))
    assert nt.JsonlFileSource([tmp_path / "a", tmp_path / "b"], field="body").field == "body"
    with pytest.raises(ValueError):
        nt.TextFileSource(tmp_path / "a.txt", separator="")
