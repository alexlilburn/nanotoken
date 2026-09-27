"""Token ids must be identical to HuggingFace `tokenizers`."""

from __future__ import annotations

import json

import pytest
from conftest import BERT_UNCASED, HUB_MODELS, hub_tokenizer_json
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import normalizers, pre_tokenizers

import nanotoken as nt


def reference(hf: HFTokenizer, docs: list[str], add_special_tokens: bool) -> list[list[int]]:
    hf.no_truncation()
    hf.no_padding()
    return [e.ids for e in hf.encode_batch(docs, add_special_tokens=add_special_tokens)]


def assert_parity(tokenizer_json: str, docs: list[str]) -> None:
    hf = HFTokenizer.from_str(tokenizer_json)
    tok = nt.Tokenizer.from_json(tokenizer_json)
    for add_special_tokens in (True, False):
        expected = reference(hf, docs, add_special_tokens)
        got = tok.encode_batch(docs, add_special_tokens=add_special_tokens).to_list()
        for doc, e, g in zip(docs, expected, got, strict=True):
            assert g == e, f"mismatch on {doc!r}"
        serial = tok.encode_batch(docs, add_special_tokens=add_special_tokens, parallel=False).to_list()
        assert serial == expected
    for doc in docs[:50]:
        assert tok.encode(doc).tolist() == hf.encode(doc).ids
    ids = [e.ids for e in hf.encode_batch(docs[:100])]
    for skip in (True, False):
        assert tok.decode_batch(ids, skip_special_tokens=skip) == hf.decode_batch(ids, skip_special_tokens=skip)


def test_fixture_parity(corpus):
    assert_parity(BERT_UNCASED.read_text(), corpus)


def _variant(**edits) -> str:
    hf = HFTokenizer.from_file(str(BERT_UNCASED))
    for name, value in edits.items():
        setattr(hf, name, value)
    return hf.to_str()


VARIANTS = {
    "cased": dict(normalizer=normalizers.BertNormalizer(lowercase=False)),
    "strip-accents-only": dict(normalizer=normalizers.BertNormalizer(lowercase=False, strip_accents=True)),
    "no-clean-text": dict(normalizer=normalizers.BertNormalizer(clean_text=False, handle_chinese_chars=False)),
    "nfkc-lowercase": dict(
        normalizer=normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase(), normalizers.StripAccents()])
    ),
    "nfd-nfc": dict(normalizer=normalizers.Sequence([normalizers.NFD(), normalizers.NFC()])),
    "whitespace-regex": dict(pre_tokenizer=pre_tokenizers.Whitespace()),
    "whitespace-split": dict(pre_tokenizer=pre_tokenizers.WhitespaceSplit()),
    "digits": dict(
        pre_tokenizer=pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Digits(individual_digits=False)]
        )
    ),
    "punct-contiguous": dict(
        pre_tokenizer=pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(behavior="contiguous")]
        )
    ),
    "punctuation-only": dict(pre_tokenizer=pre_tokenizers.Punctuation()),
    "no-normalizer-no-pretokenizer": dict(normalizer=None, pre_tokenizer=None),
}


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_parity(name, corpus):
    assert_parity(_variant(**VARIANTS[name]), corpus)


def _edit_added(**flags) -> str:
    data = json.loads(BERT_UNCASED.read_text())
    for token in data["added_tokens"]:
        if token["content"] in flags:
            token.update(flags[token["content"]])
    return json.dumps(data)


@pytest.mark.parametrize(
    "flags",
    [
        {"[MASK]": {"lstrip": True}},
        {"[SEP]": {"rstrip": True}, "[CLS]": {"lstrip": True, "rstrip": True}},
        {"[MASK]": {"single_word": True}},
        {"[MASK]": {"normalized": True}},
    ],
    ids=["lstrip", "strips", "single-word", "normalized"],
)
def test_added_token_flags(flags, corpus):
    assert_parity(_edit_added(**flags), corpus)


def test_added_non_special_tokens(corpus):
    hf = HFTokenizer.from_file(str(BERT_UNCASED))
    hf.add_tokens(["hello world", "Straße", "中文", "##ly"])
    hf.add_special_tokens(["<ent>"])
    assert_parity(hf.to_str(), corpus + ["say hello world <ent> Straße 中文 quickly"])


@pytest.mark.network
@pytest.mark.parametrize("repo_id", HUB_MODELS)
def test_hub_parity(repo_id, corpus):
    assert_parity(hub_tokenizer_json(repo_id), corpus)
