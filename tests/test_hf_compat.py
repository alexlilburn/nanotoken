"""HFCompat must behave like the transformers tokenizer it replaces."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from conftest import BERT_UNCASED, adversarial

import nanotoken as nt

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

TEXTS = adversarial()[:30] + ["short", "a much longer sentence with quite a few more tokens in it than the others"]


def _normalize(value):
    if isinstance(value, torch.Tensor):
        return ("pt", str(value.dtype), value.tolist())
    if isinstance(value, np.ndarray):
        return (
            "np",
            str(value.dtype),
            [np.asarray(v).tolist() for v in value] if value.dtype == object else value.tolist(),
        )
    return ("py", value)


def outcome(fn, *args, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fn(*args, **kwargs)
    except Exception as err:  # noqa: BLE001
        return ("error", type(err).__name__)
    if hasattr(result, "keys"):
        return ("ok", list(result.keys()), {k: _normalize(result[k]) for k in result.keys()})
    return ("ok", _normalize(result))


def offline_pair():
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(BERT_UNCASED),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=512,
    )
    return hf, nt.Tokenizer(hf).as_hf()


def hub_pair(repo_id):
    hf = transformers.AutoTokenizer.from_pretrained(repo_id)
    return hf, nt.Tokenizer(repo_id).as_hf()


CALL_GRID = list(
    itertools.product(
        [False, True, "longest", "max_length"],
        [None, True, False, "only_first", "only_second"],
        [None, 1, 2, 3, 16],
        [None, 8],
        [None, "np", "pt"],
    )
)


def check_calls(hf, ours, texts):
    mismatches = []
    for padding, truncation, max_length, multiple, tensors in CALL_GRID:
        for side in ("right", "left"):
            hf.padding_side = ours.padding_side = side
            hf.truncation_side = ours.truncation_side = side
            for inputs in (texts, texts[0]):
                kwargs = dict(
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    pad_to_multiple_of=multiple,
                    return_tensors=tensors,
                )
                got = outcome(ours, inputs, **kwargs)
                truncating = truncation not in (None, False) or (truncation is None and padding is False)
                if truncating and max_length is not None and max_length < ours.num_special_tokens_to_add():
                    # HF keeps one whole word here; nanotoken refuses instead.
                    if got != ("error", "ValueError"):
                        mismatches.append((kwargs, "expected ValueError", got[:2]))
                    continue
                expected = outcome(hf, inputs, **kwargs)
                if expected == ("error", "Exception") and got[0] == "error":
                    continue
                if expected != got:
                    mismatches.append((kwargs, side, isinstance(inputs, str), expected[:2], got[:2]))
    hf.padding_side = ours.padding_side = "right"
    hf.truncation_side = ours.truncation_side = "right"
    assert not mismatches, mismatches[:3]


def check_call_options(hf, ours, texts):
    for kwargs in [
        dict(add_special_tokens=False, padding=True),
        dict(return_special_tokens_mask=True, padding=True),
        dict(return_length=True, padding="max_length", max_length=24, truncation=True),
        dict(return_length=True),
        dict(return_attention_mask=False, return_token_type_ids=False, padding=True, return_tensors="np"),
        dict(padding=True, padding_side="left", return_tensors="pt"),
        dict(padding="max_length", max_length=12),
        dict(max_length=5),
    ]:
        assert outcome(ours, texts, **kwargs) == outcome(hf, texts, **kwargs), kwargs
    for kwargs in [
        dict(),
        dict(add_special_tokens=False),
        dict(max_length=4, truncation=True),
        dict(return_tensors="pt"),
    ]:
        assert outcome(ours.encode, texts[1], **kwargs) == outcome(hf.encode, texts[1], **kwargs)
    for bad in [dict(text_pair="x"), dict(return_offsets_mapping=True), dict(is_split_into_words=True)]:
        with pytest.raises(NotImplementedError):
            ours(texts, **bad)


def check_decoding(hf, ours, texts):
    batch = hf(texts, padding=True)["input_ids"]
    for skip in (False, True):
        for cleanup in (None, True, False):
            kw = dict(skip_special_tokens=skip, clean_up_tokenization_spaces=cleanup)
            assert ours.batch_decode(batch, **kw) == hf.batch_decode(batch, **kw)
            assert ours.decode(batch[0], **kw) == hf.decode(batch[0], **kw)
            assert ours.decode(batch, **kw) == hf.decode(batch, **kw)
            assert ours.decode(np.array(batch[1]), **kw) == hf.decode(np.array(batch[1]), **kw)
            assert ours.decode(torch.tensor(batch[2]), **kw) == hf.decode(torch.tensor(batch[2]), **kw)
    for text in texts:
        assert ours.tokenize(text) == hf.tokenize(text)
        assert ours.tokenize(text, add_special_tokens=True) == hf.tokenize(text, add_special_tokens=True)
        tokens = hf.tokenize(text)
        assert ours.convert_tokens_to_ids(tokens) == hf.convert_tokens_to_ids(tokens)
        assert ours.convert_tokens_to_string(tokens) == hf.convert_tokens_to_string(tokens)
    ids = batch[0]
    for skip in (False, True):
        assert ours.convert_ids_to_tokens(ids, skip_special_tokens=skip) == hf.convert_ids_to_tokens(
            ids, skip_special_tokens=skip
        )
    assert ours.convert_ids_to_tokens(ids[1]) == hf.convert_ids_to_tokens(ids[1])
    assert ours.convert_tokens_to_ids("not-a-token-at-all") == hf.convert_tokens_to_ids("not-a-token-at-all")


def check_attributes(hf, ours):
    for attr in [
        "vocab_size",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "model_input_names",
        "special_tokens_map",
        "all_special_tokens",
        "all_special_ids",
        "unk_token",
        "unk_token_id",
        "sep_token_id",
        "pad_token_id",
        "cls_token_id",
        "mask_token_id",
        "bos_token",
        "eos_token_id",
        "pad_token_type_id",
    ]:
        assert getattr(ours, attr) == getattr(hf, attr), attr
    assert len(ours) == len(hf)
    assert ours.get_vocab() == hf.get_vocab()
    assert ours.get_added_vocab() == hf.get_added_vocab()
    assert ours.num_special_tokens_to_add() == hf.num_special_tokens_to_add()


def check_pad(hf, ours, texts):
    features = [hf(t) for t in texts[:6]]
    for kwargs in [
        dict(),
        dict(return_tensors="pt"),
        dict(padding="max_length", max_length=64),
        dict(pad_to_multiple_of=8),
    ]:
        assert outcome(ours.pad, features, **kwargs) == outcome(hf.pad, features, **kwargs), kwargs
    no_mask = [{"input_ids": f["input_ids"]} for f in features]
    assert outcome(ours.pad, no_mask) == outcome(hf.pad, no_mask)


@pytest.fixture(scope="module")
def offline():
    return offline_pair()


def test_offline_calls(offline):
    check_calls(*offline, TEXTS[:8])


def test_offline_call_options(offline):
    check_call_options(*offline, TEXTS)


def test_offline_decoding(offline):
    check_decoding(*offline, TEXTS)


def test_offline_attributes(offline):
    check_attributes(*offline)


def test_offline_pad(offline):
    check_pad(*offline, TEXTS)


HUB = [
    "google-bert/bert-base-uncased",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google-bert/bert-base-chinese",
]


@pytest.mark.network
@pytest.mark.parametrize("repo_id", HUB)
def test_hub_compat(repo_id):
    hf, ours = hub_pair(repo_id)
    check_attributes(hf, ours)
    check_call_options(hf, ours, TEXTS)
    check_decoding(hf, ours, TEXTS)
    check_pad(hf, ours, TEXTS)


@pytest.mark.network
def test_hub_calls_grid():
    check_calls(*hub_pair("BAAI/bge-small-en-v1.5"), TEXTS[:8])


def test_batch_encoding_to_device(offline):
    _, ours = offline
    batch = ours(["a", "b c"], padding=True, return_tensors="pt").to("cpu")
    assert batch.input_ids.device.type == "cpu"
    assert list(batch) == ours.model_input_names
