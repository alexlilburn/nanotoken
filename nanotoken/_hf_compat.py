"""HFCompat: a nanotoken-backed drop-in for the transformers fast-tokenizer API."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nanotoken._tokenizer import Tokenizer

VERY_LARGE_INTEGER = int(1e30)
LARGE_INTEGER = int(1e20)

SPECIAL_TOKEN_ATTRIBUTES = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")
BERT_DEFAULTS = {
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}
PADDING_STRATEGIES = ("longest", "max_length", "do_not_pad")
TRUNCATION_STRATEGIES = ("longest_first", "only_first", "only_second", "do_not_truncate")


def _cleanup_spaces(text: str) -> str:
    return (
        text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _to_py(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_py(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@dataclass(frozen=True)
class AddedToken:
    content: str
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True
    special: bool = False

    def __str__(self) -> str:
        return self.content


class BatchEncoding(dict):
    """dict of model inputs with attribute access, like transformers.BatchEncoding."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getstate__(self) -> dict[str, Any]:
        return {"data": dict(self)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.update(state["data"])

    def to(self, device: Any, *, non_blocking: bool = False) -> BatchEncoding:
        for key, value in self.items():
            if hasattr(value, "to"):
                self[key] = value.to(device=device, non_blocking=non_blocking)
        return self

    def convert_to_tensors(self, tensor_type: Any = None) -> BatchEncoding:
        for key, value in self.items():
            self[key] = _as_tensor(value, tensor_type)
        return self


def _as_tensor(value: Any, return_tensors: Any) -> Any:
    kind = _enum_value(return_tensors)
    if kind is None:
        return value
    if kind not in ("np", "pt"):
        raise NotImplementedError(f'nanotoken supports return_tensors="np" and "pt", got {kind!r}')
    if isinstance(value, np.ndarray):
        array = value
    elif value and isinstance(value[0], list) and len({len(row) for row in value}) > 1:
        if kind == "pt":
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or padding with "
                "'padding=True' 'truncation=True' to have batched tensors with the same length."
            )
        array = np.empty(len(value), dtype=object)
        array[:] = [np.asarray(row, dtype=np.int64) for row in value]
        return array
    else:
        array = np.asarray(value, dtype=np.int64)
    if kind == "np":
        return array
    import torch

    return torch.from_numpy(array)


class HFCompat:
    """Drop-in for a transformers WordPiece fast tokenizer (e.g. BertTokenizer):
    `__call__`, `encode`, `tokenize`, `decode`, `batch_decode`, `pad`, token/id
    conversion and the special-token attributes, all producing identical
    output. Sentence pairs, offset mappings, overflowing tokens and
    pre-tokenized input raise NotImplementedError."""

    is_fast = True
    pad_token_type_id = 0

    bos_token: str | None
    eos_token: str | None
    unk_token: str | None
    sep_token: str | None
    pad_token: str | None
    cls_token: str | None
    mask_token: str | None
    bos_token_id: int | None
    eos_token_id: int | None
    unk_token_id: int | None
    sep_token_id: int | None
    pad_token_id: int | None
    cls_token_id: int | None
    mask_token_id: int | None

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self._backend = tokenizer.backend
        config = tokenizer.config
        self.model_max_length: int = int(config.get("model_max_length", VERY_LARGE_INTEGER))
        self.padding_side: str = config.get("padding_side", "right")
        self.truncation_side: str = config.get("truncation_side", "right")
        self.clean_up_tokenization_spaces: bool = bool(config.get("clean_up_tokenization_spaces", False))
        self.model_input_names: list[str] = list(
            config.get("model_input_names") or ["input_ids", "token_type_ids", "attention_mask"]
        )
        self.split_special_tokens = False
        self.name_or_path = config.get("name_or_path", "")
        self._special: dict[str, str | None] = {}
        for attr in SPECIAL_TOKEN_ATTRIBUTES:
            value = config.get(attr, BERT_DEFAULTS.get(attr))
            self._special[attr] = value if value is not None and self._backend.token_to_id(value) is not None else None

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tok

    @cached_property
    def _json(self) -> dict[str, Any]:
        return json.loads(self._tok.to_json())

    def __repr__(self) -> str:
        return f"HFCompat(vocab_size={self.vocab_size}, model_max_length={self.model_max_length})"

    def _strategies(
        self, padding: Any, truncation: Any, max_length: int | None, pad_to_multiple_of: int | None
    ) -> tuple[str, str, int | None]:
        padding, truncation = _enum_value(padding), _enum_value(truncation)
        if max_length is not None and padding is False and truncation is None:
            truncation = "longest_first"
        if padding is True:
            padding_strategy = "longest"
        elif padding is False or padding is None:
            padding_strategy = "do_not_pad"
        elif padding in PADDING_STRATEGIES:
            padding_strategy = padding
        else:
            raise ValueError(f"{padding!r} is not a valid PaddingStrategy, please select one of {PADDING_STRATEGIES}")
        if truncation is True:
            truncation_strategy = "longest_first"
        elif truncation is False or truncation is None:
            truncation_strategy = "do_not_truncate"
        elif truncation in TRUNCATION_STRATEGIES:
            truncation_strategy = truncation
        else:
            raise ValueError(
                f"{truncation!r} is not a valid TruncationStrategy, please select one of {TRUNCATION_STRATEGIES}"
            )
        if max_length is None:
            if padding_strategy == "max_length":
                if self.model_max_length > LARGE_INTEGER:
                    padding_strategy = "do_not_pad"
                else:
                    max_length = self.model_max_length
            if truncation_strategy != "do_not_truncate":
                if self.model_max_length > LARGE_INTEGER:
                    truncation_strategy = "do_not_truncate"
                else:
                    max_length = self.model_max_length
        if padding_strategy != "do_not_pad" and self.pad_token is None:
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token`."
            )
        if (
            truncation_strategy != "do_not_truncate"
            and padding_strategy != "do_not_pad"
            and pad_to_multiple_of is not None
            and max_length is not None
            and max_length % pad_to_multiple_of != 0
        ):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )
        return padding_strategy, truncation_strategy, max_length

    def __call__(
        self,
        text: str | Sequence[str] | None = None,
        text_pair: Any = None,
        text_target: Any = None,
        text_pair_target: Any = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        if tokenizer_kwargs:
            merged = {"padding": padding, "truncation": truncation, "max_length": max_length, **tokenizer_kwargs}
            padding, truncation, max_length = merged["padding"], merged["truncation"], merged["max_length"]
        unsupported = {
            "text_pair": text_pair is not None,
            "text_target": text_target is not None,
            "text_pair_target": text_pair_target is not None,
            "is_split_into_words": is_split_into_words,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_offsets_mapping": return_offsets_mapping,
            "split_special_tokens": bool(kwargs.pop("split_special_tokens", False)),
        }
        for name, used in unsupported.items():
            if used:
                raise NotImplementedError(f"nanotoken.HFCompat does not support {name}")
        if text is None:
            raise ValueError("You need to specify either `text` or `text_target`.")
        batched = not isinstance(text, str)
        texts = [text] if not batched else list(text)
        if not all(isinstance(t, str) for t in texts):
            raise ValueError(
                "text input must be of type `str` (single example) or `list[str]` (batch); "
                "nanotoken.HFCompat does not support pre-tokenized input"
            )
        padding_strategy, truncation_strategy, max_length = self._strategies(
            padding, truncation, max_length, pad_to_multiple_of
        )
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        truncate = truncation_strategy != "do_not_truncate" and max_length is not None
        if truncate and truncation_strategy == "only_second":
            added = self.num_special_tokens_to_add() if add_special_tokens else 0
            if max_length > added:
                offsets = self._backend.encode_batch(texts, add_special_tokens=add_special_tokens)[1]
                if (np.diff(offsets) > max_length).any():
                    raise ValueError("Truncation error: Second sequence not provided")
                truncate = False
        tensors = _enum_value(return_tensors)
        out = self._backend.encode_batch_padded(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncate,
            truncation_side=self.truncation_side,
            padding=None if padding_strategy == "do_not_pad" else padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side or self.padding_side,
            pad_id=self.pad_token_id if self.pad_token_id is not None else 0,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_special_tokens_mask=return_special_tokens_mask,
            as_lists=tensors is None,
        )
        if return_length:
            ids = out["input_ids"]
            out["length"] = [len(row) for row in ids] if isinstance(ids, list) else [ids.shape[1]] * ids.shape[0]
        if tensors is not None:
            return BatchEncoding({k: _as_tensor(v, tensors) for k, v in out.items()})
        if not batched:
            return BatchEncoding({k: v[0] for k, v in out.items()})
        return BatchEncoding(out)

    def encode_plus(self, text: str, *args: Any, **kwargs: Any) -> BatchEncoding:
        return self(text, *args, **kwargs)

    def batch_encode_plus(self, batch_text_or_text_pairs: Sequence[str], *args: Any, **kwargs: Any) -> BatchEncoding:
        return self(list(batch_text_or_text_pairs), *args, **kwargs)

    def encode(
        self,
        text: str,
        text_pair: Any = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str | None = None,
        max_length: int | None = None,
        stride: int = 0,
        padding_side: str | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            padding_side=padding_side,
            return_tensors=return_tensors,
            **kwargs,
        )["input_ids"]

    def tokenize(
        self, text: str, pair: str | None = None, add_special_tokens: bool = False, **kwargs: Any
    ) -> list[str]:
        if pair is not None:
            raise NotImplementedError("nanotoken.HFCompat does not support sentence pairs")
        tokens = self._backend.tokenize(text)
        if add_special_tokens:
            prefix = [self._backend.id_to_token(i) for i in self._backend.prefix_ids]
            suffix = [self._backend.id_to_token(i) for i in self._backend.suffix_ids]
            tokens = prefix + tokens + suffix
        return tokens

    def pad(
        self,
        encoded_inputs: Any,
        padding: bool | str = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | None = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """Pad already-encoded inputs (a dict of lists, or a list of dicts as
        produced by a dataset), as transformers' `pad`."""
        if isinstance(encoded_inputs, Sequence) and encoded_inputs and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0]}
        encoded_inputs = {k: _to_py(v) for k, v in dict(encoded_inputs).items()}
        if "input_ids" not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method that includes input_ids"
            )
        ids = encoded_inputs["input_ids"]
        batched = bool(ids) and isinstance(ids[0], list)
        if not batched:
            encoded_inputs = {k: [v] for k, v in encoded_inputs.items()}
        padding_strategy, _, max_length = self._strategies(padding, False, max_length, pad_to_multiple_of)
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        rows = encoded_inputs["input_ids"]
        if padding_strategy == "longest":
            max_length = max((len(r) for r in rows), default=0)
        if max_length is not None and pad_to_multiple_of and max_length % pad_to_multiple_of:
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        side = padding_side or self.padding_side
        fill = {
            "input_ids": self.pad_token_id,
            "token_type_ids": self.pad_token_type_id,
            "attention_mask": 0,
            "special_tokens_mask": 1,
        }
        out: dict[str, list[Any]] = {k: [] for k in encoded_inputs}
        if return_attention_mask and "attention_mask" not in out:
            out["attention_mask"] = []
        for i, row in enumerate(rows):
            n = len(row)
            missing = max(0, max_length - n) if padding_strategy != "do_not_pad" and max_length is not None else 0
            for key in out:
                values = encoded_inputs[key][i] if key in encoded_inputs else [1] * n
                if key == "attention_mask" and key not in encoded_inputs:
                    values = [1] * n
                if missing and key in fill:
                    pad = [fill[key]] * missing
                    values = values + pad if side == "right" else pad + values
                out[key].append(values)
        result = {k: v if batched else v[0] for k, v in out.items()}
        if return_tensors is not None:
            return BatchEncoding({k: _as_tensor(v, return_tensors) for k, v in result.items()})
        return BatchEncoding(result)

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        token_ids = _to_py(token_ids)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            return [self.decode(seq, skip_special_tokens, clean_up_tokenization_spaces) for seq in token_ids]
        text = self._backend.decode(token_ids, skip_special_tokens)
        cleanup = (
            self.clean_up_tokenization_spaces if clean_up_tokenization_spaces is None else clean_up_tokenization_spaces
        )
        return _cleanup_spaces(text) if cleanup else text

    def batch_decode(
        self,
        sequences: Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: Any,
    ) -> list[str]:
        result = self.decode(_to_py(sequences), skip_special_tokens, clean_up_tokenization_spaces)
        return [result] if isinstance(result, str) else result

    def convert_ids_to_tokens(self, ids: int | Iterable[int], skip_special_tokens: bool = False) -> Any:
        ids = _to_py(ids)
        if isinstance(ids, int):
            return self._backend.id_to_token(ids)
        skip = set(self.all_special_ids) if skip_special_tokens else set()
        return [self._backend.id_to_token(int(i)) for i in ids if int(i) not in skip]

    def convert_tokens_to_ids(self, tokens: str | Iterable[str] | None) -> Any:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._token_to_id(tokens)
        return [self._token_to_id(t) for t in tokens]

    def _token_to_id(self, token: str) -> int | None:
        index = self._backend.token_to_id(token)
        return self.unk_token_id if index is None else index

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        decoder = self._json.get("decoder")
        if not decoder:
            return " ".join(tokens)
        prefix, cleanup = decoder.get("prefix", "##"), decoder.get("cleanup", True)
        out = []
        for i, token in enumerate(tokens):
            if i:
                token = token[len(prefix) :] if token.startswith(prefix) else " " + token
            out.append(
                token.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" do not", " don't")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
                if cleanup
                else token
            )
        return "".join(out)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if pair:
            raise NotImplementedError("nanotoken.HFCompat does not support sentence pairs")
        return len(self._backend.prefix_ids) + len(self._backend.suffix_ids)

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(False)

    def __len__(self) -> int:
        return self._backend.get_vocab_size(True)

    def get_vocab(self) -> dict[str, int]:
        return self._backend.get_vocab(True)

    @property
    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        return {
            t["id"]: AddedToken(t["content"], t["single_word"], t["lstrip"], t["rstrip"], t["normalized"], t["special"])
            for t in self._backend.added_tokens
        }

    @property
    def added_tokens_encoder(self) -> dict[str, int]:
        return {t.content: i for i, t in sorted(self.added_tokens_decoder.items())}

    def get_added_vocab(self) -> dict[str, int]:
        return self.added_tokens_encoder

    @property
    def special_tokens_map(self) -> dict[str, str]:
        return {k: v for k, v in self._special.items() if v is not None}

    @property
    def all_special_tokens(self) -> list[str]:
        tokens = list(dict.fromkeys(self.special_tokens_map.values()))
        for token in self._backend.added_tokens:
            if token["special"] and token["content"] not in tokens:
                tokens.append(token["content"])
        return tokens

    @property
    def all_special_ids(self) -> list[int]:
        return [self._backend.token_to_id(t) for t in self.all_special_tokens]


def _special_token_property(attr: str) -> tuple[property, property]:
    def get_token(self: HFCompat) -> str | None:
        return self._special.get(attr)

    def set_token(self: HFCompat, value: str | None) -> None:
        self._special[attr] = None if value is None else str(value)

    def get_id(self: HFCompat) -> int | None:
        token = self._special.get(attr)
        return None if token is None else self._backend.token_to_id(token)

    return property(get_token, set_token), property(get_id)


for _attr in SPECIAL_TOKEN_ATTRIBUTES:
    _token_prop, _id_prop = _special_token_property(_attr)
    setattr(HFCompat, _attr, _token_prop)
    setattr(HFCompat, f"{_attr}_id", _id_prop)
