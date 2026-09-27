from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from nanotoken._load import Source, load_source, read_vocab_txt, vocab_to_tokenizer_json
from nanotoken._parallel import resolve_parallel
from nanotoken._ragged import RaggedIds
from nanotoken.nanotoken_rs import JsonlFileSource, TextFileSource, WordPieceTokenizer

if TYPE_CHECKING:
    from nanotoken._hf_compat import HFCompat

Texts = Sequence[str] | Sequence[bytes] | Iterable[str] | Iterable[bytes]
FileSourceLike = TextFileSource | JsonlFileSource | str | os.PathLike[str] | Sequence[str | os.PathLike[str]]


class Tokenizer:
    """A WordPiece tokenizer with exact HuggingFace `tokenizers` output.

    Construct it from a Hub repo id (e.g. "BAAI/bge-small-en-v1.5"), a path to
    a tokenizer.json, vocab.txt or a directory holding one, a
    `tokenizers.Tokenizer`, or a transformers fast tokenizer. Use `as_hf()`
    for a drop-in replacement of the transformers tokenizer API.

    Batch methods take `parallel=None` by default: parallel on a shared
    thread pool, except inside multiprocessing workers and forked children,
    which encode on the calling thread. Output is identical either way.
    """

    def __init__(self, source: Any, *, revision: str = "main") -> None:
        if isinstance(source, Tokenizer):
            self._source = source._source
            self._backend = source._backend
            return
        self._source = source if isinstance(source, Source) else load_source(source, revision)
        self._backend = WordPieceTokenizer.from_json(self._source.tokenizer_json)

    @classmethod
    def from_json(cls, data: str | bytes, config: dict[str, Any] | None = None) -> Tokenizer:
        """Load from tokenizer.json contents."""
        text = data.decode() if isinstance(data, bytes) else data
        return cls(Source(text, dict(config or {})))

    @classmethod
    def from_vocab(cls, path: str | os.PathLike[str], *, lowercase: bool = True, **kwargs: Any) -> Tokenizer:
        """Build a BERT tokenizer from a vocab.txt, as transformers' BertTokenizer does."""
        return cls(Source(vocab_to_tokenizer_json(read_vocab_txt(path), lowercase=lowercase, **kwargs)))

    @property
    def backend(self) -> WordPieceTokenizer:
        return self._backend

    @property
    def config(self) -> dict[str, Any]:
        """transformers-side settings (tokenizer_config.json) this tokenizer was loaded with."""
        return self._source.config

    def to_json(self) -> str:
        return self._source.tokenizer_json

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary including added tokens."""
        return self._backend.get_vocab_size(True)

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        return self._backend.get_vocab(with_added_tokens)

    def token_to_id(self, token: str) -> int | None:
        return self._backend.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self._backend.id_to_token(id)

    @property
    def pad_token_id(self) -> int | None:
        return self._backend.token_to_id(self.config.get("pad_token", "[PAD]"))

    def encode(self, text: str | bytes, *, add_special_tokens: bool = True) -> npt.NDArray[np.uint32]:
        return self._backend.encode(text, add_special_tokens)

    def encode_batch(
        self,
        texts: Texts,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation_side: Literal["right", "left"] = "right",
        parallel: bool | None = None,
    ) -> RaggedIds:
        """Encode documents into a RaggedIds (flat ids + row offsets).
        `max_length` truncates each row, special tokens included."""
        ids, offsets = self._backend.encode_batch(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation_side=truncation_side,
            parallel=resolve_parallel(parallel),
        )
        return RaggedIds(ids, offsets)

    def encode_batch_list(
        self,
        texts: Texts,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation_side: Literal["right", "left"] = "right",
        parallel: bool | None = None,
    ) -> list[list[int]]:
        return self._backend.encode_batch_list(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation_side=truncation_side,
            parallel=resolve_parallel(parallel),
        )

    def encode_batch_padded(
        self,
        texts: Texts,
        *,
        max_length: int | None = None,
        truncation: bool = True,
        padding: Literal["longest", "max_length"] = "longest",
        pad_to_multiple_of: int | None = None,
        padding_side: Literal["right", "left"] = "right",
        truncation_side: Literal["right", "left"] = "right",
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
        return_special_tokens_mask: bool = False,
        parallel: bool | None = None,
    ) -> dict[str, npt.NDArray[np.int64]]:
        """Model-ready int64 (rows x width) arrays: `input_ids` plus
        `token_type_ids` / `attention_mask` / `special_tokens_mask` as
        requested. Truncation to `max_length` applies when it is set."""
        pad_id = self.pad_token_id
        if pad_id is None:
            raise ValueError("this tokenizer has no pad token")
        out = self._backend.encode_batch_padded(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation and max_length is not None,
            truncation_side=truncation_side,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            pad_id=pad_id,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_special_tokens_mask=return_special_tokens_mask,
            parallel=resolve_parallel(parallel),
        )
        if not isinstance(out["input_ids"], np.ndarray):
            raise ValueError(
                f"some documents are longer than max_length={max_length}; pass truncation=True to get a rectangular batch"
            )
        return out

    def encode_files(
        self,
        source: FileSourceLike,
        *,
        add_special_tokens: bool = True,
        parallel: bool | None = None,
    ) -> RaggedIds:
        """Encode every document in files: a TextFileSource (optionally split
        on a separator), a JsonlFileSource, or path(s) of whole-file documents.
        Files are memory-mapped and read entirely in Rust."""
        ids, offsets = self._backend.encode_files(
            source, add_special_tokens=add_special_tokens, parallel=resolve_parallel(parallel)
        )
        return RaggedIds(ids, offsets)

    def decode(self, ids: Sequence[int] | npt.NDArray[np.integer], *, skip_special_tokens: bool = False) -> str:
        return self._backend.decode(_as_ids(ids), skip_special_tokens)

    def decode_batch(self, sequences: Iterable[Any], *, skip_special_tokens: bool = False) -> list[str]:
        return self._backend.decode_batch([_as_ids(s) for s in sequences], skip_special_tokens)

    def tokenize(self, text: str | bytes) -> list[str]:
        """Token strings of `text`, without special tokens."""
        return self._backend.tokenize(text)

    def as_hf(self) -> HFCompat:
        """A drop-in for the transformers fast-tokenizer API backed by this tokenizer."""
        from nanotoken._hf_compat import HFCompat

        return HFCompat(self)

    def clear_cache(self) -> None:
        self._backend.clear_cache()

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"


def _as_ids(ids: Any) -> Any:
    if isinstance(ids, (list, np.ndarray)):
        return ids
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return list(ids)
