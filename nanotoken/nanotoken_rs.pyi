import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

_Paths = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]

class UnsupportedTokenizerError(ValueError): ...

def set_cache_entries(entries: int) -> None: ...
def get_cache_entries() -> int: ...

class TextFileSource:
    paths: list[str]
    separator: bytes | None
    def __init__(self, paths: _Paths, separator: str | bytes | None = None) -> None: ...

class JsonlFileSource:
    paths: list[str]
    field: str
    def __init__(self, paths: _Paths, field: str = "text") -> None: ...

class WordPieceTokenizer:
    @staticmethod
    def from_json(data: str | bytes) -> WordPieceTokenizer: ...
    def encode(self, text: str | bytes, add_special_tokens: bool = True) -> npt.NDArray[np.uint32]: ...
    def encode_batch(
        self,
        inputs: Iterable[str] | Iterable[bytes],
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation_side: str = "right",
        parallel: bool = True,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]: ...
    def encode_batch_list(
        self,
        inputs: Iterable[str] | Iterable[bytes],
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation_side: str = "right",
        parallel: bool = True,
    ) -> list[list[int]]: ...
    def encode_batch_padded(
        self,
        inputs: Iterable[str] | Iterable[bytes],
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
        truncation_side: str = "right",
        padding: str | None = None,
        pad_to_multiple_of: int | None = None,
        padding_side: str = "right",
        pad_id: int = 0,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
        return_special_tokens_mask: bool = False,
        as_lists: bool = False,
        parallel: bool = True,
    ) -> dict[str, Any]: ...
    def encode_files(
        self,
        source: TextFileSource | JsonlFileSource | _Paths,
        *,
        add_special_tokens: bool = True,
        parallel: bool = True,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]: ...
    def decode(self, ids: Sequence[int] | npt.NDArray[np.integer], skip_special_tokens: bool = False) -> str: ...
    def decode_batch(self, sequences: Iterable[Any], skip_special_tokens: bool = False) -> list[str]: ...
    def tokenize(self, text: str | bytes) -> list[str]: ...
    def token_to_id(self, token: str) -> int | None: ...
    def id_to_token(self, id: int) -> str | None: ...
    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]: ...
    def get_vocab_size(self, with_added_tokens: bool = True) -> int: ...
    @property
    def added_tokens(self) -> list[dict[str, Any]]: ...
    @property
    def prefix_ids(self) -> list[int]: ...
    @property
    def suffix_ids(self) -> list[int]: ...
    @property
    def unk_token(self) -> str: ...
    @property
    def continuing_subword_prefix(self) -> str: ...
    def clear_cache(self) -> None: ...
