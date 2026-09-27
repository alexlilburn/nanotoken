from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

import numpy as np
import numpy.typing as npt


class RaggedIds:
    """Token ids of many documents: one flat uint32 buffer plus int64 row
    offsets, so a batch costs two numpy arrays instead of one Python object
    per document. Row `i` is `ids[offsets[i]:offsets[i + 1]]` (a view)."""

    __slots__ = ("ids", "offsets")

    def __init__(self, ids: npt.NDArray[np.uint32], offsets: npt.NDArray[np.int64]) -> None:
        self.ids = ids
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @overload
    def __getitem__(self, index: int) -> npt.NDArray[np.uint32]: ...
    @overload
    def __getitem__(self, index: slice) -> RaggedIds: ...
    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("RaggedIds only supports contiguous slices")
            stop = max(start, stop)
            base = self.offsets[start]
            return RaggedIds(self.ids[base : self.offsets[stop]], self.offsets[start : stop + 1] - base)
        n = len(self)
        i = index + n if index < 0 else index
        if not 0 <= i < n:
            raise IndexError(f"row {index} out of range for {n} rows")
        return self.ids[self.offsets[i] : self.offsets[i + 1]]

    def __iter__(self) -> Iterator[npt.NDArray[np.uint32]]:
        for i in range(len(self)):
            yield self.ids[self.offsets[i] : self.offsets[i + 1]]

    @property
    def lengths(self) -> npt.NDArray[np.int64]:
        return np.diff(self.offsets)

    @property
    def num_tokens(self) -> int:
        return int(self.offsets[-1])

    def to_list(self) -> list[list[int]]:
        flat = self.ids.tolist()
        bounds = self.offsets.tolist()
        return [flat[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]

    def to_awkward(self) -> Any:
        import awkward as ak

        return ak.unflatten(self.ids, self.lengths)

    def __repr__(self) -> str:
        return f"RaggedIds(rows={len(self)}, tokens={self.num_tokens})"
