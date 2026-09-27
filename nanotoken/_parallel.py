from __future__ import annotations

import multiprocessing
import os

_forked_child = False


def _mark_forked_child() -> None:
    global _forked_child
    _forked_child = True


# A rayon pool that existed before os.fork() has no worker threads in the
# child, so the first parallel encode there would wait forever. Forked
# children (and spawn/forkserver workers, which would otherwise each size a
# pool to every core) therefore default to encoding on the calling thread.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_mark_forked_child)


def in_worker_process() -> bool:
    return _forked_child or multiprocessing.parent_process() is not None


def resolve_parallel(parallel: bool | None) -> bool:
    return not in_worker_process() if parallel is None else parallel
