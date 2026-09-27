"""Resolve a tokenizer source into tokenizer.json contents plus transformers-style config."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HUB_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

CONFIG_KEYS = (
    "model_max_length",
    "padding_side",
    "truncation_side",
    "clean_up_tokenization_spaces",
    "model_input_names",
    "do_lower_case",
    "tokenizer_class",
)
NAMED_SPECIAL_TOKENS = ("unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token")


@dataclass
class Source:
    tokenizer_json: str
    config: dict[str, Any] = field(default_factory=dict)


def cache_dir() -> Path:
    root = os.environ.get("NANOTOKEN_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "nanotoken"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _hf_hub_cached(repo_id: str, filename: str, revision: str) -> Path | None:
    """A file already present in the huggingface_hub cache, if any."""
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"), "hub"
    )
    repo = Path(root) / ("models--" + repo_id.replace("/", "--"))
    ref = repo / "refs" / revision
    snapshot = ref.read_text().strip() if ref.is_file() else revision
    path = repo / "snapshots" / snapshot / filename
    return path if path.is_file() else None


def hub_file(repo_id: str, filename: str, revision: str = "main") -> Path | None:
    """Download `filename` from a Hub model repo (cached); None if the repo has no such file."""
    cached = _hf_hub_cached(repo_id, filename, revision)
    if cached is not None:
        return cached
    target = cache_dir() / repo_id.replace("/", "--") / revision / filename
    missing = target.with_name(target.name + ".missing")
    if target.is_file():
        return target
    if missing.is_file():
        return None
    url = f"{HUB_ENDPOINT}/{repo_id}/resolve/{revision}/{filename}"
    request = urllib.request.Request(url, headers={"User-Agent": "nanotoken"})
    token = _hf_token()
    if token and url.startswith("https://huggingface.co/"):
        request.add_header("Authorization", f"Bearer {token}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            missing.touch()
            return None
        if err.code in (401, 403):
            raise PermissionError(
                f"{repo_id}: access denied ({err.code}); set HF_TOKEN for gated/private repos"
            ) from err
        raise
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def _read_config(read: Any) -> dict[str, Any]:
    """transformers-relevant settings from tokenizer_config.json / special_tokens_map.json."""
    config: dict[str, Any] = {}
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        path = read(name)
        if path is None:
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in CONFIG_KEYS:
            if key in data:
                config[key] = data[key]
        for key in NAMED_SPECIAL_TOKENS:
            value = data.get(key)
            if isinstance(value, dict):
                value = value.get("content")
            if isinstance(value, str):
                config[key] = value
    return config


def vocab_to_tokenizer_json(
    vocab: list[str],
    *,
    lowercase: bool = True,
    strip_accents: bool | None = None,
    handle_chinese_chars: bool = True,
    unk_token: str = "[UNK]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    pad_token: str = "[PAD]",
    mask_token: str = "[MASK]",
) -> str:
    """The tokenizer.json transformers' BertTokenizer builds from a vocab.txt."""
    ids = {token: i for i, token in enumerate(vocab)}
    specials = [t for t in (pad_token, unk_token, cls_token, sep_token, mask_token) if t in ids]
    template = [
        {"SpecialToken": {"id": cls_token, "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": sep_token, "type_id": 0}},
    ]
    return json.dumps(
        {
            "version": "1.0",
            "added_tokens": [
                {
                    "id": ids[t],
                    "content": t,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for t in sorted(specials, key=ids.__getitem__)
            ],
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": True,
                "handle_chinese_chars": handle_chinese_chars,
                "strip_accents": strip_accents,
                "lowercase": lowercase,
            },
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": template,
                "pair": template,
                "special_tokens": {t: {"id": t, "ids": [ids[t]], "tokens": [t]} for t in (cls_token, sep_token)},
            },
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
            "model": {
                "type": "WordPiece",
                "unk_token": unk_token,
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": ids,
            },
        }
    )


def read_vocab_txt(path: str | os.PathLike[str]) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _from_vocab_dir(read: Any, config: dict[str, Any]) -> str | None:
    vocab = read("vocab.txt")
    if vocab is None:
        return None
    specials = {k: config[k] for k in ("unk_token", "cls_token", "sep_token", "pad_token", "mask_token") if k in config}
    return vocab_to_tokenizer_json(read_vocab_txt(vocab), lowercase=bool(config.get("do_lower_case", True)), **specials)


def _from_files(read: Any, where: str) -> Source:
    config = _read_config(read)
    path = read("tokenizer.json")
    if path is not None:
        return Source(Path(path).read_text(encoding="utf-8"), config)
    data = _from_vocab_dir(read, config)
    if data is None:
        raise FileNotFoundError(f"{where}: no tokenizer.json or vocab.txt found")
    return Source(data, config)


def _config_from_transformers(tok: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        if hasattr(tok, key):
            config[key] = getattr(tok, key)
    for key in NAMED_SPECIAL_TOKENS:
        value = getattr(tok, key, None)
        if value is not None:
            config[key] = str(value)
    return config


def load_source(obj: Any, revision: str = "main") -> Source:
    """Accepts a Hub repo id, a tokenizer.json / vocab.txt path, a directory, a
    `tokenizers.Tokenizer`, or a transformers fast tokenizer."""
    if isinstance(obj, (str, os.PathLike)):
        path = Path(obj).expanduser()
        if path.is_file():
            if path.suffix == ".txt":
                return Source(vocab_to_tokenizer_json(read_vocab_txt(path)))
            sibling = path.parent
            return Source(path.read_text(encoding="utf-8"), _read_config(lambda n: _existing(sibling / n)))
        if path.is_dir():
            return _from_files(lambda n: _existing(path / n), str(path))
        repo_id = os.fspath(obj)
        if repo_id.count("/") > 1 or repo_id.startswith((".", "/", "~")):
            raise FileNotFoundError(f"{repo_id}: no such file or directory")
        return _from_files(lambda n: hub_file(repo_id, n, revision), repo_id)
    backend = getattr(obj, "backend_tokenizer", None)
    if backend is not None:
        return Source(backend.to_str(), _config_from_transformers(obj))
    if hasattr(obj, "to_str") and hasattr(obj, "encode_batch"):
        return Source(obj.to_str())
    raise TypeError(
        f"cannot load a tokenizer from {type(obj).__name__}; pass a Hub repo id, a path to tokenizer.json, "
        "vocab.txt or a directory, a tokenizers.Tokenizer, or a transformers fast tokenizer"
    )


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None
