from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
BERT_UNCASED = FIXTURES / "bert-base-uncased.json"

HUB_MODELS = [
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-cased",
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-base-chinese",
    "distilbert/distilbert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
]

_EXTRA = (
    "éèêëàâäôöûüçñßÆØÅæøåΣσςΑαİıŞşЁёЖжЯя中文字词語日本語カタカナひらがな한국어"
    "😀👍🏽👨‍👩‍👧🇺🇸̧́̈​‌‍﻿  　 \u0085"
    "«»„“”‘’‹›¡¿—–…·•§¶†‡※‽ﬁﬂ²³½¼ＡＢ１２�\x00\x01\x07\x0b\x0c\x1c\x7f"
)


def adversarial() -> list[str]:
    return [json.loads(line) for line in (FIXTURES / "adversarial.txt").read_text(encoding="utf-8").split("\n") if line]


def fuzz(n: int = 400, seed: int = 1234) -> list[str]:
    rng = random.Random(seed)
    ascii_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    docs = []
    for i in range(n):
        chars = []
        for _ in range(rng.randint(0, 120)):
            r = rng.random()
            if r < 0.55:
                chars.append(rng.choice(ascii_pool))
            elif r < 0.7:
                chars.append(rng.choice(" \n\t\r  "))
            elif r < 0.9:
                chars.append(rng.choice(_EXTRA))
            else:
                cp = rng.randint(0x80, 0x2FFFF)
                if 0xD800 <= cp <= 0xDFFF:
                    cp = 0x41
                chars.append(chr(cp))
        if i % 17 == 0:
            chars.insert(rng.randint(0, len(chars)), rng.choice(["[MASK]", " [SEP] ", "[CLS]x", "[PAD]"]))
        docs.append("".join(chars))
    return docs


@pytest.fixture(scope="session")
def corpus() -> list[str]:
    docs = adversarial() + fuzz()
    docs.append(" ".join(docs))
    return docs


def hub_tokenizer_json(repo_id: str) -> str:
    from nanotoken._load import hub_file

    path = hub_file(repo_id, "tokenizer.json")
    if path is None:
        pytest.skip(f"{repo_id} has no tokenizer.json")
    return path.read_text(encoding="utf-8")
