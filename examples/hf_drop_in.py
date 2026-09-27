"""Swap nanotoken into code written against a transformers tokenizer and check
the tensors are identical."""

import torch
from transformers import AutoTokenizer

import nanotoken as nt

model_id = "sentence-transformers/all-MiniLM-L6-v2"
texts = ["nanotoken is a drop-in replacement.", "It returns exactly the same tensors!", "Short."]

hf = AutoTokenizer.from_pretrained(model_id)
fast = nt.Tokenizer(hf).as_hf()  # or nt.Tokenizer(model_id).as_hf()

expected = hf(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
got = fast(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

assert list(got) == list(expected)
for key in expected:
    assert torch.equal(got[key], expected[key]), key
print("identical:", {k: tuple(v.shape) for k, v in got.items()})
print(fast.batch_decode(got["input_ids"], skip_special_tokens=True))
