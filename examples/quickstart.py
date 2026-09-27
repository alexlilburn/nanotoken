"""Encode, batch, pad and decode with the native nanotoken API."""

import nanotoken as nt

tok = nt.Tokenizer("BAAI/bge-small-en-v1.5")

print(tok.encode("Hello, world!"))
print(tok.tokenize("unaffable tokenization"))

batch = tok.encode_batch(["The quick brown fox.", "Jumps over the lazy dog!"])
print(batch, batch[0], batch.to_list())

inputs = tok.encode_batch_padded(["short", "a somewhat longer sentence"], max_length=512)
print(inputs["input_ids"].shape, inputs["attention_mask"])

print(tok.decode(batch[1], skip_special_tokens=True))
