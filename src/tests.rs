use serde_json::Value;

use crate::batch::encode_fragmented;
use crate::tokenizer::Tokenizer;

const BERT_UNCASED: &[u8] = include_bytes!("../tests/fixtures/bert-base-uncased.json");
const ADVERSARIAL: &str = include_str!("../tests/fixtures/adversarial.txt");

fn adversarial() -> Vec<String> {
    ADVERSARIAL
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn variant(edit: impl FnOnce(&mut Value)) -> Tokenizer {
    let mut json: Value = serde_json::from_slice(BERT_UNCASED).unwrap();
    edit(&mut json);
    Tokenizer::from_json(&serde_json::to_vec(&json).unwrap()).unwrap()
}

fn variants() -> Vec<(&'static str, Tokenizer)> {
    vec![
        ("uncased", variant(|_| {})),
        (
            "cased",
            variant(|j| {
                j["normalizer"]["lowercase"] = false.into();
            }),
        ),
        (
            "no-clean-text",
            variant(|j| {
                j["normalizer"]["clean_text"] = false.into();
                j["normalizer"]["handle_chinese_chars"] = false.into();
            }),
        ),
        (
            "mask-lstrip-rstrip",
            variant(|j| {
                j["added_tokens"][4]["lstrip"] = true.into();
                j["added_tokens"][3]["rstrip"] = true.into();
                j["added_tokens"][2]["single_word"] = true.into();
            }),
        ),
        (
            "nfkc-whitespace",
            variant(|j| {
                j["normalizer"] = serde_json::json!({"type": "Sequence", "normalizers": [
                    {"type": "NFKC"}, {"type": "Lowercase"}, {"type": "StripAccents"}]});
                j["pre_tokenizer"] = serde_json::json!({"type": "Sequence", "pretokenizers": [
                    {"type": "Whitespace"}, {"type": "Digits", "individual_digits": true}]});
            }),
        ),
        (
            "punctuation-only",
            variant(|j| {
                j["pre_tokenizer"] = serde_json::json!({"type": "Punctuation"});
            }),
        ),
    ]
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn text(&mut self, len: usize) -> String {
        const EXTRA: &[char] = &[
            'é', 'ß', 'Σ', 'ς', 'İ', '中', '文', '😀', '\u{0301}', '\u{200b}', '\u{00a0}', '\u{3000}', '«', '—',
            '\u{0b}', '\u{0c}', '\u{0}', '\u{7f}', '\u{fffd}', 'ﬁ', '²', '½',
        ];
        (0..len)
            .map(|_| match self.next() % 10 {
                0..=5 => (b'a' + (self.next() % 26) as u8) as char,
                6 => [' ', ' ', '\n', '\t', '\r'][(self.next() % 5) as usize],
                7 => b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789"[(self.next() % 42) as usize] as char,
                8 => (b'A' + (self.next() % 26) as u8) as char,
                _ => EXTRA[(self.next() % EXTRA.len() as u64) as usize],
            })
            .collect()
    }
}

fn corpus() -> Vec<String> {
    let mut docs = adversarial();
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for i in 0..300 {
        docs.push(rng.text(1 + i % 97));
    }
    docs.push(docs.join(" [MASK] "));
    docs
}

#[test]
fn cached_chunks_match_reference_path() {
    for (name, tok) in variants() {
        let mut enc = tok.encoder();
        for doc in corpus() {
            let mut reference = Vec::new();
            enc.encode_uncached(&doc, &mut reference);
            for pass in 0..2 {
                let mut fast = Vec::new();
                enc.encode(&doc, &mut fast);
                assert_eq!(fast, reference, "{name} pass {pass}: {doc:?}");
            }
        }
    }
}

#[test]
fn fragments_match_whole_document() {
    for (name, tok) in variants() {
        if !tok.chunkable() {
            continue;
        }
        let doc = corpus().join(" [SEP] ");
        let whole = tok.encode(&doc);
        for target in [7, 64, 1000] {
            assert_eq!(encode_fragmented(&tok, &doc, target), whole, "{name} target {target}");
        }
    }
}

#[test]
fn known_encodings() {
    let tok = variant(|_| {});
    assert_eq!(tok.encode("Hello, world!"), [7592, 1010, 2088, 999]);
    assert_eq!(tok.encode("unaffable [MASK]"), [14477, 20961, 3468, 103]);
    assert_eq!(tok.decode(&[101, 7592, 1010, 2088, 999, 102], true), "hello, world!");
    assert!(
        !variant(|j| {
            j["pre_tokenizer"] = serde_json::json!({"type": "Punctuation"});
        })
        .chunkable()
    );
}
