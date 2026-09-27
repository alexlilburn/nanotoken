use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::trie::Trie;

#[derive(Clone, Debug)]
pub struct WordPiece {
    pub vocab: FxHashMap<String, u32>,
    pub unk_token: String,
    pub unk_id: u32,
    pub prefix: String,
    pub max_input_chars_per_word: usize,
    initial: Trie,
    continuing: Trie,
}

impl WordPiece {
    pub fn new(
        vocab: FxHashMap<String, u32>,
        unk_token: String,
        prefix: String,
        max_input_chars_per_word: usize,
    ) -> Result<Self> {
        let unk_id = *vocab.get(&unk_token).ok_or_else(|| {
            Error::Invalid(format!(
                "WordPiece unk_token {unk_token:?} is missing from the vocabulary"
            ))
        })?;
        let initial = Trie::new(vocab.iter().map(|(k, &v)| (k.as_bytes(), v)));
        let continuing = Trie::new(
            vocab
                .iter()
                .filter_map(|(k, &v)| k.strip_prefix(prefix.as_str()).map(|rest| (rest.as_bytes(), v))),
        );
        Ok(WordPiece {
            vocab,
            unk_token,
            unk_id,
            prefix,
            max_input_chars_per_word,
            initial,
            continuing,
        })
    }

    pub fn from_json(model: &Value) -> Result<Self> {
        let kind = model.get("type").and_then(Value::as_str);
        let looks_like_wordpiece =
            kind.is_none() && model.get("merges").is_none() && model.get("max_input_chars_per_word").is_some();
        if kind != Some("WordPiece") && !looks_like_wordpiece {
            return Err(Error::Unsupported(format!(
                "model type {} (nanotoken only implements WordPiece)",
                kind.map_or_else(|| "<untyped>".to_string(), |k| format!("{k:?}"))
            )));
        }
        let vocab_json = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Invalid("WordPiece model without a `vocab` object".into()))?;
        let mut vocab = FxHashMap::default();
        vocab.reserve(vocab_json.len());
        for (token, id) in vocab_json {
            let id = id
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| Error::Invalid(format!("invalid id for vocab entry {token:?}")))?;
            vocab.insert(token.clone(), id);
        }
        let unk_token = model
            .get("unk_token")
            .and_then(Value::as_str)
            .unwrap_or("[UNK]")
            .to_string();
        let prefix = model
            .get("continuing_subword_prefix")
            .and_then(Value::as_str)
            .unwrap_or("##")
            .to_string();
        let max_chars = model
            .get("max_input_chars_per_word")
            .and_then(Value::as_u64)
            .unwrap_or(100) as usize;
        Self::new(vocab, unk_token, prefix, max_chars)
    }

    /// Greedy longest-match-first. Any unmatchable remainder turns the whole
    /// word into a single unk token, as in HF.
    pub fn tokenize(&self, word: &str, out: &mut Vec<u32>) {
        let bytes = word.as_bytes();
        let too_long = if bytes.len() <= self.max_input_chars_per_word {
            false
        } else {
            word.chars().count() > self.max_input_chars_per_word
        };
        if too_long {
            out.push(self.unk_id);
            return;
        }
        let before = out.len();
        let mut start = 0;
        while start < bytes.len() {
            let trie = if start == 0 { &self.initial } else { &self.continuing };
            match trie.longest_prefix(&bytes[start..]) {
                Some((len, id)) => {
                    out.push(id);
                    start += len;
                }
                None => {
                    out.truncate(before);
                    out.push(self.unk_id);
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> WordPiece {
        let vocab: FxHashMap<String, u32> = ["[UNK]", "un", "##aff", "##able", "a", "##ff", "é", "##é"]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        WordPiece::new(vocab, "[UNK]".into(), "##".into(), 10).unwrap()
    }

    fn tok(m: &WordPiece, w: &str) -> Vec<u32> {
        let mut out = vec![];
        m.tokenize(w, &mut out);
        out
    }

    #[test]
    fn greedy_longest_match() {
        let m = model();
        assert_eq!(tok(&m, "unaffable"), [1, 2, 3]);
        assert_eq!(tok(&m, "aff"), [4, 5]);
        assert_eq!(tok(&m, "éé"), [6, 7]);
    }

    #[test]
    fn unk_rules() {
        let m = model();
        assert_eq!(tok(&m, "unx"), [0]);
        assert_eq!(tok(&m, "aaaaaaaaaaa"), [0]);
        assert_eq!(tok(&m, "éééééééééé"), [6, 7, 7, 7, 7, 7, 7, 7, 7, 7]);
    }
}
