use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use regex::Regex;
use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::normalize::Normalizer;

#[derive(Clone, Debug)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

#[derive(Clone, Debug)]
struct Matcher {
    automaton: AhoCorasick,
    tokens: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    Text(usize, usize),
    Token(u32, usize, usize),
}

#[derive(Clone, Debug)]
pub struct AddedVocab {
    pub tokens: Vec<AddedToken>,
    pub by_id: FxHashMap<u32, usize>,
    pub by_content: FxHashMap<String, u32>,
    /// Content the decoder emits for normalized tokens whose normalized
    /// form differs (HF's `normalized_cache`).
    pub decoded: FxHashMap<u32, String>,
    raw: Option<Matcher>,
    normalized: Option<Matcher>,
    re_word_start: Regex,
    re_word_end: Regex,
    re_space_start: Regex,
    re_space_end: Regex,
}

fn build_matcher(patterns: Vec<(String, usize)>) -> Result<Option<Matcher>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let automaton = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns.iter().map(|(p, _)| p.as_str()))
        .map_err(|e| Error::Invalid(format!("cannot build added-token matcher: {e}")))?;
    Ok(Some(Matcher {
        automaton,
        tokens: patterns.into_iter().map(|(_, i)| i).collect(),
    }))
}

impl AddedVocab {
    pub fn from_json(value: Option<&Value>, normalizer: &Normalizer) -> Result<Self> {
        let mut tokens = Vec::new();
        for entry in value.and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default() {
            let flag = |name: &str, default: bool| entry.get(name).and_then(Value::as_bool).unwrap_or(default);
            let content = entry
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Invalid("added token without `content`".into()))?
                .to_string();
            let id = entry
                .get("id")
                .and_then(Value::as_u64)
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| Error::Invalid(format!("added token {content:?} without a valid `id`")))?;
            if content.is_empty() {
                continue;
            }
            let special = flag("special", false);
            tokens.push(AddedToken {
                id,
                content,
                single_word: flag("single_word", false),
                lstrip: flag("lstrip", false),
                rstrip: flag("rstrip", false),
                normalized: flag("normalized", !special),
                special,
            });
        }
        Self::new(tokens, normalizer)
    }

    pub fn new(tokens: Vec<AddedToken>, normalizer: &Normalizer) -> Result<Self> {
        let mut by_id = FxHashMap::default();
        let mut by_content = FxHashMap::default();
        let mut decoded = FxHashMap::default();
        let mut raw_patterns = Vec::new();
        let mut normalized_patterns = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            by_id.insert(token.id, i);
            by_content.insert(token.content.clone(), token.id);
            if token.normalized {
                let normed = normalizer.normalize(&token.content);
                if normed != token.content {
                    decoded.insert(token.id, normed.clone());
                }
                if !normed.is_empty() {
                    normalized_patterns.push((normed, i));
                }
            } else {
                raw_patterns.push((token.content.clone(), i));
            }
        }
        Ok(AddedVocab {
            raw: build_matcher(raw_patterns)?,
            normalized: build_matcher(normalized_patterns)?,
            tokens,
            by_id,
            by_content,
            decoded,
            re_word_start: Regex::new(r"^\w").expect("valid regex"),
            re_word_end: Regex::new(r"\w$").expect("valid regex"),
            re_space_start: Regex::new(r"^\s*").expect("valid regex"),
            re_space_end: Regex::new(r"\s*$").expect("valid regex"),
        })
    }

    pub fn has_raw(&self) -> bool {
        self.raw.is_some()
    }

    pub fn has_normalized(&self) -> bool {
        self.normalized.is_some()
    }

    pub fn is_special(&self, content: &str) -> bool {
        self.by_content
            .get(content)
            .and_then(|id| self.by_id.get(id))
            .is_some_and(|&i| self.tokens[i].special)
    }

    /// The token string HF's decoder sees for an added-token id.
    pub fn decoded_content(&self, id: u32) -> Option<&str> {
        let &i = self.by_id.get(&id)?;
        Some(
            self.decoded
                .get(&id)
                .map_or(self.tokens[i].content.as_str(), String::as_str),
        )
    }

    /// HF `AddedVocabulary::find_matches`, also reporting through
    /// `on_span` every candidate match's extent (including candidates
    /// rejected by `single_word`) so callers can find split points that no
    /// match can straddle.
    fn find(&self, s: &str, matcher: &Matcher, out: &mut Vec<Segment>, mut on_span: impl FnMut(usize, usize)) {
        out.clear();
        let mut start_offset = 0;
        for m in matcher.automaton.find_iter(s) {
            let token = &self.tokens[matcher.tokens[m.pattern().as_usize()]];
            let (mut start, mut stop) = (m.start(), m.end());
            if token.single_word {
                let start_space = start == 0 || !self.re_word_end.is_match(&s[..start]);
                let stop_space = stop == s.len() || !self.re_word_start.is_match(&s[stop..]);
                if !start_space || !stop_space {
                    on_span(start, stop);
                    continue;
                }
            }
            if token.lstrip {
                let new_start = self.re_space_end.find(&s[..start]).map_or(start, |m| m.start());
                start = new_start.max(start_offset);
            }
            if token.rstrip {
                stop += self.re_space_start.find(&s[stop..]).map_or(0, |m| m.end());
            }
            on_span(start, stop);
            if start_offset < start {
                out.push(Segment::Text(start_offset, start));
            }
            out.push(Segment::Token(token.id, start, stop));
            start_offset = stop;
        }
        if start_offset < s.len() {
            out.push(Segment::Text(start_offset, s.len()));
        }
    }

    pub fn split_raw(&self, s: &str, out: &mut Vec<Segment>) {
        match &self.raw {
            Some(matcher) => self.find(s, matcher, out, |_, _| {}),
            None => {
                out.clear();
                out.push(Segment::Text(0, s.len()));
            }
        }
    }

    pub fn split_normalized(&self, s: &str, out: &mut Vec<Segment>) {
        match &self.normalized {
            Some(matcher) => self.find(s, matcher, out, |_, _| {}),
            None => {
                out.clear();
                out.push(Segment::Text(0, s.len()));
            }
        }
    }

    /// Byte extents of raw-text matches (after lstrip/rstrip), which a
    /// document split point must avoid.
    pub fn raw_spans(&self, s: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        if let Some(matcher) = &self.raw {
            let mut segments = Vec::new();
            self.find(s, matcher, &mut segments, |a, b| spans.push((a, b)));
        }
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(tokens: &[(&str, bool, bool, bool)]) -> AddedVocab {
        let tokens = tokens
            .iter()
            .enumerate()
            .map(|(i, &(content, single_word, lstrip, rstrip))| AddedToken {
                id: 100 + i as u32,
                content: content.to_string(),
                single_word,
                lstrip,
                rstrip,
                normalized: false,
                special: true,
            })
            .collect();
        AddedVocab::new(tokens, &Normalizer::new(vec![])).unwrap()
    }

    fn split(v: &AddedVocab, s: &str) -> Vec<Segment> {
        let mut out = vec![];
        v.split_raw(s, &mut out);
        out
    }

    #[test]
    fn strips_and_single_word() {
        let v = vocab(&[
            ("[MASK]", false, true, false),
            ("[SEP]", false, false, true),
            ("cat", true, false, false),
        ]);
        assert_eq!(
            split(&v, "a  [MASK] b"),
            [Segment::Text(0, 1), Segment::Token(100, 1, 9), Segment::Text(9, 11)]
        );
        assert_eq!(split(&v, "[SEP]  x"), [Segment::Token(101, 0, 7), Segment::Text(7, 8)]);
        assert_eq!(split(&v, "concat"), [Segment::Text(0, 6)]);
        assert_eq!(
            split(&v, "a cat."),
            [Segment::Text(0, 2), Segment::Token(102, 2, 5), Segment::Text(5, 6)]
        );
    }
}
