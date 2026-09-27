use regex::Regex;
use serde_json::Value;
use unicode_categories::UnicodeCategories;

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Behavior {
    Removed,
    Isolated,
    Contiguous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreOp {
    Whitespace,
    WordRegex,
    Punctuation(Behavior),
    Digits(Behavior),
}

#[derive(Clone, Debug)]
pub struct PreTokenizer {
    ops: Vec<PreOp>,
    word_re: Regex,
}

const ASCII_SPACE: u8 = 0;
const ASCII_WORD: u8 = 1;
const ASCII_OTHER: u8 = 2;

fn ascii_class(b: u8) -> u8 {
    if b.is_ascii_alphanumeric() || b == b'_' {
        ASCII_WORD
    } else if (b as char).is_whitespace() {
        ASCII_SPACE
    } else {
        ASCII_OTHER
    }
}

fn split_pred_ascii(
    s: &[u8],
    start: usize,
    end: usize,
    pred: impl Fn(u8) -> bool,
    behavior: Behavior,
    out: &mut Vec<(usize, usize)>,
) {
    let mut run_start = start;
    let mut run_is_match = false;
    for (i, &b) in s[start..end].iter().enumerate() {
        let i = start + i;
        let is_match = pred(b);
        match behavior {
            Behavior::Removed | Behavior::Isolated => {
                if is_match {
                    if run_start < i {
                        out.push((run_start, i));
                    }
                    if behavior == Behavior::Isolated {
                        out.push((i, i + 1));
                    }
                    run_start = i + 1;
                }
            }
            Behavior::Contiguous => {
                if is_match != run_is_match {
                    if run_start < i {
                        out.push((run_start, i));
                    }
                    run_start = i;
                    run_is_match = is_match;
                }
            }
        }
    }
    if run_start < end {
        out.push((run_start, end));
    }
}

/// The `\w+|[^\w\s]+` regex restricted to ASCII: maximal runs of word
/// bytes or of non-word, non-space bytes.
fn split_word_regex_ascii(s: &[u8], start: usize, end: usize, out: &mut Vec<(usize, usize)>) {
    let mut run_start = start;
    let mut run_class = ASCII_SPACE;
    for (i, &b) in s[start..end].iter().enumerate() {
        let i = start + i;
        let class = ascii_class(b);
        if class != run_class {
            if run_class != ASCII_SPACE && run_start < i {
                out.push((run_start, i));
            }
            run_start = i;
            run_class = class;
        }
    }
    if run_class != ASCII_SPACE && run_start < end {
        out.push((run_start, end));
    }
}

fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || c.is_punctuation()
}

fn parse_behavior(value: &Value, default: Behavior) -> Result<Behavior> {
    match value.get("behavior").and_then(Value::as_str) {
        None => Ok(default),
        Some("Removed") => Ok(Behavior::Removed),
        Some("Isolated") => Ok(Behavior::Isolated),
        Some("Contiguous") => Ok(Behavior::Contiguous),
        Some(other) => Err(Error::Unsupported(format!("split behavior {other:?}"))),
    }
}

fn parse_ops(value: &Value, ops: &mut Vec<PreOp>) -> Result<()> {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "BertPreTokenizer" => {
            ops.push(PreOp::Whitespace);
            ops.push(PreOp::Punctuation(Behavior::Isolated));
        }
        "WhitespaceSplit" => ops.push(PreOp::Whitespace),
        "Whitespace" => ops.push(PreOp::WordRegex),
        "Punctuation" => ops.push(PreOp::Punctuation(parse_behavior(value, Behavior::Isolated)?)),
        "Digits" => {
            let individual = value.get("individual_digits").and_then(Value::as_bool).unwrap_or(false);
            ops.push(PreOp::Digits(if individual {
                Behavior::Isolated
            } else {
                Behavior::Contiguous
            }));
        }
        "Sequence" => {
            let children = value
                .get("pretokenizers")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Invalid("Sequence pre_tokenizer without `pretokenizers`".into()))?;
            for child in children {
                parse_ops(child, ops)?;
            }
        }
        other => {
            return Err(Error::Unsupported(format!(
                "pre_tokenizer {other:?} (supported: BertPreTokenizer, Whitespace, WhitespaceSplit, Punctuation, Digits, Sequence)"
            )));
        }
    }
    Ok(())
}

/// Appends the pieces of `s[range]` split on chars matching `pred`, with
/// HF `SplitDelimiterBehavior` semantics (each matching char is its own
/// match; empty pieces are dropped).
fn split_pred(
    s: &str,
    start: usize,
    end: usize,
    pred: impl Fn(char) -> bool,
    behavior: Behavior,
    out: &mut Vec<(usize, usize)>,
) {
    let mut run_start = start;
    let mut run_is_match = false;
    for (i, c) in s[start..end].char_indices() {
        let i = start + i;
        let is_match = pred(c);
        match behavior {
            Behavior::Removed | Behavior::Isolated => {
                if is_match {
                    if run_start < i {
                        out.push((run_start, i));
                    }
                    let next = i + c.len_utf8();
                    if behavior == Behavior::Isolated {
                        out.push((i, next));
                    }
                    run_start = next;
                }
            }
            Behavior::Contiguous => {
                if is_match != run_is_match {
                    if run_start < i {
                        out.push((run_start, i));
                    }
                    run_start = i;
                    run_is_match = is_match;
                }
            }
        }
    }
    if run_start < end {
        out.push((run_start, end));
    }
}

impl PreTokenizer {
    pub fn new(ops: Vec<PreOp>) -> Self {
        PreTokenizer {
            ops,
            word_re: Regex::new(r"\w+|[^\w\s]+").expect("valid regex"),
        }
    }

    pub fn from_json(value: Option<&Value>) -> Result<Self> {
        let mut ops = Vec::new();
        if let Some(value) = value.filter(|v| !v.is_null()) {
            parse_ops(value, &mut ops)?;
        }
        Ok(Self::new(ops))
    }

    /// Whether every piece this produces is free of whitespace, which makes
    /// ASCII whitespace a hard boundary between independently encodable
    /// chunks.
    pub fn splits_on_whitespace(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, PreOp::Whitespace | PreOp::WordRegex))
    }

    /// Whether this is exactly BertPreTokenizer: whitespace removed, each
    /// punctuation char isolated.
    pub fn is_bert(&self) -> bool {
        self.ops == [PreOp::Whitespace, PreOp::Punctuation(Behavior::Isolated)]
    }

    pub fn split(&self, s: &str, out: &mut Vec<(usize, usize)>, scratch: &mut Vec<(usize, usize)>) {
        self.split_impl(s, s.is_ascii(), out, scratch);
    }

    fn split_impl(&self, s: &str, ascii: bool, out: &mut Vec<(usize, usize)>, scratch: &mut Vec<(usize, usize)>) {
        out.clear();
        if !s.is_empty() {
            out.push((0, s.len()));
        }
        let bytes = s.as_bytes();
        for op in &self.ops {
            std::mem::swap(out, scratch);
            out.clear();
            for &(start, end) in scratch.iter() {
                if ascii {
                    match *op {
                        PreOp::Whitespace => split_pred_ascii(
                            bytes,
                            start,
                            end,
                            |b| (b as char).is_whitespace(),
                            Behavior::Removed,
                            out,
                        ),
                        PreOp::Punctuation(behavior) => {
                            split_pred_ascii(bytes, start, end, |b| b.is_ascii_punctuation(), behavior, out)
                        }
                        PreOp::Digits(behavior) => {
                            split_pred_ascii(bytes, start, end, |b| b.is_ascii_digit(), behavior, out)
                        }
                        PreOp::WordRegex => split_word_regex_ascii(bytes, start, end, out),
                    }
                    continue;
                }
                match *op {
                    PreOp::Whitespace => split_pred(s, start, end, char::is_whitespace, Behavior::Removed, out),
                    PreOp::Punctuation(behavior) => split_pred(s, start, end, is_punctuation, behavior, out),
                    PreOp::Digits(behavior) => split_pred(s, start, end, char::is_numeric, behavior, out),
                    PreOp::WordRegex => {
                        for m in self.word_re.find_iter(&s[start..end]) {
                            out.push((start + m.start(), start + m.end()));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces<'a>(p: &PreTokenizer, s: &'a str) -> Vec<&'a str> {
        let (mut out, mut scratch) = (Vec::new(), Vec::new());
        p.split(s, &mut out, &mut scratch);
        out.iter().map(|&(a, b)| &s[a..b]).collect()
    }

    #[test]
    fn bert_pre_tokenizer() {
        let p = PreTokenizer::new(vec![PreOp::Whitespace, PreOp::Punctuation(Behavior::Isolated)]);
        assert_eq!(
            pieces(&p, "Hey, you...  ok?"),
            ["Hey", ",", "you", ".", ".", ".", "ok", "?"]
        );
        assert_eq!(pieces(&p, "  "), Vec::<&str>::new());
    }

    #[test]
    fn ascii_path_matches_char_path() {
        let ops = [
            vec![PreOp::Whitespace, PreOp::Punctuation(Behavior::Isolated)],
            vec![PreOp::WordRegex],
            vec![
                PreOp::Whitespace,
                PreOp::Digits(Behavior::Contiguous),
                PreOp::Punctuation(Behavior::Contiguous),
            ],
            vec![PreOp::Punctuation(Behavior::Removed), PreOp::Digits(Behavior::Isolated)],
        ];
        let mut state = 0x2545f4914f6cdd1du64;
        for ops in ops {
            let p = PreTokenizer::new(ops);
            for _ in 0..500 {
                let text: String = (0..(state % 40))
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state % 128) as u8 as char
                    })
                    .collect();
                let (mut fast, mut slow, mut scratch) = (Vec::new(), Vec::new(), Vec::new());
                p.split_impl(&text, true, &mut fast, &mut scratch);
                p.split_impl(&text, false, &mut slow, &mut scratch);
                assert_eq!(fast, slow, "{text:?}");
            }
        }
    }

    #[test]
    fn word_regex_and_digits() {
        let p = PreTokenizer::new(vec![PreOp::WordRegex]);
        assert_eq!(pieces(&p, "Hey, you...ok?!"), ["Hey", ",", "you", "...", "ok", "?!"]);
        let p = PreTokenizer::new(vec![PreOp::Whitespace, PreOp::Digits(Behavior::Contiguous)]);
        assert_eq!(pieces(&p, "ab123cd 45"), ["ab", "123", "cd", "45"]);
        let p = PreTokenizer::new(vec![PreOp::Digits(Behavior::Isolated)]);
        assert_eq!(pieces(&p, "a12"), ["a", "1", "2"]);
    }
}
