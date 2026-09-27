// Unicode tables must come from the same crates HF `tokenizers` uses
// (unicode_categories, unicode-normalization-alignments): they pin older
// Unicode versions than std, and swapping them for newer tables silently
// changes which characters count as controls, punctuation or accents.
use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;
use unicode_normalization_alignments::char::is_combining_mark;

use crate::error::{Error, Result};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum NormOp {
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
    },
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Lowercase,
    StripAccents,
}

pub const DROP: u8 = 0xFF;

#[derive(Clone, Debug)]
pub struct Normalizer {
    ops: Vec<NormOp>,
    ascii_table: Option<[u8; 128]>,
}

fn is_bert_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_bert_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}

fn is_chinese_char(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

fn apply(op: &NormOp, s: &str) -> String {
    match *op {
        NormOp::Bert {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        } => {
            let mut out = String::with_capacity(s.len());
            for c in s.chars() {
                let c = if clean_text {
                    if c == '\0' || c == '\u{fffd}' || is_bert_control(c) {
                        continue;
                    }
                    if is_bert_whitespace(c) { ' ' } else { c }
                } else {
                    c
                };
                if handle_chinese_chars && is_chinese_char(c) {
                    out.push(' ');
                    out.push(c);
                    out.push(' ');
                } else {
                    out.push(c);
                }
            }
            if strip_accents {
                out = out.nfd().map(|(c, _)| c).filter(|c| !c.is_mark_nonspacing()).collect();
            }
            if lowercase {
                out = out.chars().flat_map(char::to_lowercase).collect();
            }
            out
        }
        NormOp::Nfc => s.nfc().map(|(c, _)| c).collect(),
        NormOp::Nfd => s.nfd().map(|(c, _)| c).collect(),
        NormOp::Nfkc => s.nfkc().map(|(c, _)| c).collect(),
        NormOp::Nfkd => s.nfkd().map(|(c, _)| c).collect(),
        NormOp::Lowercase => s.chars().flat_map(char::to_lowercase).collect(),
        NormOp::StripAccents => s.chars().filter(|&c| !is_combining_mark(c)).collect(),
    }
}

fn parse_ops(value: &Value, ops: &mut Vec<NormOp>) -> Result<()> {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
    let flag = |name: &str, default: bool| value.get(name).and_then(Value::as_bool).unwrap_or(default);
    match kind {
        "BertNormalizer" => {
            let lowercase = flag("lowercase", true);
            ops.push(NormOp::Bert {
                clean_text: flag("clean_text", true),
                handle_chinese_chars: flag("handle_chinese_chars", true),
                strip_accents: value.get("strip_accents").and_then(Value::as_bool).unwrap_or(lowercase),
                lowercase,
            });
        }
        "NFC" => ops.push(NormOp::Nfc),
        "NFD" => ops.push(NormOp::Nfd),
        "NFKC" => ops.push(NormOp::Nfkc),
        "NFKD" => ops.push(NormOp::Nfkd),
        "Lowercase" => ops.push(NormOp::Lowercase),
        "StripAccents" => ops.push(NormOp::StripAccents),
        "Sequence" => {
            let children = value
                .get("normalizers")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Invalid("Sequence normalizer without `normalizers`".into()))?;
            for child in children {
                parse_ops(child, ops)?;
            }
        }
        other => {
            return Err(Error::Unsupported(format!(
                "normalizer {other:?} (supported: BertNormalizer, NFC, NFD, NFKC, NFKD, Lowercase, StripAccents, Sequence)"
            )));
        }
    }
    Ok(())
}

impl Normalizer {
    pub fn new(ops: Vec<NormOp>) -> Self {
        let mut normalizer = Normalizer { ops, ascii_table: None };
        normalizer.ascii_table = normalizer.build_ascii_table();
        normalizer
    }

    pub fn from_json(value: Option<&Value>) -> Result<Self> {
        let mut ops = Vec::new();
        if let Some(value) = value.filter(|v| !v.is_null()) {
            parse_ops(value, &mut ops)?;
        }
        Ok(Self::new(ops))
    }

    pub fn is_identity(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn normalize(&self, s: &str) -> String {
        let mut ops = self.ops.iter();
        let Some(first) = ops.next() else {
            return s.to_owned();
        };
        let mut out = apply(first, s);
        for op in ops {
            out = apply(op, &out);
        }
        out
    }

    /// Maps each ASCII byte to its normalized byte or `DROP`, derived from
    /// the Unicode path itself so the two can never disagree. None when
    /// some ASCII char does not normalize to at most one ASCII char.
    fn build_ascii_table(&self) -> Option<[u8; 128]> {
        let mut table = [0u8; 128];
        for b in 0u8..128 {
            let normalized = self.normalize(&(b as char).to_string());
            table[b as usize] = match normalized.as_bytes() {
                [] => DROP,
                [c] if c.is_ascii() => *c,
                _ => return None,
            };
        }
        Some(table)
    }

    pub fn ascii_table(&self) -> Option<&[u8; 128]> {
        self.ascii_table.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bert(lowercase: bool) -> Normalizer {
        Normalizer::new(vec![NormOp::Bert {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: lowercase,
            lowercase,
        }])
    }

    #[test]
    fn bert_uncased() {
        let n = bert(true);
        assert_eq!(n.normalize("Héllo\u{0}\tWörld 中文"), "hello world  中  文 ");
        assert_eq!(n.normalize("a\u{200b}b"), "ab");
        assert_eq!(n.normalize("x\u{0b}y\u{0c}z"), "xyz");
    }

    #[test]
    fn ascii_table_matches_unicode_path() {
        let n = bert(true);
        let table = n.ascii_table().unwrap();
        assert_eq!(table[b'A' as usize], b'a');
        assert_eq!(table[0], DROP);
        assert_eq!(table[0x0b], DROP);
        assert_eq!(table[b'\t' as usize], b' ');
    }
}
