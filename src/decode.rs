use serde_json::Value;

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decoder {
    WordPiece { prefix: String, cleanup: bool },
    JoinSpaces,
}

/// HF applies this per token, not to the joined string, so multi-token
/// patterns like " do not" only fire inside a single token.
fn cleanup(token: &str) -> String {
    token
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

impl Decoder {
    pub fn from_json(value: Option<&Value>) -> Result<Self> {
        let Some(value) = value.filter(|v| !v.is_null()) else {
            return Ok(Decoder::JoinSpaces);
        };
        match value.get("type").and_then(Value::as_str).unwrap_or("") {
            "WordPiece" => Ok(Decoder::WordPiece {
                prefix: value.get("prefix").and_then(Value::as_str).unwrap_or("##").to_string(),
                cleanup: value.get("cleanup").and_then(Value::as_bool).unwrap_or(true),
            }),
            other => Err(Error::Unsupported(format!("decoder {other:?} (supported: WordPiece)"))),
        }
    }

    pub fn decode(&self, tokens: &[&str]) -> String {
        match self {
            Decoder::JoinSpaces => tokens.join(" "),
            Decoder::WordPiece { prefix, cleanup: clean } => {
                let mut out = String::with_capacity(tokens.iter().map(|t| t.len() + 1).sum());
                for (i, &token) in tokens.iter().enumerate() {
                    let piece = if i == 0 {
                        token.to_string()
                    } else if let Some(rest) = token.strip_prefix(prefix.as_str()) {
                        rest.to_string()
                    } else {
                        format!(" {token}")
                    };
                    if *clean {
                        out.push_str(&cleanup(&piece));
                    } else {
                        out.push_str(&piece);
                    }
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordpiece_decoding() {
        let d = Decoder::WordPiece {
            prefix: "##".into(),
            cleanup: true,
        };
        assert_eq!(
            d.decode(&["hello", "##world", ",", "it", "'", "s", "."]),
            "helloworld, it ' s."
        );
        assert_eq!(d.decode(&["##a", "b"]), "##a b");
    }
}
