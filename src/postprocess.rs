use serde_json::Value;

use crate::error::{Error, Result};

/// The single-sequence template: ids (and type ids) placed around the
/// encoded text when special tokens are added.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PostProcessor {
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    pub prefix_types: Vec<u32>,
    pub suffix_types: Vec<u32>,
    pub sequence_type: u32,
}

fn token_pair(value: &Value, name: &str) -> Result<u32> {
    value
        .get(name)
        .and_then(Value::as_array)
        .and_then(|pair| pair.get(1))
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| Error::Invalid(format!("post_processor `{name}` must be [token, id]")))
}

impl PostProcessor {
    pub fn from_json(value: Option<&Value>) -> Result<Self> {
        let Some(value) = value.filter(|v| !v.is_null()) else {
            return Ok(Self::default());
        };
        match value.get("type").and_then(Value::as_str).unwrap_or("") {
            "BertProcessing" | "RobertaProcessing" => {
                let cls = token_pair(value, "cls")?;
                let sep = token_pair(value, "sep")?;
                Ok(PostProcessor {
                    prefix: vec![cls],
                    suffix: vec![sep],
                    prefix_types: vec![0],
                    suffix_types: vec![0],
                    sequence_type: 0,
                })
            }
            "TemplateProcessing" => Self::from_template(value),
            other => Err(Error::Unsupported(format!(
                "post_processor {other:?} (supported: BertProcessing, RobertaProcessing, TemplateProcessing)"
            ))),
        }
    }

    fn from_template(value: &Value) -> Result<Self> {
        let pieces = value
            .get("single")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Invalid("TemplateProcessing without a `single` template".into()))?;
        let specials = value.get("special_tokens");
        let mut out = PostProcessor::default();
        let mut seen_sequence = false;
        for piece in pieces {
            if let Some(seq) = piece.get("Sequence") {
                if seen_sequence || seq.get("id").and_then(Value::as_str) != Some("A") {
                    return Err(Error::Unsupported("single template must contain exactly one $A".into()));
                }
                seen_sequence = true;
                out.sequence_type = seq.get("type_id").and_then(Value::as_u64).unwrap_or(0) as u32;
            } else if let Some(special) = piece.get("SpecialToken") {
                let name = special
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::Invalid("template SpecialToken without `id`".into()))?;
                let type_id = special.get("type_id").and_then(Value::as_u64).unwrap_or(0) as u32;
                let ids = specials
                    .and_then(|s| s.get(name))
                    .and_then(|s| s.get("ids"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::Invalid(format!("template special token {name:?} has no ids")))?;
                for id in ids {
                    let id = id
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| Error::Invalid(format!("invalid id for template token {name:?}")))?;
                    let (ids, types) = if seen_sequence {
                        (&mut out.suffix, &mut out.suffix_types)
                    } else {
                        (&mut out.prefix, &mut out.prefix_types)
                    };
                    ids.push(id);
                    types.push(type_id);
                }
            } else {
                return Err(Error::Invalid("unrecognized template piece".into()));
            }
        }
        if !seen_sequence {
            return Err(Error::Unsupported("single template without $A".into()));
        }
        Ok(out)
    }

    pub fn num_added(&self) -> usize {
        self.prefix.len() + self.suffix.len()
    }
}
