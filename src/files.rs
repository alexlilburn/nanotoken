use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use memchr::memmem;
use memmap2::Mmap;
use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, Visitor};

use crate::batch::{self, Assembled, EncodeOptions, Unit, WorkerPool};
use crate::encoder::Encoder;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocFormat {
    /// Documents are `file.split(separator)` (a trailing empty document is
    /// dropped); without a separator the whole file is one document.
    Text { separator: Option<Vec<u8>> },
    /// One JSON object per non-blank line; documents are `line[field]`.
    Jsonl { field: String },
}

pub struct MappedFile {
    path: PathBuf,
    map: Option<Mmap>,
}

impl MappedFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?;
        let len = file.metadata()?.len();
        let map = if len == 0 {
            None
        } else {
            // SAFETY: the mapping is read-only; as with any mmap, the file
            // must not be truncated while it is being encoded.
            Some(unsafe { Mmap::map(&file) }.map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?)
        };
        Ok(MappedFile {
            path: path.to_path_buf(),
            map,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        self.map.as_deref().unwrap_or_default()
    }
}

fn has_border(sep: &[u8]) -> bool {
    (1..sep.len()).any(|k| sep[..k] == sep[sep.len() - k..])
}

/// Splits `bytes` into regions of roughly `target` bytes that each end
/// right after a document boundary (or at the end of the file).
fn region_bounds(bytes: &[u8], format: &DocFormat, target: usize) -> Vec<usize> {
    let mut bounds = vec![0];
    if bytes.len() <= target {
        bounds.push(bytes.len());
        return bounds;
    }
    match format {
        DocFormat::Jsonl { .. } => {
            let mut pos = target;
            while pos < bytes.len() {
                match memchr::memchr(b'\n', &bytes[pos..]) {
                    Some(i) => {
                        bounds.push(pos + i + 1);
                        pos += i + 1 + target;
                    }
                    None => break,
                }
            }
        }
        DocFormat::Text { separator: Some(sep) } if !has_border(sep) => {
            let finder = memmem::Finder::new(sep);
            let mut pos = target;
            while pos < bytes.len() {
                match finder.find(&bytes[pos..]) {
                    Some(i) => {
                        bounds.push(pos + i + sep.len());
                        pos += i + sep.len() + target;
                    }
                    None => break,
                }
            }
        }
        DocFormat::Text { separator: Some(sep) } => {
            // A separator that overlaps itself (e.g. "\n\n") can match at
            // offsets a left-to-right split never uses, so walk the real
            // occurrences from the start.
            let mut next = target;
            for i in memmem::find_iter(bytes, sep) {
                let end = i + sep.len();
                if end >= next {
                    bounds.push(end);
                    next = end + target;
                }
            }
        }
        DocFormat::Text { separator: None } => {}
    }
    if *bounds.last().unwrap() < bytes.len() {
        bounds.push(bytes.len());
    }
    bounds
}

pub(crate) fn plan_regions<'a>(bytes: &'a [u8], format: &'a DocFormat, target: usize, units: &mut Vec<Unit<'a>>) {
    for w in region_bounds(bytes, format, target).windows(2) {
        if w[0] < w[1] {
            units.push(Unit::Region {
                bytes: &bytes[w[0]..w[1]],
                format,
            });
        }
    }
}

struct FieldSeed<'f>(&'f str);

struct StrVisitor;

impl<'de> Visitor<'de> for StrVisitor {
    type Value = Cow<'de, str>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a string")
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> std::result::Result<Self::Value, E> {
        Ok(Cow::Borrowed(v))
    }

    fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E> {
        Ok(Cow::Owned(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> std::result::Result<Self::Value, E> {
        Ok(Cow::Owned(v))
    }
}

struct CowStr<'de>(Cow<'de, str>);

impl<'de> serde::Deserialize<'de> for CowStr<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        d.deserialize_str(StrVisitor).map(CowStr)
    }
}

impl<'de> DeserializeSeed<'de> for FieldSeed<'_> {
    type Value = Option<Cow<'de, str>>;

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for FieldSeed<'_> {
    type Value = Option<Cow<'de, str>>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Self::Value, A::Error> {
        let mut found = None;
        while let Some(key) = map.next_key::<CowStr>()? {
            if key.0 == self.0 && found.is_none() {
                found = Some(map.next_value::<CowStr>()?.0);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(found)
    }
}

pub(crate) fn extract_field<'a>(line: &'a [u8], field: &str) -> Result<Cow<'a, str>> {
    let mut de = serde_json::Deserializer::from_slice(line);
    let value = FieldSeed(field)
        .deserialize(&mut de)
        .and_then(|v| de.end().map(|_| v))
        .map_err(|e| {
            Error::Invalid(format!(
                "invalid JSONL line ({e}): {}",
                String::from_utf8_lossy(&line[..line.len().min(80)])
            ))
        })?;
    value.ok_or_else(|| {
        Error::Invalid(format!(
            "JSONL line has no string field {field:?}: {}",
            String::from_utf8_lossy(&line[..line.len().min(80)])
        ))
    })
}

pub(crate) fn encode_region(
    encoder: &mut Encoder,
    bytes: &[u8],
    format: &DocFormat,
    ids: &mut Vec<u32>,
    lens: &mut Vec<u32>,
) -> Result<()> {
    let mut emit = |doc: &str, ids: &mut Vec<u32>| {
        let before = ids.len();
        encoder.encode(doc, ids);
        lens.push((ids.len() - before) as u32);
    };
    match format {
        DocFormat::Jsonl { field } => {
            for line in bytes.split(|&b| b == b'\n') {
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                let text = extract_field(line, field)?;
                emit(&text, ids);
            }
        }
        DocFormat::Text { separator: Some(sep) } if !sep.is_empty() => {
            let mut start = 0;
            for i in memmem::find_iter(bytes, sep) {
                emit(&batch::as_text(&bytes[start..i]), ids);
                start = i + sep.len();
            }
            if start < bytes.len() {
                emit(&batch::as_text(&bytes[start..]), ids);
            }
        }
        DocFormat::Text { .. } => {
            if !bytes.is_empty() {
                emit(&batch::as_text(bytes), ids);
            }
        }
    }
    Ok(())
}

pub fn encode_files<T>(
    tok: &Tokenizer,
    pool: &WorkerPool,
    files: &[MappedFile],
    format: &DocFormat,
    options: &EncodeOptions,
) -> Result<Assembled<T>>
where
    T: From<u32> + Copy + Send + Sync,
{
    let total: usize = files.iter().map(|f| f.bytes().len()).sum();
    let parallel = options.parallel && total > (1 << 20);
    let target = if parallel {
        batch::unit_target_bytes(total)
    } else {
        usize::MAX
    };
    let whole_docs: Vec<&[u8]> = match format {
        DocFormat::Text { separator: None } => files.iter().map(|f| f.bytes()).filter(|b| !b.is_empty()).collect(),
        DocFormat::Text { separator: Some(sep) } if sep.is_empty() => {
            return Err(Error::Invalid("separator must not be empty".into()));
        }
        _ => Vec::new(),
    };
    let lossy = batch::lossy_oversized(&whole_docs, target);
    let mut units = Vec::new();
    if matches!(format, DocFormat::Text { separator: None }) {
        batch::plan_docs(tok, &whole_docs, &lossy, target, &mut units);
    } else {
        for file in files {
            plan_regions(file.bytes(), format, target, &mut units);
        }
    }
    let outs = batch::encode_units(tok, pool, &units, parallel).map_err(|e| match e {
        Error::Invalid(msg) if files.len() == 1 => Error::Invalid(format!("{}: {msg}", files[0].path.display())),
        other => other,
    })?;
    batch::assemble(tok, &outs, &EncodeOptions { parallel, ..*options })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_detection() {
        assert!(has_border(b"\n\n"));
        assert!(has_border(b"abab"));
        assert!(!has_border(b"<|endoftext|>"));
    }

    #[test]
    fn field_extraction() {
        assert_eq!(
            extract_field(br#"{"id": 1, "text": "a\nb", "x": [1]}"#, "text").unwrap(),
            "a\nb"
        );
        assert!(matches!(
            extract_field(br#"{"text": "plain"}"#, "text").unwrap(),
            Cow::Borrowed(_)
        ));
        assert!(extract_field(br#"{"other": "x"}"#, "text").is_err());
        assert!(extract_field(br#"{"text": 3}"#, "text").is_err());
    }

    #[test]
    fn region_bounds_align_with_separators() {
        let sep = b"\n\n".to_vec();
        let data = b"aaa\n\n\nbbb\n\nccc\n\n\n\nddd";
        let format = DocFormat::Text {
            separator: Some(sep.clone()),
        };
        let bounds = region_bounds(data, &format, 4);
        let mut docs = Vec::new();
        for w in bounds.windows(2) {
            let region = &data[w[0]..w[1]];
            let mut start = 0;
            for i in memmem::find_iter(region, &sep) {
                docs.push(region[start..i].to_vec());
                start = i + sep.len();
            }
            if start < region.len() {
                docs.push(region[start..].to_vec());
            }
        }
        let expected: Vec<Vec<u8>> = vec![
            b"aaa".to_vec(),
            b"\nbbb".to_vec(),
            b"ccc".to_vec(),
            b"".to_vec(),
            b"ddd".to_vec(),
        ];
        assert_eq!(docs, expected);
    }
}
