use std::borrow::Cow;
use std::sync::Mutex;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::cache::WordCache;
use crate::encoder::Encoder;
use crate::error::{Error, Result};
use crate::files::{DocFormat, encode_region};
use crate::tokenizer::{SEPARATORS, Tokenizer};

const MIN_UNIT_BYTES: usize = 1 << 20;

/// Word caches that persist across calls, checked out by whichever thread
/// encodes the next unit.
#[derive(Default)]
pub struct WorkerPool {
    caches: Mutex<Vec<WordCache>>,
}

impl WorkerPool {
    pub fn with_encoder<R>(&self, tok: &Tokenizer, f: impl FnOnce(&mut Encoder) -> R) -> R {
        let cache = self
            .caches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default();
        let mut encoder = Encoder::new(tok, cache);
        let result = f(&mut encoder);
        self.caches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(encoder.into_cache());
        result
    }

    pub fn clear(&self) {
        self.caches.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

pub(crate) enum Unit<'a> {
    Docs(&'a [&'a [u8]]),
    /// Consecutive fragments of one document, split where no chunk or
    /// added-token match straddles the boundary.
    Fragment {
        text: &'a str,
        first: bool,
    },
    Region {
        bytes: &'a [u8],
        format: &'a DocFormat,
    },
}

#[derive(Default)]
pub(crate) struct UnitOut {
    ids: Vec<u32>,
    lens: Vec<u32>,
    continues: bool,
}

pub(crate) fn unit_target_bytes(total: usize) -> usize {
    (total / (16 * rayon::current_num_threads())).max(MIN_UNIT_BYTES)
}

/// `from_utf8` validates much faster than `from_utf8_lossy`'s chunked
/// scan, so only invalid input pays for the lossy conversion.
pub(crate) fn as_text(bytes: &[u8]) -> Cow<'_, str> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Cow::Borrowed(text),
        Err(_) => String::from_utf8_lossy(bytes),
    }
}

fn estimated_ids(bytes: usize) -> usize {
    bytes / 3 + 16
}

fn encode_unit(encoder: &mut Encoder, unit: &Unit) -> Result<UnitOut> {
    let mut out = UnitOut::default();
    match unit {
        Unit::Docs(docs) => {
            out.ids.reserve(estimated_ids(docs.iter().map(|d| d.len()).sum()));
            out.lens.reserve(docs.len());
            for doc in docs.iter() {
                let before = out.ids.len();
                encoder.encode(&as_text(doc), &mut out.ids);
                out.lens.push((out.ids.len() - before) as u32);
            }
        }
        Unit::Fragment { text, first } => {
            out.ids.reserve(estimated_ids(text.len()));
            encoder.encode(text, &mut out.ids);
            out.lens.push(out.ids.len() as u32);
            out.continues = !first;
        }
        Unit::Region { bytes, format } => {
            out.ids.reserve(estimated_ids(bytes.len()));
            encode_region(encoder, bytes, format, &mut out.ids, &mut out.lens)?
        }
    }
    Ok(out)
}

/// Split points of an oversized document: separator bytes at least
/// `target` apart that lie inside no added-token match.
fn fragment_bounds(tok: &Tokenizer, text: &str, target: usize) -> Vec<usize> {
    let bytes = text.as_bytes();
    let spans = tok.added.raw_spans(text);
    let mut bounds = vec![0];
    let mut span_idx = 0;
    let mut pos = target;
    while pos < bytes.len() {
        let Some(found) = bytes[pos..].iter().position(|b| SEPARATORS.contains(b)) else {
            break;
        };
        let p = pos + found;
        while span_idx < spans.len() && spans[span_idx].1 <= p {
            span_idx += 1;
        }
        if span_idx < spans.len() && spans[span_idx].0 <= p {
            pos = spans[span_idx].1;
            continue;
        }
        bounds.push(p);
        pos = p + target;
    }
    bounds.push(bytes.len());
    bounds
}

/// Lossy UTF-8 copies of the oversized invalid documents, which fragments
/// must borrow as `str`.
pub(crate) fn lossy_oversized(docs: &[&[u8]], target: usize) -> FxHashMap<usize, String> {
    docs.iter()
        .enumerate()
        .filter(|(_, d)| d.len() > target && std::str::from_utf8(d).is_err())
        .map(|(i, d)| (i, String::from_utf8_lossy(d).into_owned()))
        .collect()
}

pub(crate) fn plan_docs<'a>(
    tok: &Tokenizer,
    docs: &'a [&'a [u8]],
    lossy: &'a FxHashMap<usize, String>,
    target: usize,
    units: &mut Vec<Unit<'a>>,
) {
    let mut run_start = 0;
    let mut run_bytes = 0;
    for (i, doc) in docs.iter().enumerate() {
        if doc.len() > target && tok.chunkable() {
            if run_start < i {
                units.push(Unit::Docs(&docs[run_start..i]));
            }
            let text = match lossy.get(&i) {
                Some(owned) => owned.as_str(),
                None => std::str::from_utf8(doc).expect("valid UTF-8 checked by lossy_oversized"),
            };
            let bounds = fragment_bounds(tok, text, target);
            for (k, w) in bounds.windows(2).enumerate() {
                units.push(Unit::Fragment {
                    text: &text[w[0]..w[1]],
                    first: k == 0,
                });
            }
            run_start = i + 1;
            run_bytes = 0;
            continue;
        }
        run_bytes += doc.len();
        if run_bytes >= target {
            units.push(Unit::Docs(&docs[run_start..=i]));
            run_start = i + 1;
            run_bytes = 0;
        }
    }
    if run_start < docs.len() {
        units.push(Unit::Docs(&docs[run_start..]));
    }
}

pub(crate) fn encode_units(tok: &Tokenizer, pool: &WorkerPool, units: &[Unit], parallel: bool) -> Result<Vec<UnitOut>> {
    if parallel && units.len() > 1 {
        units
            .par_iter()
            .map(|unit| pool.with_encoder(tok, |enc| encode_unit(enc, unit)))
            .collect()
    } else {
        pool.with_encoder(tok, |enc| units.iter().map(|unit| encode_unit(enc, unit)).collect())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RowSpec {
    pub add_special_tokens: bool,
    pub max_length: Option<usize>,
    pub truncate_left: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PadSpec {
    /// None pads to the longest row.
    pub width: Option<usize>,
    pub multiple_of: Option<usize>,
    pub pad_left: bool,
    pub pad_id: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Want {
    pub attention_mask: bool,
    pub type_ids: bool,
    pub special_mask: bool,
}

/// How documents become rows: special tokens, truncation, padding, which
/// masks to build, and whether to fan out across threads.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    pub spec: RowSpec,
    pub pad: Option<PadSpec>,
    pub want: Want,
    pub parallel: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            spec: RowSpec {
                add_special_tokens: true,
                ..Default::default()
            },
            pad: None,
            want: Want::default(),
            parallel: true,
        }
    }
}

pub struct Assembled<T> {
    pub ids: Vec<T>,
    pub offsets: Vec<i64>,
    pub attention_mask: Option<Vec<T>>,
    pub type_ids: Option<Vec<T>>,
    pub special_mask: Option<Vec<T>>,
    /// Set when every row has the same length (a padded batch).
    pub width: Option<usize>,
}

#[derive(Clone, Copy)]
struct SharedMut<T>(*mut T);
unsafe impl<T: Send> Send for SharedMut<T> {}
unsafe impl<T: Send> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    /// SAFETY: callers must write disjoint ranges within the allocation.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

fn round_up(n: usize, multiple: Option<usize>) -> usize {
    match multiple {
        Some(m) if m > 0 => n.div_ceil(m) * m,
        _ => n,
    }
}

/// Lays out encoded units as rows: special tokens, truncation, optional
/// padding, and the requested masks, all written in parallel into flat
/// buffers.
pub(crate) fn assemble<T>(tok: &Tokenizer, outs: &[UnitOut], options: &EncodeOptions) -> Result<Assembled<T>>
where
    T: From<u32> + Copy + Send + Sync,
{
    let EncodeOptions {
        spec,
        pad,
        want,
        parallel,
    } = *options;
    let mut unit_first_doc = Vec::with_capacity(outs.len());
    let mut unit_first_offset = Vec::with_capacity(outs.len());
    let mut raw_len: Vec<usize> = Vec::new();
    for out in outs {
        let continues = out.continues && !raw_len.is_empty();
        let first_doc = if continues { raw_len.len() - 1 } else { raw_len.len() };
        unit_first_doc.push(first_doc);
        unit_first_offset.push(if continues { raw_len[first_doc] } else { 0 });
        for (k, &len) in out.lens.iter().enumerate() {
            if k == 0 && continues {
                raw_len[first_doc] += len as usize;
            } else {
                raw_len.push(len as usize);
            }
        }
    }
    let docs = raw_len.len();

    let post = &tok.post;
    let (prefix, suffix, prefix_types, suffix_types): (&[u32], &[u32], &[u32], &[u32]) = if spec.add_special_tokens {
        (&post.prefix, &post.suffix, &post.prefix_types, &post.suffix_types)
    } else {
        (&[], &[], &[], &[])
    };
    let added = prefix.len() + suffix.len();
    let cap = match spec.max_length {
        Some(max) if max < added => {
            return Err(Error::Invalid(format!(
                "max_length={max} leaves no room for the {added} special tokens"
            )));
        }
        Some(max) => max - added,
        None => usize::MAX,
    };

    let content_len: Vec<usize> = raw_len.iter().map(|&r| r.min(cap)).collect();
    let row_len: Vec<usize> = content_len.iter().map(|&c| c + added).collect();
    let width = pad.map(|p| {
        round_up(
            p.width.unwrap_or_else(|| row_len.iter().copied().max().unwrap_or(0)),
            p.multiple_of,
        )
    });
    let out_len: Vec<usize> = row_len.iter().map(|&r| width.map_or(r, |w| w.max(r))).collect();
    let mut offsets = Vec::with_capacity(docs + 1);
    offsets.push(0i64);
    let mut total = 0usize;
    for &len in &out_len {
        total += len;
        offsets.push(total as i64);
    }
    let row_start = |d: usize| -> usize {
        let pad_before = match pad {
            Some(p) if p.pad_left => out_len[d] - row_len[d],
            _ => 0,
        };
        offsets[d] as usize + pad_before
    };
    let keep_start = |d: usize| -> usize {
        if spec.truncate_left {
            raw_len[d] - content_len[d]
        } else {
            0
        }
    };

    let pad_id = T::from(pad.map_or(0, |p| p.pad_id));
    let mut ids = vec![pad_id; total];
    let ids_ptr = SharedMut(ids.as_mut_ptr());

    let copy_unit = |u: usize| {
        let out = &outs[u];
        let mut src = 0usize;
        for (k, &len) in out.lens.iter().enumerate() {
            let len = len as usize;
            let d = unit_first_doc[u] + k;
            let raw_off = if k == 0 { unit_first_offset[u] } else { 0 };
            let ks = keep_start(d);
            let ke = ks + content_len[d];
            let lo = raw_off.max(ks);
            let hi = (raw_off + len).min(ke);
            if lo < hi {
                let dest = row_start(d) + prefix.len() + (lo - ks);
                // SAFETY: each (doc, raw range) pair maps to a distinct
                // destination range, so concurrent units never overlap.
                let dst = unsafe { ids_ptr.slice(dest, hi - lo) };
                for (dst, &id) in dst.iter_mut().zip(&out.ids[src + (lo - raw_off)..src + (hi - raw_off)]) {
                    *dst = T::from(id);
                }
            }
            src += len;
        }
    };
    if parallel && outs.len() > 1 {
        (0..outs.len()).into_par_iter().for_each(copy_unit);
    } else {
        (0..outs.len()).for_each(copy_unit);
    }

    let one = T::from(1);
    let zero = T::from(0);
    let mut attention_mask = want.attention_mask.then(|| vec![zero; total]);
    let mut type_ids = want.type_ids.then(|| vec![zero; total]);
    let mut special_mask = want.special_mask.then(|| vec![one; total]);
    let mask_ptr = attention_mask.as_mut().map(|v| SharedMut(v.as_mut_ptr()));
    let type_ptr = type_ids.as_mut().map(|v| SharedMut(v.as_mut_ptr()));
    let special_ptr = special_mask.as_mut().map(|v| SharedMut(v.as_mut_ptr()));
    let sequence_type = T::from(post.sequence_type);

    let fill_doc = |d: usize| {
        let start = row_start(d);
        let content = content_len[d];
        // SAFETY: rows occupy disjoint [offsets[d], offsets[d + 1]) ranges.
        unsafe {
            ids_ptr
                .slice(start, prefix.len())
                .iter_mut()
                .zip(prefix)
                .for_each(|(o, &id)| *o = T::from(id));
            ids_ptr
                .slice(start + prefix.len() + content, suffix.len())
                .iter_mut()
                .zip(suffix)
                .for_each(|(o, &id)| *o = T::from(id));
            if let Some(ptr) = mask_ptr {
                ptr.slice(start, row_len[d]).fill(one);
            }
            if let Some(ptr) = type_ptr {
                let row = ptr.slice(start, row_len[d]);
                let (pre, rest) = row.split_at_mut(prefix.len());
                let (mid, post_row) = rest.split_at_mut(content);
                pre.iter_mut().zip(prefix_types).for_each(|(o, &t)| *o = T::from(t));
                mid.fill(sequence_type);
                post_row
                    .iter_mut()
                    .zip(suffix_types)
                    .for_each(|(o, &t)| *o = T::from(t));
            }
            if let Some(ptr) = special_ptr {
                ptr.slice(start + prefix.len(), content).fill(zero);
            }
        }
    };
    if parallel && docs > 4096 {
        (0..docs).into_par_iter().with_min_len(1024).for_each(fill_doc);
    } else {
        (0..docs).for_each(fill_doc);
    }

    Ok(Assembled {
        ids,
        offsets,
        attention_mask,
        type_ids,
        special_mask,
        width: width.filter(|&w| out_len.iter().all(|&l| l == w)),
    })
}

/// Encodes documents (UTF-8 bytes; invalid sequences are replaced) and
/// assembles them into rows.
pub fn encode_docs<T>(
    tok: &Tokenizer,
    pool: &WorkerPool,
    docs: &[&[u8]],
    options: &EncodeOptions,
) -> Result<Assembled<T>>
where
    T: From<u32> + Copy + Send + Sync,
{
    let total: usize = docs.iter().map(|d| d.len()).sum();
    let parallel = options.parallel && total > MIN_UNIT_BYTES;
    let target = if parallel { unit_target_bytes(total) } else { usize::MAX };
    let lossy = lossy_oversized(docs, target);
    let mut units = Vec::new();
    plan_docs(tok, docs, &lossy, target, &mut units);
    let outs = encode_units(tok, pool, &units, parallel)?;
    assemble(tok, &outs, &EncodeOptions { parallel, ..*options })
}

#[cfg(test)]
pub(crate) fn encode_fragmented(tok: &Tokenizer, doc: &str, target: usize) -> Vec<u32> {
    let pool = WorkerPool::default();
    let docs = [doc.as_bytes()];
    let lossy = FxHashMap::default();
    let mut units = Vec::new();
    plan_docs(tok, &docs, &lossy, target, &mut units);
    let outs = encode_units(tok, &pool, &units, true).unwrap();
    let options = EncodeOptions {
        spec: RowSpec::default(),
        ..Default::default()
    };
    assemble::<u32>(tok, &outs, &options).unwrap().ids
}
