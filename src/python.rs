use std::path::PathBuf;
use std::sync::Arc;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyDict, PyList, PyString};

use crate::batch::{Assembled, EncodeOptions, PadSpec, RowSpec, Want, WorkerPool, encode_docs};
use crate::error::Error;
use crate::files::{DocFormat, MappedFile, encode_files};
use crate::tokenizer::Tokenizer;

create_exception!(
    nanotoken_rs,
    UnsupportedTokenizerError,
    PyValueError,
    "The tokenizer uses a component nanotoken does not implement."
);

fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::Unsupported(_) => UnsupportedTokenizerError::new_err(err.to_string()),
        Error::Invalid(msg) => PyValueError::new_err(msg),
        Error::Io(e) => PyOSError::new_err(e.to_string()),
    }
}

enum Doc {
    Str(PyBackedStr),
    Bytes(PyBackedBytes),
}

impl Doc {
    fn extract(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.cast::<PyString>() {
            return Ok(Doc::Str(s.clone().try_into()?));
        }
        if let Ok(b) = obj.cast::<PyBytes>() {
            return Ok(Doc::Bytes(b.clone().into()));
        }
        Err(PyTypeError::new_err(format!(
            "expected str or bytes documents, got {}",
            obj.get_type().name()?
        )))
    }

    fn as_bytes(&self) -> &[u8] {
        match self {
            Doc::Str(s) => s.as_bytes(),
            Doc::Bytes(b) => b,
        }
    }
}

fn extract_docs(inputs: &Bound<'_, PyAny>) -> PyResult<Vec<Doc>> {
    if inputs.is_instance_of::<PyString>() || inputs.is_instance_of::<PyBytes>() {
        return Err(PyTypeError::new_err(
            "expected a list of documents; wrap a single document in a list or use encode()",
        ));
    }
    if let Ok(list) = inputs.cast::<PyList>() {
        return list.iter().map(|d| Doc::extract(&d)).collect();
    }
    inputs.try_iter()?.map(|d| Doc::extract(&d?)).collect()
}

fn extract_ids(ids: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(arr) = ids.extract::<PyReadonlyArray1<u32>>() {
        return Ok(arr.as_array().to_vec());
    }
    let values: Vec<i64> = if let Ok(arr) = ids.extract::<PyReadonlyArray1<i64>>() {
        arr.as_array().to_vec()
    } else if let Ok(arr) = ids.extract::<PyReadonlyArray1<i32>>() {
        arr.as_array().iter().map(|&v| v as i64).collect()
    } else {
        ids.extract::<Vec<i64>>()?
    };
    values
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| PyValueError::new_err(format!("invalid token id {v}"))))
        .collect()
}

fn side(value: &str, name: &str) -> PyResult<bool> {
    match value {
        "right" => Ok(false),
        "left" => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "{name} must be 'left' or 'right', got {other:?}"
        ))),
    }
}

#[pyclass(frozen, from_py_object, module = "nanotoken.nanotoken_rs")]
#[derive(Clone)]
struct TextFileSource {
    #[pyo3(get)]
    paths: Vec<PathBuf>,
    separator: Option<Vec<u8>>,
}

#[derive(FromPyObject)]
enum PathsArg {
    One(PathBuf),
    Many(Vec<PathBuf>),
}

impl PathsArg {
    fn into_vec(self) -> Vec<PathBuf> {
        match self {
            PathsArg::One(p) => vec![p],
            PathsArg::Many(ps) => ps,
        }
    }
}

#[derive(FromPyObject)]
enum SeparatorArg {
    Str(String),
    Bytes(Vec<u8>),
}

#[pymethods]
impl TextFileSource {
    #[new]
    #[pyo3(signature = (paths, separator = None))]
    fn new(paths: PathsArg, separator: Option<SeparatorArg>) -> PyResult<Self> {
        let separator = separator.map(|s| match s {
            SeparatorArg::Str(s) => s.into_bytes(),
            SeparatorArg::Bytes(b) => b,
        });
        if separator.as_ref().is_some_and(Vec::is_empty) {
            return Err(PyValueError::new_err("separator must not be empty"));
        }
        Ok(TextFileSource {
            paths: paths.into_vec(),
            separator,
        })
    }

    #[getter]
    fn separator<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.separator.as_ref().map(|s| PyBytes::new(py, s))
    }

    fn __repr__(&self) -> String {
        match &self.separator {
            Some(sep) => format!(
                "TextFileSource({} files, separator={:?})",
                self.paths.len(),
                String::from_utf8_lossy(sep)
            ),
            None => format!("TextFileSource({} files)", self.paths.len()),
        }
    }
}

#[pyclass(frozen, from_py_object, module = "nanotoken.nanotoken_rs")]
#[derive(Clone)]
struct JsonlFileSource {
    #[pyo3(get)]
    paths: Vec<PathBuf>,
    #[pyo3(get)]
    field: String,
}

#[pymethods]
impl JsonlFileSource {
    #[new]
    #[pyo3(signature = (paths, field = "text".to_string()))]
    fn new(paths: PathsArg, field: String) -> Self {
        JsonlFileSource {
            paths: paths.into_vec(),
            field,
        }
    }

    fn __repr__(&self) -> String {
        format!("JsonlFileSource({} files, field={:?})", self.paths.len(), self.field)
    }
}

fn file_source(source: &Bound<'_, PyAny>) -> PyResult<(Vec<PathBuf>, DocFormat)> {
    if let Ok(text) = source.extract::<TextFileSource>() {
        return Ok((
            text.paths,
            DocFormat::Text {
                separator: text.separator,
            },
        ));
    }
    if let Ok(jsonl) = source.extract::<JsonlFileSource>() {
        return Ok((jsonl.paths, DocFormat::Jsonl { field: jsonl.field }));
    }
    if let Ok(paths) = source.extract::<PathsArg>() {
        return Ok((paths.into_vec(), DocFormat::Text { separator: None }));
    }
    Err(PyTypeError::new_err(
        "expected a TextFileSource, JsonlFileSource, path, or list of paths",
    ))
}

type RaggedArrays<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<i64>>);

#[pyclass(frozen, module = "nanotoken.nanotoken_rs")]
struct WordPieceTokenizer {
    tok: Arc<Tokenizer>,
    pool: WorkerPool,
}

impl WordPieceTokenizer {
    fn run<T>(&self, py: Python<'_>, inputs: &Bound<'_, PyAny>, options: EncodeOptions) -> PyResult<Assembled<T>>
    where
        T: From<u32> + Copy + Send + Sync,
    {
        let docs = extract_docs(inputs)?;
        let views: Vec<&[u8]> = docs.iter().map(Doc::as_bytes).collect();
        py.detach(|| encode_docs(&self.tok, &self.pool, &views, &options))
            .map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn options(
        &self,
        add_special_tokens: bool,
        max_length: Option<usize>,
        truncation: bool,
        truncation_side: &str,
        padding: Option<&str>,
        pad_to_multiple_of: Option<usize>,
        padding_side: &str,
        pad_id: u32,
        want: Want,
        parallel: bool,
    ) -> PyResult<EncodeOptions> {
        let pad = match padding {
            None | Some("do_not_pad") => None,
            Some("longest") => Some(None),
            Some("max_length") => {
                Some(Some(max_length.ok_or_else(|| {
                    PyValueError::new_err("padding='max_length' requires max_length")
                })?))
            }
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "padding must be 'longest', 'max_length' or 'do_not_pad', got {other:?}"
                )));
            }
        };
        Ok(EncodeOptions {
            spec: RowSpec {
                add_special_tokens,
                max_length: if truncation { max_length } else { None },
                truncate_left: side(truncation_side, "truncation_side")?,
            },
            pad: match pad {
                Some(width) => Some(PadSpec {
                    width,
                    multiple_of: pad_to_multiple_of,
                    pad_left: side(padding_side, "padding_side")?,
                    pad_id,
                }),
                None => None,
            },
            want,
            parallel,
        })
    }
}

fn ragged_arrays<'py>(py: Python<'py>, out: Assembled<u32>) -> RaggedArrays<'py> {
    (out.ids.into_pyarray(py), out.offsets.into_pyarray(py))
}

fn lists<'py, T>(py: Python<'py>, values: &[T], offsets: &[i64]) -> PyResult<Bound<'py, PyList>>
where
    T: Copy + IntoPyObject<'py>,
{
    let rows = offsets
        .windows(2)
        .map(|w| PyList::new(py, values[w[0] as usize..w[1] as usize].iter().copied()))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, rows)
}

#[pymethods]
impl WordPieceTokenizer {
    /// Load from the contents of a HuggingFace tokenizer.json (str or bytes).
    #[staticmethod]
    fn from_json(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bytes: Vec<u8> = if let Ok(s) = data.extract::<PyBackedStr>() {
            s.as_bytes().to_vec()
        } else {
            data.extract::<Vec<u8>>()?
        };
        let tok = Tokenizer::from_json(&bytes).map_err(to_py_err)?;
        Ok(WordPieceTokenizer {
            tok: Arc::new(tok),
            pool: WorkerPool::default(),
        })
    }

    /// Token ids of one document as a uint32 array.
    #[pyo3(signature = (text, add_special_tokens = true))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
        add_special_tokens: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let doc = Doc::extract(text)?;
        let docs = [doc.as_bytes()];
        let spec = RowSpec {
            add_special_tokens,
            ..Default::default()
        };
        let out = py
            .detach(|| {
                let options = EncodeOptions {
                    spec,
                    parallel: false,
                    ..Default::default()
                };
                encode_docs::<u32>(&self.tok, &self.pool, &docs, &options)
            })
            .map_err(to_py_err)?;
        Ok(out.ids.into_pyarray(py))
    }

    /// Encode documents into one flat uint32 id array plus int64 row
    /// offsets (length n + 1). Optional truncation keeps at most
    /// `max_length` ids per row, special tokens included.
    #[pyo3(signature = (inputs, *, add_special_tokens = true, max_length = None, truncation_side = "right", parallel = true))]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
        add_special_tokens: bool,
        max_length: Option<usize>,
        truncation_side: &str,
        parallel: bool,
    ) -> PyResult<RaggedArrays<'py>> {
        let options = self.options(
            add_special_tokens,
            max_length,
            max_length.is_some(),
            truncation_side,
            None,
            None,
            "right",
            0,
            Want::default(),
            parallel,
        )?;
        let out = self.run::<u32>(py, inputs, options)?;
        Ok(ragged_arrays(py, out))
    }

    /// Encode, truncate and pad documents into model-ready int64 arrays.
    /// Returns a dict with `input_ids` and the requested masks: 2-D arrays
    /// when every row has the same length, otherwise lists of lists (with
    /// `as_lists`, always lists of lists).
    #[pyo3(signature = (
        inputs, *, add_special_tokens = true, max_length = None, truncation = false,
        truncation_side = "right", padding = None, pad_to_multiple_of = None, padding_side = "right",
        pad_id = 0, return_attention_mask = true, return_token_type_ids = true,
        return_special_tokens_mask = false, as_lists = false, parallel = true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn encode_batch_padded<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
        add_special_tokens: bool,
        max_length: Option<usize>,
        truncation: bool,
        truncation_side: &str,
        padding: Option<&str>,
        pad_to_multiple_of: Option<usize>,
        padding_side: &str,
        pad_id: u32,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        return_special_tokens_mask: bool,
        as_lists: bool,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let want = Want {
            attention_mask: return_attention_mask,
            type_ids: return_token_type_ids,
            special_mask: return_special_tokens_mask,
        };
        let options = self.options(
            add_special_tokens,
            max_length,
            truncation,
            truncation_side,
            padding,
            pad_to_multiple_of,
            padding_side,
            pad_id,
            want,
            parallel,
        )?;
        let out = self.run::<i64>(py, inputs, options)?;
        let dict = PyDict::new(py);
        let rows = out.offsets.len() - 1;
        let fields = [
            ("input_ids", Some(out.ids)),
            ("token_type_ids", out.type_ids),
            ("attention_mask", out.attention_mask),
            ("special_tokens_mask", out.special_mask),
        ];
        for (name, values) in fields {
            let Some(values) = values else { continue };
            match out.width.filter(|_| !as_lists) {
                Some(width) => {
                    let matrix = Array2::from_shape_vec((rows, width), values)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    dict.set_item(name, matrix.into_pyarray(py))?;
                }
                None => dict.set_item(name, lists(py, &values, &out.offsets)?)?,
            }
        }
        Ok(dict)
    }

    /// encode_batch returned as a list of lists of ints.
    #[pyo3(signature = (inputs, *, add_special_tokens = true, max_length = None, truncation_side = "right", parallel = true))]
    fn encode_batch_list<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
        add_special_tokens: bool,
        max_length: Option<usize>,
        truncation_side: &str,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let options = self.options(
            add_special_tokens,
            max_length,
            max_length.is_some(),
            truncation_side,
            None,
            None,
            "right",
            0,
            Want::default(),
            parallel,
        )?;
        let out = self.run::<u32>(py, inputs, options)?;
        lists(py, &out.ids, &out.offsets)
    }

    /// Encode every document of a TextFileSource / JsonlFileSource (or
    /// path / list of paths, each file one document), returning flat ids
    /// and row offsets like encode_batch.
    #[pyo3(signature = (source, *, add_special_tokens = true, parallel = true))]
    fn encode_files<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        add_special_tokens: bool,
        parallel: bool,
    ) -> PyResult<RaggedArrays<'py>> {
        let (paths, format) = file_source(source)?;
        let spec = RowSpec {
            add_special_tokens,
            ..Default::default()
        };
        let out = py
            .detach(|| {
                let files = paths
                    .iter()
                    .map(|p| MappedFile::open(p))
                    .collect::<crate::error::Result<Vec<_>>>()?;
                let options = EncodeOptions {
                    spec,
                    parallel,
                    ..Default::default()
                };
                encode_files::<u32>(&self.tok, &self.pool, &files, &format, &options)
            })
            .map_err(to_py_err)?;
        Ok(ragged_arrays(py, out))
    }

    #[pyo3(signature = (ids, skip_special_tokens = false))]
    fn decode(&self, ids: &Bound<'_, PyAny>, skip_special_tokens: bool) -> PyResult<String> {
        Ok(self.tok.decode(&extract_ids(ids)?, skip_special_tokens))
    }

    #[pyo3(signature = (sequences, skip_special_tokens = false))]
    fn decode_batch(&self, sequences: &Bound<'_, PyAny>, skip_special_tokens: bool) -> PyResult<Vec<String>> {
        sequences
            .try_iter()?
            .map(|seq| Ok(self.tok.decode(&extract_ids(&seq?)?, skip_special_tokens)))
            .collect()
    }

    /// Token strings of one document, without special tokens.
    fn tokenize(&self, text: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let doc = Doc::extract(text)?;
        let ids = self.tok.encode(&String::from_utf8_lossy(doc.as_bytes()));
        Ok(ids
            .iter()
            .map(|&id| self.tok.id_to_token(id).unwrap_or_default().to_string())
            .collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tok.token_to_id(token)
    }

    fn id_to_token(&self, id: i64) -> Option<String> {
        u32::try_from(id)
            .ok()
            .and_then(|id| self.tok.id_to_token(id))
            .map(str::to_string)
    }

    #[pyo3(signature = (with_added_tokens = true))]
    fn get_vocab<'py>(&self, py: Python<'py>, with_added_tokens: bool) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (token, id) in self.tok.vocab(with_added_tokens) {
            dict.set_item(token, id)?;
        }
        Ok(dict)
    }

    #[pyo3(signature = (with_added_tokens = true))]
    fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tok.vocab_size(with_added_tokens)
    }

    /// Added tokens as dicts in id order, mirroring tokenizer.json.
    #[getter]
    fn added_tokens<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut tokens: Vec<_> = self.tok.added.tokens.iter().collect();
        tokens.sort_by_key(|t| t.id);
        tokens
            .into_iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("id", t.id)?;
                d.set_item("content", &t.content)?;
                d.set_item("special", t.special)?;
                d.set_item("single_word", t.single_word)?;
                d.set_item("lstrip", t.lstrip)?;
                d.set_item("rstrip", t.rstrip)?;
                d.set_item("normalized", t.normalized)?;
                Ok(d)
            })
            .collect()
    }

    #[getter]
    fn prefix_ids(&self) -> Vec<u32> {
        self.tok.post.prefix.clone()
    }

    #[getter]
    fn suffix_ids(&self) -> Vec<u32> {
        self.tok.post.suffix.clone()
    }

    #[getter]
    fn unk_token(&self) -> String {
        self.tok.model.unk_token.clone()
    }

    #[getter]
    fn continuing_subword_prefix(&self) -> String {
        self.tok.model.prefix.clone()
    }

    /// Drop the per-worker word caches.
    fn clear_cache(&self) {
        self.pool.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "WordPieceTokenizer(vocab_size={}, added_tokens={})",
            self.tok.vocab_size(true),
            self.tok.added.tokens.len()
        )
    }
}

/// Maximum cached entries per level (whitespace chunks, words) per
/// encoding thread; takes effect for caches filled after the call.
#[pyfunction]
fn set_cache_entries(entries: usize) {
    crate::cache::set_capacity(entries);
}

#[pyfunction]
fn get_cache_entries() -> usize {
    crate::cache::capacity()
}

#[pymodule]
fn nanotoken_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_cache_entries, m)?)?;
    m.add_function(wrap_pyfunction!(get_cache_entries, m)?)?;
    m.add_class::<WordPieceTokenizer>()?;
    m.add_class::<TextFileSource>()?;
    m.add_class::<JsonlFileSource>()?;
    m.add(
        "UnsupportedTokenizerError",
        m.py().get_type::<UnsupportedTokenizerError>(),
    )?;
    Ok(())
}
