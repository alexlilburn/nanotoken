use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::added::AddedVocab;
use crate::decode::Decoder;
use crate::encoder::Encoder;
use crate::error::{Error, Result};
use crate::normalize::Normalizer;
use crate::postprocess::PostProcessor;
use crate::pretokenize::PreTokenizer;
use crate::wordpiece::WordPiece;

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub model: WordPiece,
    pub normalizer: Normalizer,
    pub pre_tokenizer: PreTokenizer,
    pub added: AddedVocab,
    pub post: PostProcessor,
    pub decoder: Decoder,
    id_to_token: Vec<Option<Box<str>>>,
    chunkable: bool,
}

pub const SEPARATORS: [u8; 4] = *b" \t\n\r";

impl Tokenizer {
    pub fn from_json(data: &[u8]) -> Result<Self> {
        let json: Value = serde_json::from_slice(data)?;
        let model = json
            .get("model")
            .ok_or_else(|| Error::Invalid("tokenizer.json without a `model`".into()))?;
        let model = WordPiece::from_json(model)?;
        let normalizer = Normalizer::from_json(json.get("normalizer"))?;
        let pre_tokenizer = PreTokenizer::from_json(json.get("pre_tokenizer"))?;
        let added = AddedVocab::from_json(json.get("added_tokens"), &normalizer)?;
        let post = PostProcessor::from_json(json.get("post_processor"))?;
        let decoder = Decoder::from_json(json.get("decoder"))?;
        Ok(Self::new(model, normalizer, pre_tokenizer, added, post, decoder))
    }

    pub fn new(
        model: WordPiece,
        normalizer: Normalizer,
        pre_tokenizer: PreTokenizer,
        added: AddedVocab,
        post: PostProcessor,
        decoder: Decoder,
    ) -> Self {
        let size = model.vocab.values().map(|&id| id as usize + 1).max().unwrap_or(0);
        let mut id_to_token = vec![None; size];
        for (token, &id) in &model.vocab {
            id_to_token[id as usize] = Some(token.clone().into_boxed_str());
        }
        // Chunking at ASCII whitespace is exact only if (1) every pre-token
        // is whitespace-free, (2) the normalizer keeps these separators as
        // whitespace (clean_text *deletes* \x0b/\x0c, which therefore join
        // their neighbors and are not separators), and (3) no added token
        // is matched after normalization, where it could span a separator.
        let separators_survive = SEPARATORS.iter().all(|&b| {
            let normalized = normalizer.normalize(&(b as char).to_string());
            !normalized.is_empty() && normalized.chars().all(char::is_whitespace)
        });
        let chunkable = pre_tokenizer.splits_on_whitespace() && separators_survive && !added.has_normalized();
        Tokenizer {
            model,
            normalizer,
            pre_tokenizer,
            added,
            post,
            decoder,
            id_to_token,
            chunkable,
        }
    }

    pub fn chunkable(&self) -> bool {
        self.chunkable
    }

    pub fn encoder(&self) -> Encoder<'_> {
        Encoder::new(self, Default::default())
    }

    /// Token ids of `text` without special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encoder().encode(text, &mut out);
        out
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added
            .by_content
            .get(token)
            .or_else(|| self.model.vocab.get(token))
            .copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.added
            .decoded_content(id)
            .or_else(|| self.id_to_token.get(id as usize).and_then(|t| t.as_deref()))
    }

    pub fn vocab(&self, with_added: bool) -> FxHashMap<&str, u32> {
        let mut vocab: FxHashMap<&str, u32> = self.model.vocab.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        if with_added {
            for token in &self.added.tokens {
                vocab.insert(token.content.as_str(), token.id);
            }
        }
        vocab
    }

    pub fn vocab_size(&self, with_added: bool) -> usize {
        if with_added {
            self.vocab(true).len()
        } else {
            self.model.vocab.len()
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let tokens: Vec<&str> = ids
            .iter()
            .filter_map(|&id| self.id_to_token(id))
            .filter(|token| !skip_special_tokens || !self.added.is_special(token))
            .collect();
        self.decoder.decode(&tokens)
    }
}
