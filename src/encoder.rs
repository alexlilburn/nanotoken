use crate::added::Segment;
use crate::cache::{MAX_LONG_KEY, MAX_SHORT_KEY, WordCache, pack_key, pack_key_at};
use crate::normalize::DROP;
use crate::tokenizer::Tokenizer;

#[inline]
fn is_separator(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[inline]
fn find_separator(bytes: &[u8], mut i: usize) -> usize {
    const LOW: u64 = 0x2121_2121_2121_2121;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    while let Some(window) = bytes.get(i..i + 8) {
        let word = u64::from_le_bytes(window.try_into().expect("8 bytes"));
        // Flags bytes below 0x21 (every separator is); the lowest flag is
        // exact, and other control bytes are skipped past.
        let below = word.wrapping_sub(LOW) & !word & HIGH;
        if below == 0 {
            i += 8;
            continue;
        }
        let j = i + (below.trailing_zeros() / 8) as usize;
        if is_separator(bytes[j]) {
            return j;
        }
        i = j + 1;
    }
    while i < bytes.len() && !is_separator(bytes[i]) {
        i += 1;
    }
    i
}

fn tokenize_pieces(
    tok: &Tokenizer,
    s: &str,
    pieces: &mut Vec<(usize, usize)>,
    scratch: &mut Vec<(usize, usize)>,
    cache: &mut WordCache,
    out: &mut Vec<u32>,
) {
    tok.pre_tokenizer.split(s, pieces, scratch);
    for &(start, end) in pieces.iter() {
        tokenize_piece(tok, &s[start..end], cache, out);
    }
}

#[inline]
fn tokenize_piece(tok: &Tokenizer, piece: &str, cache: &mut WordCache, out: &mut Vec<u32>) {
    if piece.len() > MAX_SHORT_KEY {
        tok.model.tokenize(piece, out);
        return;
    }
    let key = pack_key(piece.as_bytes());
    if let Some(cached) = cache.word(key) {
        cache.emit(cached, out);
        return;
    }
    let before = out.len();
    tok.model.tokenize(piece, out);
    cache.put_word(key, &out[before..]);
}

const LOOKAHEAD: usize = 64;

pub struct Encoder<'t> {
    tok: &'t Tokenizer,
    cache: WordCache,
    ascii_buf: Vec<u8>,
    pieces: Vec<(usize, usize)>,
    scratch: Vec<(usize, usize)>,
    segments: Vec<Segment>,
    inner_segments: Vec<Segment>,
}

impl<'t> Encoder<'t> {
    pub fn new(tok: &'t Tokenizer, cache: WordCache) -> Self {
        Encoder {
            tok,
            cache,
            ascii_buf: Vec::new(),
            pieces: Vec::new(),
            scratch: Vec::new(),
            segments: Vec::new(),
            inner_segments: Vec::new(),
        }
    }

    pub fn into_cache(self) -> WordCache {
        self.cache
    }

    /// Appends the ids of `doc`, without post-processor special tokens.
    pub fn encode(&mut self, doc: &str, out: &mut Vec<u32>) {
        if !self.tok.added.has_raw() {
            self.encode_text(doc, out);
            return;
        }
        let mut segments = std::mem::take(&mut self.segments);
        self.tok.added.split_raw(doc, &mut segments);
        for &segment in &segments {
            match segment {
                Segment::Token(id, _, _) => out.push(id),
                Segment::Text(start, end) => self.encode_text(&doc[start..end], out),
            }
        }
        self.segments = segments;
    }

    fn encode_text(&mut self, text: &str, out: &mut Vec<u32>) {
        if self.tok.chunkable() {
            self.encode_chunks(text, out);
        } else {
            self.encode_general(text, out);
        }
    }

    fn encode_general(&mut self, text: &str, out: &mut Vec<u32>) {
        let normalized = self.tok.normalizer.normalize(text);
        if !self.tok.added.has_normalized() {
            tokenize_pieces(
                self.tok,
                &normalized,
                &mut self.pieces,
                &mut self.scratch,
                &mut self.cache,
                out,
            );
            return;
        }
        let mut segments = std::mem::take(&mut self.inner_segments);
        self.tok.added.split_normalized(&normalized, &mut segments);
        for &segment in &segments {
            match segment {
                Segment::Token(id, _, _) => out.push(id),
                Segment::Text(start, end) => tokenize_pieces(
                    self.tok,
                    &normalized[start..end],
                    &mut self.pieces,
                    &mut self.scratch,
                    &mut self.cache,
                    out,
                ),
            }
        }
        self.inner_segments = segments;
    }

    /// Finds up to LOOKAHEAD chunks and prefetches their cache slots before
    /// looking any of them up, so rare chunks' DRAM misses overlap.
    fn encode_chunks(&mut self, text: &str, out: &mut Vec<u32>) {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut spans = [(0usize, 0usize); LOOKAHEAD];
        let mut keys = [0u128; LOOKAHEAD];
        let mut i = 0;
        loop {
            let mut count = 0;
            while count < LOOKAHEAD {
                while i < n && is_separator(bytes[i]) {
                    i += 1;
                }
                if i >= n {
                    break;
                }
                let start = i;
                i = find_separator(bytes, i + 1);
                let len = i - start;
                spans[count] = (start, i);
                if len <= MAX_SHORT_KEY {
                    keys[count] = pack_key_at(bytes, start, len);
                    self.cache.prefetch_chunk(keys[count]);
                }
                count += 1;
            }
            for k in 0..count {
                let (start, end) = spans[k];
                if end - start > MAX_SHORT_KEY {
                    self.encode_long_chunk(&text[start..end], out);
                    continue;
                }
                match self.cache.chunk(keys[k]) {
                    Some(cached) => self.cache.emit(cached, out),
                    None => {
                        let before = out.len();
                        self.compute_chunk(&text[start..end], out);
                        self.cache.put_chunk(keys[k], &out[before..]);
                    }
                }
            }
            if count < LOOKAHEAD {
                break;
            }
        }
    }

    fn encode_long_chunk(&mut self, chunk: &str, out: &mut Vec<u32>) {
        let bytes = chunk.as_bytes();
        if bytes.len() > MAX_LONG_KEY {
            self.compute_chunk(chunk, out);
            return;
        }
        if let Some(cached) = self.cache.long_chunk(bytes) {
            self.cache.emit(cached, out);
            return;
        }
        let before = out.len();
        self.compute_chunk(chunk, out);
        self.cache.put_long_chunk(bytes, &out[before..]);
    }

    /// BertPreTokenizer over an ASCII chunk in one pass: normalize each
    /// byte through the table, cut words at whitespace, isolate punctuation.
    fn compute_bert_ascii(&mut self, chunk: &[u8], table: &[u8; 128], out: &mut Vec<u32>) {
        let tok = self.tok;
        self.ascii_buf.clear();
        for &b in chunk {
            let m = table[b as usize];
            if m == DROP {
                continue;
            }
            if m.is_ascii_punctuation() || m.is_ascii_whitespace() || m == 0x0b {
                if !self.ascii_buf.is_empty() {
                    let word = std::str::from_utf8(&self.ascii_buf).expect("ASCII is UTF-8");
                    tokenize_piece(tok, word, &mut self.cache, out);
                    self.ascii_buf.clear();
                }
                if m.is_ascii_punctuation() {
                    let punct = [m];
                    let punct = std::str::from_utf8(&punct).expect("ASCII is UTF-8");
                    tokenize_piece(tok, punct, &mut self.cache, out);
                }
            } else {
                self.ascii_buf.push(m);
            }
        }
        if !self.ascii_buf.is_empty() {
            let word = std::str::from_utf8(&self.ascii_buf).expect("ASCII is UTF-8");
            tokenize_piece(tok, word, &mut self.cache, out);
        }
    }

    fn compute_chunk(&mut self, chunk: &str, out: &mut Vec<u32>) {
        let tok = self.tok;
        if let Some(table) = tok.normalizer.ascii_table().filter(|_| chunk.is_ascii()) {
            if tok.pre_tokenizer.is_bert() {
                self.compute_bert_ascii(chunk.as_bytes(), table, out);
                return;
            }
            self.ascii_buf.clear();
            self.ascii_buf
                .extend(chunk.bytes().map(|b| table[b as usize]).filter(|&b| b != DROP));
            let normalized = std::str::from_utf8(&self.ascii_buf).expect("ASCII is UTF-8");
            tokenize_pieces(
                tok,
                normalized,
                &mut self.pieces,
                &mut self.scratch,
                &mut self.cache,
                out,
            );
        } else {
            let normalized = tok.normalizer.normalize(chunk);
            tokenize_pieces(
                tok,
                &normalized,
                &mut self.pieces,
                &mut self.scratch,
                &mut self.cache,
                out,
            );
        }
    }

    /// Reference path with no chunking, caching or ASCII table.
    #[cfg(test)]
    pub fn encode_uncached(&mut self, doc: &str, out: &mut Vec<u32>) {
        let mut segments = Vec::new();
        self.tok.added.split_raw(doc, &mut segments);
        for segment in segments {
            match segment {
                Segment::Token(id, _, _) => out.push(id),
                Segment::Text(start, end) => {
                    let normalized = self.tok.normalizer.normalize(&doc[start..end]);
                    let mut inner = Vec::new();
                    self.tok.added.split_normalized(&normalized, &mut inner);
                    for piece in inner {
                        match piece {
                            Segment::Token(id, _, _) => out.push(id),
                            Segment::Text(a, b) => {
                                let s = &normalized[a..b];
                                self.tok.pre_tokenizer.split(s, &mut self.pieces, &mut self.scratch);
                                for &(x, y) in self.pieces.iter() {
                                    self.tok.model.tokenize(&s[x..y], out);
                                }
                            }
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

    #[test]
    fn separator_scan() {
        let text = b"abc\x0bdefghij\x01klmnop qrs\ttu\nvw\rxyz0123456789";
        for start in 0..text.len() {
            let expected = (start..text.len())
                .find(|&i| is_separator(text[i]))
                .unwrap_or(text.len());
            assert_eq!(find_separator(text, start), expected, "from {start}");
        }
    }
}
