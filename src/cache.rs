use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap;

pub const MAX_SHORT_KEY: usize = 15;
pub const MAX_LONG_KEY: usize = 64;
const DEFAULT_ENTRIES: usize = 1 << 20;
const EMPTY: u128 = 0;

static CAPACITY: AtomicUsize = AtomicUsize::new(0);

/// Maximum entries per cache level per worker (chunks and words each).
/// Full caches take up to about 130 MiB per encoding thread.
pub fn capacity() -> usize {
    match CAPACITY.load(Ordering::Relaxed) {
        0 => {
            let initial = std::env::var("NANOTOKEN_CACHE_ENTRIES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_ENTRIES);
            CAPACITY.store(initial, Ordering::Relaxed);
            initial
        }
        n => n,
    }
}

pub fn set_capacity(entries: usize) {
    CAPACITY.store(entries.max(1), Ordering::Relaxed);
}

/// Up to two ids inline; longer encodings live in the cache's arena.
#[derive(Clone, Copy, Default)]
pub struct Cached {
    len: u32,
    a: u32,
    b: u32,
}

/// A key of at most 15 bytes packed into a u128 with its length in the top
/// byte, so keys of different lengths never collide and no key is `EMPTY`.
#[inline]
pub fn pack_key(bytes: &[u8]) -> u128 {
    debug_assert!(!bytes.is_empty() && bytes.len() <= MAX_SHORT_KEY);
    let mut buf = [0u8; 16];
    buf[..bytes.len()].copy_from_slice(bytes);
    buf[15] = bytes.len() as u8;
    u128::from_le_bytes(buf)
}

/// `pack_key(&buf[start..start + len])`, reading 16 bytes at once when the
/// buffer extends that far.
#[inline]
pub fn pack_key_at(buf: &[u8], start: usize, len: usize) -> u128 {
    match buf.get(start..start + 16) {
        Some(window) => {
            let word = u128::from_le_bytes(window.try_into().expect("16 bytes"));
            let mask = (1u128 << (8 * len)) - 1;
            (word & mask) | ((len as u128) << 120)
        }
        None => pack_key(&buf[start..start + len]),
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Slot {
    key: u128,
    value: Cached,
}

/// Linear-probing table from packed keys to cached encodings. Slots are 32
/// bytes, two per cache line, so a probe usually touches one line; the
/// table is kept at most half full.
struct Table {
    slots: Vec<Slot>,
    len: usize,
    shift: u32,
}

impl Default for Table {
    fn default() -> Self {
        Table {
            slots: vec![
                Slot {
                    key: EMPTY,
                    value: Cached::default(),
                };
                1024
            ],
            len: 0,
            shift: 64 - 10,
        }
    }
}

#[inline]
fn hash(key: u128) -> u64 {
    let folded = (key as u64) ^ ((key >> 64) as u64).rotate_left(29);
    folded.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[inline(always)]
fn prefetch<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetches are hints and never fault.
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prefetches are hints and never fault.
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

impl Table {
    #[inline]
    fn prefetch(&self, key: u128) {
        let i = (hash(key) >> self.shift) as usize;
        prefetch(self.slots.as_ptr().wrapping_add(i));
    }

    #[inline]
    fn get(&self, key: u128) -> Option<Cached> {
        let mask = self.slots.len() - 1;
        let mut i = (hash(key) >> self.shift) as usize;
        loop {
            let slot = &self.slots[i];
            if slot.key == key {
                return Some(slot.value);
            }
            if slot.key == EMPTY {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    fn insert(&mut self, key: u128, value: Cached) {
        if (self.len + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = (hash(key) >> self.shift) as usize;
        while self.slots[i].key != EMPTY {
            if self.slots[i].key == key {
                return;
            }
            i = (i + 1) & mask;
        }
        self.slots[i] = Slot { key, value };
        self.len += 1;
    }

    fn grow(&mut self) {
        let doubled = vec![
            Slot {
                key: EMPTY,
                value: Cached::default(),
            };
            self.slots.len() * 2
        ];
        let old = std::mem::replace(&mut self.slots, doubled);
        self.len = 0;
        self.shift -= 1;
        for slot in old {
            if slot.key != EMPTY {
                self.insert(slot.key, slot.value);
            }
        }
    }
}

/// Two cache levels: raw whitespace-delimited chunks (e.g. `"world,"`)
/// and normalized pre-tokens (`"world"`, `","`), which have far fewer
/// distinct values, so a chunk miss is usually still a word hit. Chunks of
/// 16 to 64 bytes go to a boxed-key map; longer inputs are not cached. A
/// full level stops admitting entries: the frequent head of the Zipf
/// distribution is seen first.
#[derive(Default)]
pub struct WordCache {
    chunks: Table,
    long_chunks: FxHashMap<Box<[u8]>, Cached>,
    words: Table,
    arena: Vec<u32>,
}

impl WordCache {
    fn store(&mut self, ids: &[u32]) -> Cached {
        match *ids {
            [] => Cached { len: 0, a: 0, b: 0 },
            [a] => Cached { len: 1, a, b: 0 },
            [a, b] => Cached { len: 2, a, b },
            _ => {
                let start = self.arena.len() as u32;
                self.arena.extend_from_slice(ids);
                Cached {
                    len: ids.len() as u32,
                    a: start,
                    b: 0,
                }
            }
        }
    }

    #[inline]
    pub fn emit(&self, cached: Cached, out: &mut Vec<u32>) {
        match cached.len {
            0 => {}
            1 => out.push(cached.a),
            2 => out.extend_from_slice(&[cached.a, cached.b]),
            n => out.extend_from_slice(&self.arena[cached.a as usize..(cached.a + n) as usize]),
        }
    }

    #[inline]
    pub fn prefetch_chunk(&self, key: u128) {
        self.chunks.prefetch(key);
    }

    #[inline]
    pub fn chunk(&self, key: u128) -> Option<Cached> {
        self.chunks.get(key)
    }

    pub fn put_chunk(&mut self, key: u128, ids: &[u32]) {
        if self.chunks.len + self.long_chunks.len() < capacity() {
            let cached = self.store(ids);
            self.chunks.insert(key, cached);
        }
    }

    #[inline]
    pub fn long_chunk(&self, bytes: &[u8]) -> Option<Cached> {
        self.long_chunks.get(bytes).copied()
    }

    pub fn put_long_chunk(&mut self, bytes: &[u8], ids: &[u32]) {
        if self.chunks.len + self.long_chunks.len() < capacity() {
            let cached = self.store(ids);
            self.long_chunks.insert(bytes.into(), cached);
        }
    }

    #[inline]
    pub fn word(&self, key: u128) -> Option<Cached> {
        self.words.get(key)
    }

    pub fn put_word(&mut self, key: u128, ids: &[u32]) {
        if self.words.len < capacity() {
            let cached = self.store(ids);
            self.words.insert(key, cached);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_keys() {
        let buf = b"hello world, this is long enough";
        for start in 0..buf.len() {
            for len in 1..=MAX_SHORT_KEY.min(buf.len() - start) {
                assert_eq!(pack_key_at(buf, start, len), pack_key(&buf[start..start + len]));
            }
        }
        assert_ne!(pack_key(b"a"), pack_key(b"a\0"));
    }

    #[test]
    fn table_grows_and_finds() {
        let mut cache = WordCache::default();
        for i in 1..5000u32 {
            cache.put_word(pack_key(&i.to_le_bytes()), &[i, i + 1, i + 2][..(i % 3 + 1) as usize]);
        }
        for i in 1..5000u32 {
            let mut out = Vec::new();
            cache.emit(cache.word(pack_key(&i.to_le_bytes())).unwrap(), &mut out);
            assert_eq!(out, [i, i + 1, i + 2][..(i % 3 + 1) as usize]);
        }
        assert!(cache.word(pack_key(b"missing")).is_none());
    }
}
