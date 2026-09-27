use std::time::Instant;

use nanotoken_rs::{DocFormat, EncodeOptions, MappedFile, Tokenizer, WorkerPool, encode_files};

fn main() {
    let tokenizer_path = std::env::var("NANOTOKEN_BENCH_TOKENIZER")
        .unwrap_or_else(|_| "tests/fixtures/bert-base-uncased.json".to_string());
    let Ok(corpus) = std::env::var("NANOTOKEN_BENCH_FILE") else {
        eprintln!("set NANOTOKEN_BENCH_FILE to a text corpus (documents separated by <|endoftext|>)");
        return;
    };
    let separator = std::env::var("NANOTOKEN_BENCH_SEPARATOR").unwrap_or_else(|_| "<|endoftext|>".to_string());
    let tok = Tokenizer::from_json(&std::fs::read(tokenizer_path).unwrap()).unwrap();
    let files = vec![MappedFile::open(corpus.as_ref()).unwrap()];
    let bytes = files[0].bytes().len() as f64;
    let format = DocFormat::Text {
        separator: Some(separator.into_bytes()),
    };
    for parallel in [false, true] {
        let pool = WorkerPool::default();
        for run in 0..3 {
            let start = Instant::now();
            let options = EncodeOptions {
                parallel,
                ..Default::default()
            };
            let out = encode_files::<u32>(&tok, &pool, &files, &format, &options).unwrap();
            let secs = start.elapsed().as_secs_f64();
            println!(
                "{} run {run}: {:.0} MB/s, {} tokens",
                if parallel { "parallel" } else { "serial" },
                bytes / secs / 1e6,
                out.ids.len()
            );
        }
    }
}
