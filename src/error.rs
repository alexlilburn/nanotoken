use std::fmt;

#[derive(Debug)]
pub enum Error {
    Unsupported(String),
    Invalid(String),
    Io(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(msg) => write!(f, "unsupported tokenizer: {msg}"),
            Error::Invalid(msg) => write!(f, "{msg}"),
            Error::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Invalid(format!("invalid tokenizer.json: {err}"))
    }
}
