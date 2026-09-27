use std::collections::BTreeMap;

const NO_TOKEN: u32 = u32::MAX;
const NO_NODE: u32 = 0;

#[derive(Clone, Copy, Debug)]
struct Node {
    token: u32,
    edge_start: u32,
    edge_len: u32,
}

/// Byte trie over vocabulary pieces answering longest-prefix queries.
/// Children of the root are indexed directly; deeper edges are stored
/// sorted per node.
#[derive(Clone, Debug)]
pub struct Trie {
    root: Box<[u32; 256]>,
    root_token: [u32; 256],
    nodes: Vec<Node>,
    edge_bytes: Vec<u8>,
    edge_nodes: Vec<u32>,
}

#[derive(Default)]
struct BuildNode {
    token: Option<u32>,
    children: BTreeMap<u8, usize>,
}

impl Trie {
    pub fn new<'a>(entries: impl IntoIterator<Item = (&'a [u8], u32)>) -> Self {
        let mut build = vec![BuildNode::default()];
        for (key, id) in entries {
            if key.is_empty() {
                continue;
            }
            let mut node = 0;
            for &b in key {
                node = match build[node].children.get(&b) {
                    Some(&child) => child,
                    None => {
                        build.push(BuildNode::default());
                        let child = build.len() - 1;
                        build[node].children.insert(b, child);
                        child
                    }
                };
            }
            build[node].token = Some(id);
        }

        let mut trie = Trie {
            root: Box::new([NO_NODE; 256]),
            root_token: [NO_TOKEN; 256],
            nodes: vec![Node {
                token: NO_TOKEN,
                edge_start: 0,
                edge_len: 0,
            }],
            edge_bytes: Vec::new(),
            edge_nodes: Vec::new(),
        };
        let mut index = vec![NO_NODE; build.len()];
        let mut order = Vec::with_capacity(build.len());
        for (&b, &child) in &build[0].children {
            index[child] = trie.nodes.len() as u32;
            trie.root[b as usize] = index[child];
            trie.root_token[b as usize] = build[child].token.unwrap_or(NO_TOKEN);
            trie.nodes.push(Node {
                token: build[child].token.unwrap_or(NO_TOKEN),
                edge_start: 0,
                edge_len: 0,
            });
            order.push(child);
        }
        let mut cursor = 0;
        while cursor < order.len() {
            let node = order[cursor];
            cursor += 1;
            let at = index[node] as usize;
            trie.nodes[at].edge_start = trie.edge_bytes.len() as u32;
            trie.nodes[at].edge_len = build[node].children.len() as u32;
            let mut pending = Vec::with_capacity(build[node].children.len());
            for (&b, &child) in &build[node].children {
                index[child] = (trie.nodes.len() + pending.len()) as u32;
                trie.edge_bytes.push(b);
                trie.edge_nodes.push(index[child]);
                pending.push(child);
            }
            for child in pending {
                trie.nodes.push(Node {
                    token: build[child].token.unwrap_or(NO_TOKEN),
                    edge_start: 0,
                    edge_len: 0,
                });
                order.push(child);
            }
        }
        trie
    }

    #[inline]
    fn child(&self, node: u32, b: u8) -> u32 {
        let n = self.nodes[node as usize];
        let start = n.edge_start as usize;
        let bytes = &self.edge_bytes[start..start + n.edge_len as usize];
        let pos = if bytes.len() <= 8 {
            bytes.iter().position(|&x| x == b)
        } else {
            bytes.binary_search(&b).ok()
        };
        match pos {
            Some(i) => self.edge_nodes[start + i],
            None => NO_NODE,
        }
    }

    /// Longest vocabulary piece that is a prefix of `s`, as (byte length, id).
    #[inline]
    pub fn longest_prefix(&self, s: &[u8]) -> Option<(usize, u32)> {
        let (&first, rest) = s.split_first()?;
        let mut node = self.root[first as usize];
        if node == NO_NODE {
            return None;
        }
        let mut best = match self.root_token[first as usize] {
            NO_TOKEN => None,
            id => Some((1, id)),
        };
        for (i, &b) in rest.iter().enumerate() {
            node = self.child(node, b);
            if node == NO_NODE {
                break;
            }
            let token = self.nodes[node as usize].token;
            if token != NO_TOKEN {
                best = Some((i + 2, token));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_prefix() {
        let entries: Vec<(&[u8], u32)> = vec![(b"a", 0), (b"ab", 1), (b"abcd", 2), (b"b", 3)];
        let trie = Trie::new(entries);
        assert_eq!(trie.longest_prefix(b"abc"), Some((2, 1)));
        assert_eq!(trie.longest_prefix(b"abcde"), Some((4, 2)));
        assert_eq!(trie.longest_prefix(b"bx"), Some((1, 3)));
        assert_eq!(trie.longest_prefix(b"x"), None);
        assert_eq!(trie.longest_prefix(b""), None);
    }
}
