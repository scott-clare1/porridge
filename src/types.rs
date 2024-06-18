use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub type Embedding = Vec<f32>;
pub type Database = Arc<Mutex<HashMap<Uuid, EmbeddingEntry>>>;

#[derive(Serialize, Deserialize, Clone)]
pub struct EmbeddingEntry {
    pub embeddings: Embedding,
    pub text: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct Neighbour {
    pub uuid: Uuid,
    pub similarity: f32,
}

impl Eq for Neighbour {}

impl PartialEq for Neighbour {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Neighbour) -> Option<Ordering> {
        other.similarity.partial_cmp(&self.similarity)
    }
}

impl Ord for Neighbour {
    fn cmp(&self, other: &Neighbour) -> Ordering {
        self.cmp(other)
    }
}

#[cfg(test)]
mod test_neighbour {
    use super::*;

    #[test]
    fn test_partial_ord() {
        let neighbour_a = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.9,
        };
        let neighbour_b = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.8,
        };
        assert!(neighbour_a < neighbour_b);
    }
}
