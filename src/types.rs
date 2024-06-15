use serde::{Deserialize, Serialize};
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
