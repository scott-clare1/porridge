use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub type Embedding = Vec<f32>;

#[derive(Serialize, Deserialize, Clone)]
pub struct EmbeddingEntry {
    pub values: Embedding,
}

#[derive(Clone)]
pub struct Database {
    pub contents: Arc<Mutex<HashMap<Uuid, EmbeddingEntry>>>,
}
