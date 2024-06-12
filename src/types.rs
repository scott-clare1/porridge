use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub type Embedding = Vec<f32>;

#[derive(Serialize, Deserialize, Clone)]
pub struct Vector {
    pub id: Uuid,
    pub values: Embedding,
}

#[derive(Clone)]
pub struct Database {
    pub vectors: Arc<Mutex<HashMap<Uuid, Vector>>>,
}
