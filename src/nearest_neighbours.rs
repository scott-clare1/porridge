use crate::search::SearchAlgorithm;
use crate::types::{Database, Embedding, Neighbour};

pub trait KNNInterface {
    fn search(&self, database: &Database, query_vector: &Embedding) -> Vec<Neighbour>;
}

#[derive(Clone)]
pub struct KNNAlgortihm {
    pub database: Database,
    pub algorithm: SearchAlgorithm,
}

impl KNNAlgortihm {
    pub fn new(database: Database, algorithm: SearchAlgorithm) -> Self {
        Self {
            database,
            algorithm,
        }
    }

    pub fn search(&self, query_vector: &Embedding) -> Vec<Neighbour> {
        self.algorithm.search(&self.database, query_vector)
    }
}
