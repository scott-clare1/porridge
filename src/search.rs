use serde::Serialize;
use uuid::Uuid;

use crate::similarity::{MetricType, SimilarityMetric};
use crate::types::{Database, Embedding};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Serialize, Clone, Debug)]
pub struct Neighbour {
    pub uuid: Uuid,
    similarity: f32,
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

#[derive(Clone)]
pub struct KNN {
    database: Database,
    k: usize,
    metric: MetricType,
}

impl KNN {
    pub fn new(database: Database, k: usize, metric: MetricType) -> Self {
        Self {
            database,
            k,
            metric,
        }
    }

    fn insert_neighbour(
        &self,
        mut heap: BinaryHeap<Neighbour>,
        neighour: Neighbour,
    ) -> BinaryHeap<Neighbour> {
        if heap.len() < self.k {
            heap.push(neighour);
        } else if neighour.similarity > heap.peek().unwrap().similarity {
            heap.pop();
            heap.push(neighour);
        }
        heap
    }

    pub fn search(&self, query_vector: &Embedding) -> Vec<Neighbour> {
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let database = self.database.contents.lock().unwrap();
        for (uuid, vector) in database.iter() {
            let similarity = self.metric.similarity(&vector.values, query_vector);
            let neighbour = Neighbour {
                uuid: *uuid,
                similarity,
            };

            heap = self.insert_neighbour(heap, neighbour);
        }
        heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod test_knn {

    use super::*;
    use crate::similarity::CosineSimilarity;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn test_insert_neighbour_to_empty_heap() {
        let database: Database = Database {
            contents: Arc::new(Mutex::new(HashMap::new())),
        };
        let search = KNN {
            database,
            k: 2 as usize,
            metric: MetricType::Cosine(CosineSimilarity),
        };
        let heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let neighbour = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.1,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(1, actual_heap.len());
        assert_eq!(0.1, actual_heap.peek().unwrap().similarity)
    }

    #[test]
    fn test_insert_neighbour_to_non_empty_heap() {
        let database: Database = Database {
            contents: Arc::new(Mutex::new(HashMap::new())),
        };
        let search = KNN {
            database,
            k: 2 as usize,
            metric: MetricType::Cosine(CosineSimilarity),
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let id = Uuid::new_v4();
        heap.push(Neighbour {
            uuid: id,
            similarity: 0.1,
        });
        let neighbour = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.2,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0.1, actual_heap.peek().unwrap().similarity);
    }

    #[test]
    fn test_insert_neighbour_into_full_heap() {
        let database: Database = Database {
            contents: Arc::new(Mutex::new(HashMap::new())),
        };
        let search = KNN {
            database,
            k: 2 as usize,
            metric: MetricType::Cosine(CosineSimilarity),
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let id = Uuid::new_v4();
        heap.push(Neighbour {
            uuid: id,
            similarity: 0.5,
        });
        heap.push(Neighbour {
            uuid: id,
            similarity: 0.2,
        });
        let neighbour = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.6,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0.5, actual_heap.peek().unwrap().similarity);
    }

    #[test]
    fn test_insert_non_largest_neighbour_into_full_heap() {
        let database: Database = Database {
            contents: Arc::new(Mutex::new(HashMap::new())),
        };
        let search = KNN {
            database,
            k: 2 as usize,
            metric: MetricType::Cosine(CosineSimilarity),
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let id = Uuid::new_v4();
        heap.push(Neighbour {
            uuid: id,
            similarity: 0.5,
        });
        heap.push(Neighbour {
            uuid: id,
            similarity: 0.2,
        });
        let neighbour = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.3,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0.3, actual_heap.peek().unwrap().similarity);
    }
}
