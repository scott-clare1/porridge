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

struct KLargestNeighboursHeap {
    heap: BinaryHeap<Neighbour>,
    k: usize,
}

impl KLargestNeighboursHeap {
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            k,
        }
    }

    fn push(&mut self, neighbour: Neighbour) {
        if self.len() < self.k {
            self.heap.push(neighbour);
        } else if neighbour.similarity > self.peek().unwrap().similarity {
            self.pop();
            self.heap.push(neighbour);
        }
    }

    fn pop(&mut self) -> Option<Neighbour> {
        self.heap.pop()
    }

    fn peek(&self) -> Option<&Neighbour> {
        self.heap.peek()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn into_sorted_vec(self) -> Vec<Neighbour> {
        self.heap.into_sorted_vec()
    }
}

#[derive(Clone)]
pub struct KNN {
    pub database: Database,
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

    pub fn search(&self, query_vector: &Embedding) -> Vec<Neighbour> {
        let mut heap: KLargestNeighboursHeap = KLargestNeighboursHeap::new(self.k);
        let database = self.database.lock().unwrap();
        for (uuid, vector) in database.iter() {
            let similarity = self.metric.similarity(&vector.embeddings, query_vector);
            let neighbour = Neighbour {
                uuid: *uuid,
                similarity,
            };

            heap.push(neighbour);
        }
        heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod test_knn {

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

    #[test]
    fn test_insert_neighbour_to_empty_heap() {
        let mut heap: KLargestNeighboursHeap = KLargestNeighboursHeap::new(2 as usize);
        let neighbour = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.1,
        };
        heap.push(neighbour);
        assert_eq!(1, heap.len());
        assert_eq!(0.1, heap.peek().unwrap().similarity)
    }

    #[test]
    fn test_insert_neighbour_to_non_empty_heap() {
        let mut heap: KLargestNeighboursHeap = KLargestNeighboursHeap::new(2 as usize);
        let id = Uuid::new_v4();
        heap.push(Neighbour {
            uuid: id,
            similarity: 0.1,
        });
        let neighbour = Neighbour {
            uuid: Uuid::new_v4(),
            similarity: 0.2,
        };
        heap.push(neighbour);
        assert_eq!(2, heap.len());
        assert_eq!(0.1, heap.peek().unwrap().similarity);
    }

    #[test]
    fn test_insert_neighbour_into_full_heap() {
        let mut heap: KLargestNeighboursHeap = KLargestNeighboursHeap::new(2 as usize);
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
        heap.push(neighbour);
        assert_eq!(2, heap.len());
        assert_eq!(0.5, heap.peek().unwrap().similarity);
    }

    #[test]
    fn test_insert_non_largest_neighbour_into_full_heap() {
        let mut heap: KLargestNeighboursHeap = KLargestNeighboursHeap::new(2 as usize);
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
        heap.push(neighbour);
        assert_eq!(2, heap.len());
        assert_eq!(0.3, heap.peek().unwrap().similarity);
    }
}
