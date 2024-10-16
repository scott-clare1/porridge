use crate::data_structures::KLargestNeighboursHeap;
use crate::nearest_neighbours::KNNInterface;
use crate::similarity::{MetricType, SimilarityMetric};
use crate::types::{Database, Embedding, Neighbour};

#[derive(Clone)]
pub enum SearchAlgorithm {
    Brute(BruteForce),
}

impl KNNInterface for SearchAlgorithm {
    fn search(&self, database: &Database, query_vector: &Embedding) -> Vec<Neighbour> {
        match self {
            SearchAlgorithm::Brute(method) => method.search(&database, &query_vector),
        }
    }
}

#[derive(Clone)]
pub struct BruteForce {
    k: usize,
    metric: MetricType,
}

impl BruteForce {
    pub fn new(k: usize, metric: MetricType) -> Self {
        Self { k, metric }
    }
}

impl KNNInterface for BruteForce {
    fn search(&self, database: &Database, query_vector: &Embedding) -> Vec<Neighbour> {
        let mut heap: KLargestNeighboursHeap = KLargestNeighboursHeap::new(self.k);
        let database = database.lock().unwrap();
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
    use uuid::Uuid;

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
