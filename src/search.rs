use serde::Serialize;
use uuid::Uuid;

use crate::similarity::CosineSimilarity;
use crate::types::{Embedding, EmbeddingEntry};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Serialize, Clone, Debug)]
pub struct Neighbour<'b> {
    pub uuid: &'b Uuid,
    similarity: f32,
}

impl<'b> Eq for Neighbour<'b> {}

impl<'b> PartialEq for Neighbour<'b> {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl<'b> PartialOrd for Neighbour<'b> {
    fn partial_cmp(&self, other: &Neighbour) -> Option<Ordering> {
        other.similarity.partial_cmp(&self.similarity)
    }
}

impl<'b> Ord for Neighbour<'b> {
    fn cmp(&self, other: &Neighbour) -> Ordering {
        self.cmp(other)
    }
}

#[derive(Clone, Copy)]
pub struct KNN<'a> {
    database: &'a HashMap<Uuid, EmbeddingEntry>,
    k: usize,
}

impl<'a> KNN<'a> {
    pub fn new(database: &'a HashMap<Uuid, EmbeddingEntry>, k: usize) -> Self {
        Self { database, k }
    }

    fn insert_neighbour(
        self,
        mut heap: BinaryHeap<Neighbour<'a>>,
        neighour: Neighbour<'a>,
    ) -> BinaryHeap<Neighbour<'a>> {
        if heap.len() < self.k {
            heap.push(neighour);
        } else if neighour.similarity > heap.peek().unwrap().similarity {
            heap.pop();
            heap.push(neighour);
        }
        heap
    }

    pub fn search(self, query_vector: &'a Embedding) -> Vec<Neighbour> {
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        for (uuid, vector) in self.database.iter() {
            let similarity = CosineSimilarity.calculate(&vector.values, query_vector);
            let neighbour = Neighbour { uuid, similarity };

            heap = self.insert_neighbour(heap, neighbour);
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
            uuid: &Uuid::new_v4(),
            similarity: 0.9,
        };
        let neighbour_b = Neighbour {
            uuid: &Uuid::new_v4(),
            similarity: 0.8,
        };
        assert!(neighbour_a < neighbour_b);
    }

    #[test]
    fn test_insert_neighbour_to_empty_heap() {
        let database: HashMap<Uuid, EmbeddingEntry> = HashMap::new();
        let search = KNN {
            database: &database,
            k: 2 as usize,
        };
        let heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let neighbour = Neighbour {
            uuid: &Uuid::new_v4(),
            similarity: 0.1,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(1, actual_heap.len());
        assert_eq!(0.1, actual_heap.peek().unwrap().similarity)
    }

    #[test]
    fn test_insert_neighbour_to_non_empty_heap() {
        let database: HashMap<Uuid, EmbeddingEntry> = HashMap::new();
        let search = KNN {
            database: &database,
            k: 2 as usize,
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let id = Uuid::new_v4();
        heap.push(Neighbour {
            uuid: &id,
            similarity: 0.1,
        });
        let neighbour = Neighbour {
            uuid: &Uuid::new_v4(),
            similarity: 0.2,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0.1, actual_heap.peek().unwrap().similarity);
    }

    #[test]
    fn test_insert_neighbour_into_full_heap() {
        let database: HashMap<Uuid, EmbeddingEntry> = HashMap::new();
        let search = KNN {
            database: &database,
            k: 2 as usize,
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let id = Uuid::new_v4();
        heap.push(Neighbour {
            uuid: &id,
            similarity: 0.5,
        });
        heap.push(Neighbour {
            uuid: &id,
            similarity: 0.2,
        });
        let neighbour = Neighbour {
            uuid: &Uuid::new_v4(),
            similarity: 0.6,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0.5, actual_heap.peek().unwrap().similarity);
    }

    #[test]
    fn test_insert_non_largest_neighbour_into_full_heap() {
        let database: HashMap<Uuid, EmbeddingEntry> = HashMap::new();
        let search = KNN {
            database: &database,
            k: 2 as usize,
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let id = Uuid::new_v4();
        heap.push(Neighbour {
            uuid: &id,
            similarity: 0.5,
        });
        heap.push(Neighbour {
            uuid: &id,
            similarity: 0.2,
        });
        let neighbour = Neighbour {
            uuid: &Uuid::new_v4(),
            similarity: 0.3,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0.3, actual_heap.peek().unwrap().similarity);
    }
}
