use crate::similarity::CosineSimilarity;
use crate::types::Embedding;
use crate::types::{Database, Vector};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Debug)]
pub struct Neighbour {
    pub index: usize,
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

pub struct KNN<'a> {
    database: &'a Database,
    k: usize,
}

impl<'a> KNN<'a> {
    pub fn new(vectors: &'a [Embedding], k: usize) -> Self {
        Self { vectors, k }
    }

    fn insert_neighbour(
        &mut self,
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

    pub fn search(&mut self, query_vector: &Embedding) -> Vec<Neighbour> {
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        for (index, vector) in self.vectors.iter().enumerate() {
            let similarity = CosineSimilarity.calculate(vector, query_vector);
            let neighbour = Neighbour { index, similarity };

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
            index: 0,
            similarity: 0.9,
        };
        let neighbour_b = Neighbour {
            index: 1,
            similarity: 0.8,
        };
        assert!(neighbour_a < neighbour_b);
    }

    #[test]
    fn test_insert_neighbour_to_empty_heap() {
        let embeddings = [vec![0.2, 0.2, 0.2]];
        let mut search = KNN {
            vectors: &embeddings,
            k: 2 as usize,
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        let neighbour = Neighbour {
            index: 0,
            similarity: 0.1,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(1, actual_heap.len());
        assert_eq!(0, actual_heap.peek().unwrap().index)
    }

    #[test]
    fn test_insert_neighbour_to_non_empty_heap() {
        let embeddings = [vec![0.2, 0.2, 0.2]];
        let mut search = KNN {
            vectors: &embeddings,
            k: 2 as usize,
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        heap.push(Neighbour {
            index: 0,
            similarity: 0.1,
        });
        let neighbour = Neighbour {
            index: 1,
            similarity: 0.2,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0, actual_heap.peek().unwrap().index);
    }

    #[test]
    fn test_insert_neighbour_into_full_heap() {
        let embeddings = [vec![0.2, 0.2, 0.2]];
        let mut search = KNN {
            vectors: &embeddings,
            k: 2 as usize,
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        heap.push(Neighbour {
            index: 0,
            similarity: 0.5,
        });
        heap.push(Neighbour {
            index: 1,
            similarity: 0.2,
        });
        let neighbour = Neighbour {
            index: 2,
            similarity: 0.6,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(0, actual_heap.peek().unwrap().index);
    }

    #[test]
    fn test_insert_non_largest_neighbour_into_full_heap() {
        let embeddings = [vec![0.2, 0.2, 0.2]];
        let mut search = KNN {
            vectors: &embeddings,
            k: 2 as usize,
        };
        let mut heap: BinaryHeap<Neighbour> = BinaryHeap::new();
        heap.push(Neighbour {
            index: 0,
            similarity: 0.5,
        });
        heap.push(Neighbour {
            index: 1,
            similarity: 0.2,
        });
        let neighbour = Neighbour {
            index: 2,
            similarity: 0.3,
        };
        let actual_heap = search.insert_neighbour(heap, neighbour);
        assert_eq!(2, actual_heap.len());
        assert_eq!(2, actual_heap.peek().unwrap().index);
    }
}
