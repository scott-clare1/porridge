use crate::types::Neighbour;
use std::collections::BinaryHeap;

pub struct KLargestNeighboursHeap {
    heap: BinaryHeap<Neighbour>,
    k: usize,
}

impl KLargestNeighboursHeap {
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            k,
        }
    }

    pub fn push(&mut self, neighbour: Neighbour) {
        if self.len() < self.k {
            self.heap.push(neighbour);
        } else if neighbour.similarity > self.peek().unwrap().similarity {
            self.pop();
            self.heap.push(neighbour);
        }
    }

    pub fn pop(&mut self) -> Option<Neighbour> {
        self.heap.pop()
    }

    pub fn peek(&self) -> Option<&Neighbour> {
        self.heap.peek()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn into_sorted_vec(self) -> Vec<Neighbour> {
        self.heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod test_k_largest_neighbours_heap {

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
