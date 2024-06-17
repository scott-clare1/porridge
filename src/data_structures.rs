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
