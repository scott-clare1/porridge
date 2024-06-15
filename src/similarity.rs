use crate::types::Embedding;

trait Norm {
    fn compute_norm(&self, embedding: &Embedding) -> f32 {
        let sum: f32 = embedding.iter().map(|x| x * x).sum();
        sum.sqrt()
    }
}

trait DotProduct {
    fn compute_dot_product(&self, a: &Embedding, b: &Embedding) -> f32 {
        let mut dot_product: f32 = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            dot_product += a_val * b_val;
        }
        dot_product
    }
}

#[derive(Clone)]
pub enum MetricType {
    Cosine(CosineSimilarity),
}

impl SimilarityMetric for MetricType {
    fn similarity(&self, a: &Embedding, b: &Embedding) -> f32 {
        match self {
            MetricType::Cosine(metric) => metric.similarity(&a, &b),
        }
    }
}

pub trait SimilarityMetric {
    fn similarity(&self, a: &Embedding, b: &Embedding) -> f32;
}

#[derive(Clone)]
pub struct CosineSimilarity;

impl Norm for CosineSimilarity {}
impl DotProduct for CosineSimilarity {}

impl SimilarityMetric for CosineSimilarity {
    fn similarity(&self, v1: &Embedding, v2: &Embedding) -> f32 {
        let v1_norm = self.compute_norm(v1);
        let v2_norm = self.compute_norm(v2);
        let dot_product = self.compute_dot_product(v1, v2);
        dot_product / (v1_norm * v2_norm)
    }
}

#[cfg(test)]
mod similarity_tests {

    use super::*;

    #[test]
    fn test_calculate_norm() {
        let embedding = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(1.118034, CosineSimilarity.compute_norm(&embedding))
    }

    #[test]
    fn test_dot_product() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let b = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        assert_eq!(1.2, CosineSimilarity.compute_dot_product(&a, &b))
    }

    #[test]
    fn test_caclulate() {}
}
