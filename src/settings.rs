use std::env;

const DEFAULT_PORT: &str = "5000";
const DEFAULT_HOST: &str = "0.0.0.0";
pub const DEFAULT_SIMILARITY_METRIC: &str = "cosine";
const DEFAULT_K_NEIGHBOURS: &str = "5";
pub const DEFAULT_SEARCH_ALGORITHM: &str = "brute";

pub struct Settings {
    pub host: String,
    pub port: String,
    pub similarity_metric: String,
    pub k_neighbours: String,
    pub search_algorithm: String,
}

impl Settings {
    pub fn new() -> Self {
        let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
        let port = env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());
        let similarity_metric =
            env::var("SIMILARITY_METRIC").unwrap_or_else(|_| DEFAULT_SIMILARITY_METRIC.to_string());
        let k_neighbours =
            env::var("K_NEIGHBOURS").unwrap_or_else(|_| DEFAULT_K_NEIGHBOURS.to_string());
        let search_algorithm =
            env::var("SEARCH_ALGORITHM").unwrap_or_else(|_| DEFAULT_SEARCH_ALGORITHM.to_owned());

        Self {
            host,
            port,
            similarity_metric,
            k_neighbours,
            search_algorithm,
        }
    }
}
