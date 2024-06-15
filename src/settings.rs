use std::env;

const DEFAULT_PORT: &str = "5000";
const DEFAULT_HOST: &str = "0.0.0.0";
pub const DEFAULT_SIMILARITY_METRIC: &str = "cosine";

pub struct Settings {
    pub host: String,
    pub port: String,
    pub similarity_metric: String,
}

fn get_environment_variables() -> (String, String, String) {
    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());
    let similarity_metric =
        env::var("SIMILARITY_METRIC").unwrap_or_else(|_| DEFAULT_SIMILARITY_METRIC.to_string());
    (host, port, similarity_metric)
}

impl Settings {
    pub fn new() -> Self {
        let (host, port, similarity_metric) = get_environment_variables();

        Self {
            host,
            port,
            similarity_metric,
        }
    }
}
