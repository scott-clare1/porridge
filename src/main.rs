mod data_structures;
mod nearest_neighbours;
mod search;
mod settings;
mod similarity;
mod types;

use crate::nearest_neighbours::KNNAlgortihm;
use crate::search::{BruteForce, SearchAlgorithm};
use crate::settings::{Settings, DEFAULT_SEARCH_ALGORITHM, DEFAULT_SIMILARITY_METRIC};
use crate::similarity::CosineSimilarity;
use crate::types::{Database, EmbeddingEntry};
use similarity::MetricType;
use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use warp::http::StatusCode;
use warp::path;
use warp::reply::{json, with_status};
use warp::Filter;

#[tokio::main]
async fn main() {
    let config = Settings::new();

    let ipv4_addr: [u8; 4] = match config.host.parse::<Ipv4Addr>() {
        Ok(ip) => ip.octets(),
        Err(_) => panic!("Invalid IP address format"),
    };

    let port = config.port.parse::<u16>().unwrap();
    let k_neighbours = config.k_neighbours.parse::<usize>().unwrap();

    let socket_addr = SocketAddr::V4(SocketAddrV4::new(
        Ipv4Addr::new(ipv4_addr[0], ipv4_addr[1], ipv4_addr[2], ipv4_addr[3]),
        port,
    ));

    println!(
        "API running on the following URL: http://{}:{}",
        config.host, config.port
    );

    let database: Database = Arc::new(Mutex::new(HashMap::new()));

    let similarity_metric = match config.similarity_metric.as_str() {
        "cosine" => MetricType::Cosine(CosineSimilarity),
        _ => {
            println!("Invalid similarity metric given falling back to default: {DEFAULT_SEARCH_ALGORITHM}");
            MetricType::Cosine(CosineSimilarity)
        }
    };

    let search_algorithm = match config.search_algorithm.as_str() {
        "brute" => {
            let algorithm = BruteForce::new(k_neighbours, similarity_metric);
            SearchAlgorithm::Brute(algorithm)
        }
        _ => {
            println!("Invalid search algorithm given falling back to default: {DEFAULT_SIMILARITY_METRIC}");
            let algorithm = BruteForce::new(k_neighbours, similarity_metric);
            SearchAlgorithm::Brute(algorithm)
        }
    };

    let search = Arc::new(KNNAlgortihm::new(database, search_algorithm));

    let knn_filter = warp::any().map(move || search.clone());

    let store = warp::post()
        .and(path("store"))
        .and(warp::body::json())
        .and(knn_filter.clone())
        .map(
            |new_entries: Vec<EmbeddingEntry>, search: Arc<KNNAlgortihm>| {
                let mut vectors = search.database.lock().unwrap();
                let mut response_ids = vec![];
                for entry in new_entries.iter() {
                    let id = Uuid::new_v4();
                    vectors.insert(id, entry.clone());
                    response_ids.push(id);
                }
                json(&response_ids)
            },
        );

    let retrieve = warp::get()
        .and(path("retrieve"))
        .and(path::param::<Uuid>())
        .and(knn_filter.clone())
        .map(|id: Uuid, search: Arc<KNNAlgortihm>| {
            let vectors = search.database.lock().unwrap();
            if let Some(vector) = vectors.get(&id) {
                with_status(json(vector), StatusCode::OK)
            } else {
                with_status(
                    json(
                        &"Vector not found - are you sure the requested ID exists in the database?",
                    ),
                    StatusCode::NOT_FOUND,
                )
            }
        });

    let search = warp::post()
        .and(path("search"))
        .and(warp::body::json())
        .and(knn_filter.clone())
        .map(|query_vector: EmbeddingEntry, search: Arc<KNNAlgortihm>| {
            if search.database.lock().unwrap().is_empty() {
                with_status(json(&"Vector store is empty - you need to upload documents with the /store endpoint."), StatusCode::NO_CONTENT)
            }
            else {
                let nearest_neighbours = search.search(&query_vector.embeddings);
                with_status(json(&nearest_neighbours), StatusCode::OK)
            }
        });

    let delete = warp::delete()
        .and(path("delete"))
        .and(path::param::<Uuid>())
        .and(knn_filter.clone())
        .map(|id: Uuid, search: Arc<KNNAlgortihm>| {
            let mut vectors = search.database.lock().unwrap();
            vectors.remove(&id);
            json(&format!("Removed entry: {}", id))
        });

    let delete_all = warp::delete()
        .and(path("delete"))
        .and(knn_filter.clone())
        .map(|search: Arc<KNNAlgortihm>| {
            let mut vectors = search.database.lock().unwrap();
            vectors.drain();
            json(&"Removed all entries from database.")
        });

    let heartbeat = warp::get().and(path("heartbeat")).map(|| {
        with_status(
            json(&serde_json::json!({"status": "I am alive"})),
            StatusCode::OK,
        )
    });

    let routes = store
        .or(retrieve)
        .or(search)
        .or(heartbeat)
        .or(delete)
        .or(delete_all);

    warp::serve(routes).run(socket_addr).await;
}
