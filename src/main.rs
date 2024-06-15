mod search;
mod settings;
mod similarity;
mod types;

use crate::search::KNN;
use crate::settings::{Settings, DEFAULT_SIMILARITY_METRIC};
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

    let socket_addr = SocketAddr::V4(SocketAddrV4::new(
        Ipv4Addr::new(ipv4_addr[0], ipv4_addr[1], ipv4_addr[2], ipv4_addr[3]),
        port,
    ));

    println!(
        "API running on the following URL: http://{}:{}",
        config.host, config.port
    );

    let db = Database {
        contents: Arc::new(Mutex::new(HashMap::new())),
    };

    let similarity_metric = match config.similarity_metric.as_str() {
        "cosine" => MetricType::Cosine(CosineSimilarity),
        _ => {
            println!("Invalid similarity metric given falling back to default: {DEFAULT_SIMILARITY_METRIC}");
            MetricType::Cosine(CosineSimilarity)
        }
    };

    let search = Arc::new(KNN::new(db.clone(), 2 as usize, similarity_metric));

    let db_filter = warp::any().map(move || db.clone());

    let add_vector = warp::post()
        .and(path("vectors"))
        .and(warp::body::json())
        .and(db_filter.clone())
        .map(|new_entries: Vec<EmbeddingEntry>, db: Database| {
            let mut vectors = db.contents.lock().unwrap();
            let mut response_ids = vec![];
            for entry in new_entries.iter() {
                let id = Uuid::new_v4();
                vectors.insert(id, entry.clone());
                response_ids.push(id);
            }
            json(&response_ids)
        });

    let get_vector = warp::get()
        .and(path("vectors"))
        .and(path::param::<Uuid>())
        .and(db_filter.clone())
        .map(|id: Uuid, db: Database| {
            let vectors = db.contents.lock().unwrap();
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

    let knn_filter = warp::any().map(move || search.clone());

    let search_database = warp::post()
        .and(path("search"))
        .and(warp::body::json())
        .and(knn_filter.clone())
        .map(|query_vector: EmbeddingEntry, search: Arc<KNN>| {
            let nearest_neighbours = search.search(&query_vector.values);
            json(&nearest_neighbours)
        });

    let heartbeat = warp::get().and(path("heartbeat")).map(|| {
        with_status(
            json(&serde_json::json!({"status": "I am alive"})),
            StatusCode::OK,
        )
    });

    let routes = warp::get().and(add_vector.or(get_vector).or(search_database).or(heartbeat));

    warp::serve(routes).run(socket_addr).await;
}
