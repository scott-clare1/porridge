use porddige::search::KNN;
use porddige::types::{Database, EmbeddingEntry};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use warp::http::StatusCode;
use warp::path;
use warp::reply::{json, with_status};
use warp::Filter;

#[tokio::main]
async fn main() {
    let db = Database {
        contents: Arc::new(Mutex::new(HashMap::new())),
    };

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

    let search_database = warp::post()
        .and(path("search"))
        .and(warp::body::json())
        .and(db_filter.clone())
        .map(|query_vector: EmbeddingEntry, db: Database| {
            let vectors = db.contents.lock().unwrap();
            let search = KNN::new(&vectors, 2 as usize);
            let nearest_neighbours = search.search(&query_vector.values);
            json(&nearest_neighbours)
        });

    let routes = add_vector.or(get_vector).or(search_database);

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
