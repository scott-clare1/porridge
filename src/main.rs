use porddige::types::{Database, Vector};
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
        vectors: Arc::new(Mutex::new(HashMap::new())),
    };

    let db_filter = warp::any().map(move || db.clone());

    let add_vector = warp::post()
        .and(path("vectors"))
        .and(warp::body::json())
        .and(db_filter.clone())
        .map(|new_vector: Vector, db: Database| {
            let mut vectors = db.vectors.lock().unwrap();
            let id = new_vector.id;
            vectors.insert(id, new_vector);
            json(&id)
        });

    let get_vector = warp::get()
        .and(path("vectors"))
        .and(path::param::<Uuid>())
        .and(db_filter.clone())
        .map(|id: Uuid, db: Database| {
            let vectors = db.vectors.lock().unwrap();
            if let Some(vector) = vectors.get(&id) {
                with_status(json(vector), StatusCode::OK)
            } else {
                with_status(json(&"Vector not found"), StatusCode::NOT_FOUND)
            }
        });

    let routes = add_vector.or(get_vector);

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
