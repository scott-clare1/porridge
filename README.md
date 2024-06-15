# Porridge 🥣

A simple in-memory vector store written in Rust, performing K-Nearest Neighbour search.

## Getting Started
Build the server with Docker:
```bash
docker build -t porridge .
```
Now run the container:
```bash
docker run -p 5000:5000 porridge
```

## Endpoints
- `/store` - Post embeddings with the following schema:
```json
[
  {
    "embedding": [
      0.1,
      0.1,
      0.1
    ],
    "text": "Example text."
  }
]
```
- `/search` - Post a query payload and return `k` most similar embeddings. The payload should take the form of a single embedding entry, e.g.
```json
{
  "embedding": [
    0.1,
    0.1,
    0.1
  ],
  "text": "Example text."
}
```
> Search returns a vector of UUID associated with the most similar entries.

- `/retrieve` - Entries to the database can be retrieved by UUID as parameter to the endpoint.
- `/heartbeat` - If the server is alive then hitting this endpoint will return:
```json
{
  "status": "I am alive."
}
```

## Configuration
Settings for the server can be configured with environment variables:
- `HOST`: The host for the server, e.g., `0.0.0.0`.
- `PORT`: The port for the server, e.g., `5000`.
- `SIMILARITY_METRIC`: The similarity metric used for search (currently only cosine similarity supported). Options: [`cosine`].
- `K_NEAREST_NEIGHBOURS`: The number of nearest neighbours to return, e.g., `5`.

## Clients
- Python: in the works.

## Roadmap:
- Persistent storage options.
- Approximate nearest neighbours.
- More options for similarity metrics.
- Python client.
- Integration tests.
