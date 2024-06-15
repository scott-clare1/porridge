# Porridge 🥣

A simple in-memory vector store written in Rust.

## Getting Started
---
Build the server with Docker:
```bash
docker build -t porridge .
```
Now run the container:
```bash
docker run -p 5000:5000 porridge
```

## Endpoints
---
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

## Clients
---
- Python: in the works.
