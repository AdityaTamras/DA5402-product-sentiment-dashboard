# High-level design document

## Application overview
The E-Commerce Product Review Sentiment Dashboard is an aspect-based
sentiment classification system that ingests product reviews, classifies
sentiment per aspect (price, quality, delivery, service), and surfaces
results on a live operational dashboard.

## Architecture
The system follows a strict three-layer architecture:
- Data layer: Airflow DAG + DVC pipeline
- Serving layer: FastAPI backend + React frontend (loose coupling via REST)
- Monitoring layer: Prometheus + Grafana

## Design choices
- DistilBERT chosen over BERT for 40% smaller size
- TF-IDF + LR retained as fallback for zero-downtime serving
- SQLite chosen for simplicity in local deployment
- Docker Compose enforces frontend/backend separation
- DVC ensures full reproducibility via artifact hashing