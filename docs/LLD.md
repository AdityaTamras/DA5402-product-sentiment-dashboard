# Low-level Design Document

## API endpoints

### POST /predict

Request:
{
  "review_text": "string (required)",
  "aspect": "string (optional, default: general)"
}

Response:
{
  "predicted": "positive | neutral | negative",
  "confidence": 0.95,
  "aspect": "quality",
  "low_confidence": false,
  "model_used": "distilbert | tfidf"
}

### GET /reviews

Query params: limit (int, default 100)
Response: Array of prediction records

### GET /health

Response: {"status": "ok", "timestamp": "ISO string"}

### GET /ready

Response: {"model_loaded": true, "model_type": "distilbert | tfidf"}