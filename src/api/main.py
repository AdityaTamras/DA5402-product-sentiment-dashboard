import logging, time, json, pickle, os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Histogram
import mlflow.pyfunc
import subprocess

BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINE_FILE=os.path.join(BASE_DIR, 'data', 'baseline', 'baseline_stats.json')
tfidf_model=os.path.join(BASE_DIR, 'models', 'tfidf_lr_model.pkl')
l_encoder=os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
MLFLOW_URI='http://localhost:5000'

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
logger=logging.getLogger(__name__)

app=FastAPI(title='Sentiment Dashboard API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

Instrumentator().instrument(app).expose(app)

DRIFT_GAUGE=Gauge('review_drift_score', 'KL divergence of text length vs training baseline')
PREDICTION_COUNTER=Counter('predictions_total', 'Total predictions made',['label', 'aspect'])
CONFIDENCE_HISTOGRAM=Histogram('prediction_confidence', 'Distribution of confidence scores', buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
LOW_CONFIDENCE_COUNTER=Counter('low_confidence_predictions_total', 'Predictions with confidence<0.6')
LATENCY_HISTOGRAM=Histogram('predict_latency_seconds', 'Prediction endpoint latency')

engine=create_engine('sqlite:///predictions.db', connect_args={'check_same_thread': False})
SessionLocal=sessionmaker(bind=engine)
Base=declarative_base()

class Prediction(Base):
    __tablename__='predictions'
    id=Column(Integer, primary_key=True, index=True)
    review_text=Column(String)
    predicted=Column(String)
    confidence=Column(Float)
    aspect=Column(String)
    ground_truth=Column(String, nullable=True)
    timestamp=Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

with open(BASELINE_FILE) as f:
    BASELINE=json.load(f)

MODEL=None
LABEL_ENCODER=None

@app.on_event('startup')
def load_model():
    global MODEL, LABEL_ENCODER
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        MODEL=mlflow.pyfunc.load_model('models:/sentiment-classifier@champion')
        logger.info('Loaded Production model from MLflow registry')
    except Exception as e:
        logger.warning(f'MLflow load failed: {e} — falling back to pickle')
        with open(tfidf_model,'rb') as f:
            MODEL=pickle.load(f)
    with open(l_encoder,'rb') as f:
        LABEL_ENCODER=pickle.load(f)

class ReviewIn(BaseModel):
    review_text: str
    aspect: str = 'general'

class PredictionOut(BaseModel):
    predicted: str
    confidence: float
    aspect: str
    low_confidence: bool

@app.get('/health')
def health(): return {'status': 'ok'}

@app.get('/ready')
def ready(): return {'model_loaded': MODEL is not None}

@app.post('/predict', response_model=PredictionOut)
def predict(review: ReviewIn):
    if MODEL is None:
        raise HTTPException(503, 'Model not loaded')
    try:
        start=time.time()
        import pandas as pd
        df_in=pd.DataFrame({'text': [review.review_text]})

        if hasattr(MODEL, 'predict_proba'):
            proba=MODEL.predict_proba(df_in['text'])[0]
            pred_idx=proba.argmax()
            confidence=float(proba[pred_idx])
            predicted=LABEL_ENCODER.inverse_transform([pred_idx])[0]
        else:
            result=MODEL.predict(df_in)
            if isinstance(result, list):
                raw_label=result[0]['label']
                confidence=float(result[0]['score'])
            else:
                raw_label=result.iloc[0]['label']
                confidence=float(result.iloc[0]['score'])
            if raw_label.startswith("LABEL_"):
                pred_idx=int(raw_label.split("_")[-1])
                predicted=LABEL_ENCODER.inverse_transform([pred_idx])[0]
            else:
                predicted=raw_label

        latency=time.time()-start
        logger.info(json.dumps({'event': 'predict', 'label': predicted, 'confidence': round(confidence, 3), 'latency_ms': round(latency*1000, 1)}))

        from scipy.special import rel_entr
        import numpy as np
        text_len=len(review.review_text)
        baseline_mean=BASELINE['text_length_mean']
        drift=abs(text_len-baseline_mean)/baseline_mean
        DRIFT_GAUGE.set(round(drift, 4))
        PREDICTION_COUNTER.labels(label=predicted, aspect=review.aspect).inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)
        LATENCY_HISTOGRAM.observe(latency)
        if confidence<0.6:
            LOW_CONFIDENCE_COUNTER.inc()

        db = SessionLocal()
        db.add(Prediction(review_text=review.review_text[:500], predicted=predicted, confidence=confidence, aspect=review.aspect))
        db.commit(); db.close()

        return PredictionOut(predicted=predicted, confidence=round(confidence,3), aspect=review.aspect, low_confidence=confidence < 0.6)
    
    except Exception as e:
        logger.error(json.dumps({'event': 'predict_error', 'error': str(e)}))
        raise HTTPException(500, str(e))

@app.get('/pipeline-status')
def pipeline_status():
    try:
        result=subprocess.run(
            ["dvc", "dag"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"status": result.stdout}
    except Exception as e:
        return {"status": f"Error: {str(e)}"}

@app.get('/reviews')
def get_reviews(limit: int=100):
    db=SessionLocal()
    rows=db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
    db.close()
    return [{'id':r.id,'text':r.review_text,'predicted':r.predicted, 'confidence':r.confidence,'aspect':r.aspect, 'timestamp':str(r.timestamp)} for r in rows]