import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client=TestClient(app)

def test_health_returns_ok():
    res=client.get('/health')
    assert res.status_code==200
    assert res.json()['status']=='ok'

def test_ready_returns_model_status():
    res=client.get('/ready')
    assert res.status_code==200
    assert 'model_loaded' in res.json()

def test_predict_positive_review():
    res=client.post('/predict', json={
        'review_text': 'Absolutely love this product, works perfectly!',
        'aspect': 'quality'
    })
    assert res.status_code == 200
    data=res.json()
    assert data['predicted'] in ['positive', 'neutral', 'negative']
    assert 0.0 <=data['confidence']<=1.0
    assert 'low_confidence' in data

def test_predict_negative_review():
    res=client.post('/predict', json={
        'review_text': 'Terrible product, broke after one day',
        'aspect': 'quality'
    })
    assert res.status_code==200

def test_predict_empty_text_rejected():
    res=client.post('/predict', json={
        'review_text': '     ',
        'aspect': 'general'
    })
    assert res.status_code==422

def test_predict_very_long_text():
    res=client.post('/predict', json={
        'review_text': 'great product ' * 200,
        'aspect': 'general'
    })
    assert res.status_code==200

def test_reviews_returns_list():
    res=client.get('/reviews?limit=10')
    assert res.status_code==200
    assert isinstance(res.json(), list)

def test_reviews_limit_respected():
    res=client.get('/reviews?limit=5')
    assert len(res.json())<=5

def test_low_confidence_flag_present():
    res=client.post('/predict', json={
        'review_text': 'ok i guess',
        'aspect': 'general'
    })
    assert isinstance(res.json()['low_confidence'], bool)

def test_aspect_echoed_in_response():
    res=client.post('/predict', json={
        'review_text': 'Fast shipping, arrived in two days!',
        'aspect': 'delivery'
    })
    assert res.json()['aspect'] == 'delivery'
