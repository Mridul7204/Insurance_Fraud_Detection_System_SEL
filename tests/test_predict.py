import os
import sys
import pytest
import pickle
import json

# make sure the project root is on the path so pytest can import app.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import app


def test_model_files_exist():
    required_files = ['svm_model.pkl', 'scaler.pkl', 'model_columns.pkl']
    for filename in required_files:
        assert os.path.exists(filename), f"{filename} should exist (run train_model.py)"


def test_home_page():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'Insurance Fraud Detection' in response.data


def test_health_endpoint():
    client = app.test_client()
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'


@pytest.mark.skipif(not os.path.exists('svm_model.pkl'), reason="Model not trained")
def test_prediction_route():
    client = app.test_client()
    response = client.post('/predict', data={
        'months_as_customer': '12',
        'policy_deductable': '500',
        'total_claim_amount': '1000',
        'umbrella_limit': '0',
        'number_of_vehicles_involved': '1',
        'incident_severity': 'Minor Damage'
    })
    assert response.status_code == 200
    assert b'prediction_text' in response.data or b'Claim appears' in response.data


@pytest.mark.skipif(not os.path.exists('svm_model.pkl'), reason="Model not trained")
def test_api_prediction_route():
    client = app.test_client()
    response = client.post('/api/predict',
                          data=json.dumps({
                              'months_as_customer': 12,
                              'policy_deductable': 500,
                              'total_claim_amount': 1000,
                              'umbrella_limit': 0,
                              'number_of_vehicles_involved': 1,
                              'incident_severity': 'Minor Damage'
                          }),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'is_fraud' in data
    assert isinstance(data['is_fraud'], bool)


def test_api_prediction_invalid_data():
    client = app.test_client()
    response = client.post('/api/predict',
                          data=json.dumps({}),
                          content_type='application/json')
    assert response.status_code == 400


def test_api_prediction_missing_data():
    client = app.test_client()
    response = client.post('/api/predict')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'application/json' in data['error']
