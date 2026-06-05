import pytest
import pandas as pd
import numpy as np
from model_engine import EpidemicPredictor
import os
from unittest.mock import MagicMock, patch

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Country': ['USA', 'India'],
        'Date': ['2023-01-01', '2023-01-01'],
        'retail_and_recreation_percent_change_from_baseline_Lag_14d': [0, -10],
        'transit_stations_percent_change_from_baseline_Lag_14d': [0, -20],
        'workplaces_percent_change_from_baseline_Lag_14d': [0, -5],
        'Vax_Lag_21d': [70, 60]
    })

def test_predictor_init():
    predictor = EpidemicPredictor(model_path="fake_model.json", dataset_path="fake_data.csv")
    assert predictor.model_path == "fake_model.json"
    assert predictor.dataset_path == "fake_data.csv"
    assert predictor.model is None

@patch('os.path.exists')
@patch('pandas.read_csv')
@patch('xgboost.XGBClassifier')
def test_load_resources_success(mock_xgb, mock_read_csv, mock_exists, sample_df):
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_df
    
    predictor = EpidemicPredictor()
    predictor.load_resources()
    
    assert predictor.df is not None
    assert predictor.model is not None
    mock_xgb.return_value.load_model.assert_called_once()

@patch('os.path.exists')
def test_load_resources_missing_file(mock_exists):
    mock_exists.return_value = False
    predictor = EpidemicPredictor()
    with pytest.raises(FileNotFoundError):
        predictor.load_resources()

def test_predict_hotspots_logic(sample_df):
    predictor = EpidemicPredictor()
    predictor.model = MagicMock()
    # Mock predict_proba to return 0.8 for the first and 0.2 for the second
    predictor.model.predict_proba.return_value = np.array([[0.2, 0.8], [0.8, 0.2]])
    predictor.df = sample_df
    
    probs = predictor.predict_hotspots(sample_df)
    
    assert len(probs) == 2
    assert probs[0] == 80.0
    assert probs[1] == 20.0
