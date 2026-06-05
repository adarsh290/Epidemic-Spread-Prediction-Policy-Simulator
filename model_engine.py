import pandas as pd
import xgboost as xgb
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EpidemicPredictor:
    def __init__(self, model_path="xgb_hotspot_model.json", dataset_path="ml_ready_dataset.csv"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.df = None
        self.features = [
            'retail_and_recreation_percent_change_from_baseline_Lag_14d',
            'transit_stations_percent_change_from_baseline_Lag_14d',
            'workplaces_percent_change_from_baseline_Lag_14d',
            'Vax_Lag_21d'
        ]

    def load_resources(self):
        """Loads the XGBoost model and the dataset."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            self.df = pd.read_csv(self.dataset_path)
            
            # Basic validation of dataset
            required_cols = ['Country', 'Date'] + self.features
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Dataset missing required columns: {missing_cols}")
            
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            logger.info("Model and dataset loaded successfully.")
            return self.model, self.df
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")
            raise

    def get_latest_data(self):
        """Returns the most recent snapshot of data for all countries."""
        if self.df is None:
            self.load_resources()
        latest_date = self.df['Date'].max()
        return self.df[self.df['Date'] == latest_date].copy()

    def predict_hotspots(self, simulated_data):
        """Generates hotspot probabilities for the provided data."""
        if self.model is None:
            self.load_resources()
        
        try:
            # Generate Predictions
            probabilities = self.model.predict_proba(simulated_data[self.features])[:, 1]
            return probabilities * 100
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
