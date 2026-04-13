import os
import pickle
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.data.database import SessionLocal
from app.data.models import MLModelRegistry
from app.ml.pipeline.processor import DataProcessor
from app.ml.pipeline.models import (
    TreeModel, NNModel, CatBoostModel, XGBModel, LGBMModel, PolyElasticNetModel
)

PROGRESS_FILE = 'training_progress.json'

class ElectionTrainer:
    def __init__(self):
        self.processor = DataProcessor()
        self.models_dir = 'app/ml/models'
        os.makedirs(self.models_dir, exist_ok=True)

    def update_progress(self, progress, status="training"):
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({"progress": progress, "status": status}, f)

    def train_all(self, selected_models=None):
        """Trains selected models and registers them in the database."""
        self.update_progress(5, "Loading data...")
        train_df, test_df = self.processor.prepare_training_data()
        
        if train_df is None or test_df is None:
            self.update_progress(100, "Error: No data")
            return "No data available for training."

        features = ['RINKEJU_SKAICIUS', 'SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
        target = 'VOTE_SHARE'

        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        # Categorical features for CatBoost
        cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']

        # Pre-encoded features for other models
        self.update_progress(10, "Encoding features...")
        X_train_enc = self.processor.encode_features(X_train)
        X_test_enc = self.processor.encode_features(X_test)
        X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

        model_configs = {
            'rf': {'class': TreeModel, 'filename': 'rf_tuned.pkl'},
            'nn': {'class': NNModel, 'filename': 'nn_model.pt'},
            'catboost': {'class': CatBoostModel, 'filename': 'catboost_tuned.cbm'},
            'xgboost': {'class': XGBModel, 'filename': 'xgboost_tuned.pkl'},
            'lgbm': {'class': LGBMModel, 'filename': 'lgbm_tuned.pkl'},
            'elasticnet': {'class': PolyElasticNetModel, 'filename': 'elasticnet_tuned.pkl'}
        }

        results = []
        session = SessionLocal()
        
        enabled_models = [m for m in model_configs.keys() if not selected_models or m in selected_models]
        if not enabled_models:
            return "No models selected."

        for i, name in enumerate(enabled_models):
            progress = 10 + int((i / len(enabled_models)) * 80)
            self.update_progress(progress, f"Training {name}...")
            
            cfg = model_configs[name]
            try:
                model_inst = cfg['class']()
                
                if name == 'catboost':
                    model_inst.train(X_train, y_train, cat_features=cat_features)
                    preds = model_inst.predict(X_test)
                    model_inst.save_model(os.path.join(self.models_dir, cfg['filename']))
                else:
                    model_inst.train(X_train_enc, y_train)
                    preds = model_inst.predict(X_test_enc)
                    with open(os.path.join(self.models_dir, cfg['filename']), 'wb') as f:
                        pickle.dump(model_inst, f)

                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                # Register in DB
                new_model = MLModelRegistry(
                    model_name=f"{name.upper()}_v1",
                    model_type=name,
                    mae=float(mae),
                    rmse=float(rmse),
                    file_path=os.path.join(self.models_dir, cfg['filename']),
                    training_date=datetime.now()
                )
                session.add(new_model)
                results.append({'model': name, 'mae': mae, 'rmse': rmse})
                
            except Exception as e:
                print(f"Failed to train {name}: {str(e)}")

        session.commit()
        session.close()
        
        self.update_progress(100, "Completed")
        return results

def train_suite(models, timeout=60):
    """Wrapper function for threading."""
    trainer = ElectionTrainer()
    output = trainer.train_all(selected_models=models)
    return output
