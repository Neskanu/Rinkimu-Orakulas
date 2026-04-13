import os
import pickle
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from app.data.database import SessionLocal
from app.data.models import MLModelRegistry
from app.ml.pipeline.processor import DataProcessor
from app.ml.pipeline.models import (
    TreeModel, NNModel, CatBoostModel, XGBModel, LGBMModel, ElasticNetModel
)

PROGRESS_FILE = 'training_progress.json'

class ElectionTrainer:
    def __init__(self):
        self.processor = DataProcessor()
        self.models_dir = 'app/ml/models'
        os.makedirs(self.models_dir, exist_ok=True)

    def update_progress(self, model_name, stage, metrics=None, message=None):
        """Updates structured progress for both UI and Console."""
        data = {}
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    data = json.load(f)
            except:
                data = {}
        
        if model_name not in data:
            data[model_name] = {"history": [], "current_stage": "", "metrics": {}, "message": ""}
        
        # Initialize suite_history if missing
        if "suite_history" not in data:
            data["suite_history"] = []
            
        data[model_name]["current_stage"] = stage
        if metrics:
            mae = metrics.get('mae', 0)
            data[model_name]["current_value"] = mae
            data[model_name]["metrics"] = metrics
            
            # Per-model history (for convergence)
            data[model_name]["history"].append({"value": mae, "time": time.time()})
            
            # Suite-wide history (if it's a final metric for a model)
            if stage == "Finished":
                # Check if this model is already in suite_history, if so update it
                found = False
                for item in data["suite_history"]:
                    if item["name"] == model_name:
                        item["value"] = mae
                        found = True
                        break
                if not found:
                    data["suite_history"].append({"name": model_name, "value": mae})
            
        if message:
            data[model_name]["message"] = message
             
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f)
        
        # Console output
        ts = datetime.now().strftime('%H:%M:%S')
        met_str = ""
        if metrics:
            met_str = f" | MAE: {metrics.get('mae',0):.4f} | RMSE: {metrics.get('rmse',0):.4f} | R2: {metrics.get('r2',0):.4f}"
        msg_str = f" | {message}" if message else ""
        print(f"[{ts}] [{model_name.upper()}] {stage}{met_str}{msg_str}")

    def train_all(self, selected_models=None, model_params=None):
        """Trains selected models with optional custom parameters."""
        try:
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
            print(f"\n{'='*60}\n🚀 STARTING ELECTION MODEL TRAINING PIPELINE\n{'='*60}")
            
            self.update_progress("SYSTEM", "Initializing", message="Loading datasets...")
            train_df, test_df = self.processor.prepare_training_data()
            
            # Release DB session before heavy compute to prevent deadlocks
            self.processor.close()
            
            if train_df is None or test_df is None:
                self.update_progress("SYSTEM", "Error", message="No training data found in database.")
                return "No data available for training."

            print(f"[*] Data Loaded: {len(train_df)} training rows, {len(test_df)} test rows.")
            cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
            num_features = ['RINKEJU_SKAICIUS']
            target = 'VOTE_SHARE'

            X_train, y_train = train_df[cat_features + num_features], train_df[target]
            X_test, y_test = test_df[cat_features + num_features], test_df[target]

            # Optimization: Fit OneHotEncoder once for models that need it (not CatBoost)
            self.update_progress("SYSTEM", "Encoding", message="Fitting OneHotEncoder on categorical features...")
            print("[*] Preprocessing: Fitting OneHotEncoder...")
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            X_train_cat = encoder.fit_transform(X_train[cat_features])
            X_test_cat = encoder.transform(X_test[cat_features])
            print(f"[*] Preprocessing: Done. Sparse shape: {X_train_cat.shape}")
            
            # Save the encoder for inference/backtest
            encoder_path = os.path.join(self.models_dir, 'feature_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder, f)
            print(f"[*] Feature encoder saved to {encoder_path}")
            
            # Create Sparse Matrix for non-CatBoost models
            X_train_enc = hstack([X_train[num_features].values, X_train_cat])
            X_test_enc = hstack([X_test[num_features].values, X_test_cat])
            print(f"[*] Preprocessing complete. Sparse feature count: {X_train_enc.shape[1]}")
            print(f"[*] Training Features: {X_train.columns.tolist() if hasattr(X_train, 'columns') else 'N/A'}")

            model_configs = {
                'rf': {'class': TreeModel, 'filename': 'rf_tuned.pkl', 'label': 'Random Forest'},
                'nn': {'class': NNModel, 'filename': 'nn_model.pt', 'label': 'Neural Network'},
                'catboost': {'class': CatBoostModel, 'filename': 'catboost_tuned.cbm', 'label': 'CatBoost'},
                'xgboost': {'class': XGBModel, 'filename': 'xgboost_tuned.pkl', 'label': 'XGBoost'},
                'lgbm': {'class': LGBMModel, 'filename': 'lgbm_tuned.pkl', 'label': 'LightGBM'},
                'elasticnet': {'class': ElasticNetModel, 'filename': 'elasticnet_tuned.pkl', 'label': 'ElasticNet'}
            }

            results = []
            suite_model_registry = {}
            session = SessionLocal()
            print(f"[*] DATA DTYPES: {train_df.dtypes.to_dict()}")
            
            enabled_models = [m for m in model_configs.keys() if not selected_models or m in selected_models]
            print(f"[*] Selected models: {', '.join([m.upper() for m in enabled_models])}")

            for name in enabled_models:
                cfg = model_configs[name]
                params = model_params.get(name, {}) if model_params else {}
                
                self.update_progress(name, "Started", message=f"Initializing {cfg['label']}...")
                
                try:
                    model_inst = cfg['class'](**params)
                    
                    self.update_progress(name, "Training", message="Fitting data...")
                    if name == 'catboost':
                        # Use RAW categorical data for CatBoost - Force String types as final safety
                        X_train_cb = X_train.copy()
                        X_train_cb[cat_features] = X_train_cb[cat_features].astype(str)
                        X_test_cb = X_test.copy()
                        X_test_cb[cat_features] = X_test_cb[cat_features].astype(str)
                        
                        model_inst.train(X_train_cb, y_train, cat_features=cat_features)
                        preds = model_inst.predict(X_test_cb)
                        self.update_progress(name, "Saving", message=f"Saving {cfg['label']} artifact...")
                        model_inst.save_model(os.path.join(self.models_dir, cfg['filename']))
                    else:
                        # Use Scaled/Encoded sparse data for others
                        model_inst.train(X_train_enc, y_train)
                        preds = model_inst.predict(X_test_enc)
                        self.update_progress(name, "Saving", message=f"Saving {cfg['label']} artifact...")
                        import joblib
                        joblib.dump(model_inst, os.path.join(self.models_dir, cfg['filename']))

                    self.update_progress(name, "Evaluation", message="Computing accuracy metrics...")
                    mae = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    r2 = r2_score(y_test, preds)
                    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

                    # Register in DB
                    self.update_progress(name, "Registering", metrics=metrics, message=f"Success! R2 Score: {r2:.4f}")
                    new_model = MLModelRegistry(
                        model_name=f"{name.upper()}_v1",
                        model_type=name,
                        mae=float(mae),
                        rmse=float(rmse),
                        file_path=os.path.join(self.models_dir, cfg['filename']),
                        training_date=datetime.now()
                    )
                    session.add(new_model)
                    session.commit() # Save progress model-by-model
                    
                    suite_model_registry[name] = model_inst
                    results.append({'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2})
                    self.update_progress(name, "Finished", metrics=metrics)

                except Exception as e:
                    import traceback
                    print(f"[!] Error training {name}: {str(e)}")
                    traceback.print_exc()
                    self.update_progress(name, "Failed", message=f"ERROR: {str(e)[:100]}...")
                    continue # Keep moving through the suite

            session.commit()
            session.close()
            
            print(f"\n{'='*60}\n✅ ALL TRAINING TASKS COMPLETED\n{'='*60}")
            return results
        except Exception as e:
            err_msg = traceback.format_exc()
            self.update_progress("SYSTEM", "CRITICAL ERROR", message=str(e))
            print(f"\n{'!'*60}\n🔥 PIPELINE CRASHED\n{err_msg}\n{'!'*60}")
            return str(e)

def train_suite(models, params=None, timeout=60):
    """Wrapper function for threading."""
    try:
        trainer = ElectionTrainer()
        output = trainer.train_all(selected_models=models, model_params=params)
        return output
    except Exception as e:
        print(f"Thread Failure: {str(e)}")
        return str(e)
        return str(e)
