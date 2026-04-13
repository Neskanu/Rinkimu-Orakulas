import os
import joblib
import json
import pandas as pd
import numpy as np
import time
import traceback
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        
        if "suite_history" not in data:
            data["suite_history"] = []
            
        data[model_name]["current_stage"] = stage
        if metrics:
            mae = metrics.get('mae', 0)
            data[model_name]["current_value"] = mae
            data[model_name]["metrics"] = metrics
            data[model_name]["history"].append({"value": mae, "time": time.time()})
            
            if stage == "Finished":
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
        
        ts = datetime.now().strftime('%H:%M:%S')
        met_str = ""
        if metrics:
            met_str = f" | MAE: {metrics.get('mae',0):.4f} | RMSE: {metrics.get('rmse',0):.4f} | R2: {metrics.get('r2',0):.4f}"
        msg_str = f" | {message}" if message else ""
        print(f"[{ts}] [{model_name.upper()}] {stage}{met_str}{msg_str}")

    def train_all(self, selected_models=None, model_params=None):
        """Trains selected models using the robust scikit-learn ColumnTransformer."""
        try:
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
            print(f"\n{'='*60}\n🚀 STARTING ELECTION MODEL TRAINING PIPELINE\n{'='*60}")
            
            self.update_progress("SYSTEM", "Initializing", message="Loading datasets...")
            train_df, test_df = self.processor.prepare_training_data()
            
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

            # 1. Sukuriame ir išsaugome produkcinį transformatorių
            self.update_progress("SYSTEM", "Encoding", message="Fitting ColumnTransformer...")
            print("[*] Preprocessing: Fitting ColumnTransformer...")
            preprocessor = self.processor.fit_and_save_preprocessor(X_train)
            
            # 2. Transformuojame duomenis
            X_train_enc = preprocessor.transform(X_train)
            X_test_enc = preprocessor.transform(X_test)
            print(f"[*] Preprocessing complete. Encoded shape: {X_train_enc.shape}")

            model_configs = {
                'rf': {'class': TreeModel, 'filename': 'rf_tuned.pkl', 'label': 'Random Forest'},
                'nn': {'class': NNModel, 'filename': 'nn_model.pt', 'label': 'Neural Network'},
                'catboost': {'class': CatBoostModel, 'filename': 'catboost_tuned.cbm', 'label': 'CatBoost'},
                'xgboost': {'class': XGBModel, 'filename': 'xgboost_tuned.pkl', 'label': 'XGBoost'},
                'lgbm': {'class': LGBMModel, 'filename': 'lgbm_tuned.pkl', 'label': 'LightGBM'},
                'elasticnet': {'class': ElasticNetModel, 'filename': 'elasticnet_tuned.pkl', 'label': 'ElasticNet'}
            }

            results = []
            session = SessionLocal()
            
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
                        # CatBoost requires raw string data, not the encoded matrix
                        X_train_cb = X_train.copy()
                        X_train_cb[cat_features] = X_train_cb[cat_features].astype(str)
                        X_test_cb = X_test.copy()
                        X_test_cb[cat_features] = X_test_cb[cat_features].astype(str)
                        
                        model_inst.train(X_train_cb, y_train, cat_features=cat_features)
                        preds = model_inst.predict(X_test_cb)
                        
                        self.update_progress(name, "Saving", message=f"Saving {cfg['label']} artifact...")
                        model_inst.save_model(os.path.join(self.models_dir, cfg['filename']))
                    else:
                        # Visi kiti modeliai naudoja transformatoriaus paruoštą Numpy masyvą (X_train_enc)
                        model_inst.train(X_train_enc, y_train)
                        preds = model_inst.predict(X_test_enc)
                        
                        self.update_progress(name, "Saving", message=f"Saving {cfg['label']} artifact...")
                        joblib.dump(model_inst, os.path.join(self.models_dir, cfg['filename']))

                    self.update_progress(name, "Evaluation", message="Computing accuracy metrics...")
                    mae = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    r2 = r2_score(y_test, preds)
                    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

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
                    session.commit()
                    
                    results.append({'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2})
                    self.update_progress(name, "Finished", metrics=metrics)

                except Exception as e:
                    print(f"[!] Error training {name}: {str(e)}")
                    traceback.print_exc()
                    self.update_progress(name, "Failed", message=f"ERROR: {str(e)[:100]}...")
                    continue 

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