import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
import joblib

class BaseModel:
    def train(self, X, y, **kwargs):
        pass
    def predict(self, X):
        pass

class TreeModel(BaseModel):
    def __init__(self, **params):
        params.setdefault('n_estimators', 100)
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', 2)
        self.model = RandomForestRegressor(**params)
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class NNModel(BaseModel):
    def __init__(self, **params):
        params.setdefault('hidden_layer_sizes', (64, 32))
        params.setdefault('max_iter', 500)
        params.setdefault('random_state', 42)
        self.model = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('mlp', MLPRegressor(**params))
        ])
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class CatBoostModel(BaseModel):
    def __init__(self, **params):
        params.setdefault('iterations', 500)
        params.setdefault('learning_rate', 0.1)
        params.setdefault('depth', 6)
        params.setdefault('random_seed', 42)
        params.setdefault('verbose', False)
        params.setdefault('allow_writing_files', False)
        self.model = CatBoostRegressor(**params)
        self.cat_features = None

    def train(self, X, y, cat_features=None, **kwargs):
        self.cat_features = cat_features
        self.model.fit(X, y, cat_features=cat_features)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, path):
        self.model.save_model(path)
    
    def load_model(self, path):
        self.model.load_model(path)

class XGBModel(BaseModel):
    def __init__(self, **params):
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        params.setdefault('n_estimators', 100)
        params.setdefault('learning_rate', 0.1)
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', 2)
        self.model = XGBRegressor(**params)
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class LGBMModel(BaseModel):
    def __init__(self, **params):
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")
        params.setdefault('n_estimators', 100)
        params.setdefault('learning_rate', 0.1)
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', 2)
        params.setdefault('verbose', -1)
        self.model = LGBMRegressor(**params)
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class ElasticNetModel(BaseModel):
    def __init__(self, **params):
        params.setdefault('alpha', 1.0)
        params.setdefault('l1_ratio', 0.5)
        params.setdefault('random_state', 42)
        self.model = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('en', ElasticNet(**params))
        ])
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class PolyElasticNetModel(BaseModel):
    def __init__(self, degree=2, **params):
        params.setdefault('random_state', 42)
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler(with_mean=False)),
            ('en', ElasticNet(**params))
        ])
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class EnsembleModel:
    def __init__(self, models=None, weights=None):
        """
        models: dict of name -> model_instance
        weights: dict of name -> weight (summing to 1 or handled by normalization)
        """
        self.models = models or {}
        self.weights = weights or {
            'rf': 0.20,
            'nn': 0.20,
            'catboost': 0.20,
            'xgboost': 0.15,
            'lgbm': 0.15,
            'elasticnet': 0.10
        }

    def predict(self, X):
        if not self.models:
            return np.zeros(len(X))
        
        # Load the feature encoder for non-CatBoost models
        import os
        import pickle
        from scipy.sparse import hstack
        
        encoder_path = 'app/ml/models/feature_encoder.pkl'
        encoder = None
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
        
        # Prepare both data formats
        cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
        num_features = ['RINKEJU_SKAICIUS']
        
        X_enc = None
        if encoder:
            # Explicitly force types before transform to prevent alignment errors
            X_prep = X[cat_features].astype(str)
            X_cat = encoder.transform(X_prep)
            X_num = pd.to_numeric(X[num_features].iloc[:, 0], errors='coerce').fillna(0).values.reshape(-1, 1)
            X_enc = hstack([X_num, X_cat])
        
        # Ensure raw X for CatBoost also has aligned types/columns
        X_raw = X[cat_features + num_features].copy()
        for col in num_features:
            X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce').fillna(0)
        
        # Normalize weights using lowercase keys for matching
        total_w = sum(self.weights.values())
        norm_weights = {k.lower(): v/total_w for k, v in self.weights.items()}
        
        final_preds = np.zeros(len(X))
        # Match available models to weights (case-insensitive)
        available_models = [m for m in self.models.keys() if m.lower() in norm_weights]
        
        if not available_models: return np.zeros(len(X))

        actual_total_w = sum(norm_weights[m.lower()] for m in available_models)
        redist_weights = {m: norm_weights[m.lower()]/actual_total_w if actual_total_w > 0 else 1.0/len(available_models) for m in available_models}

        for name in available_models:
            # FORCE column alignment and types by position: [0:SARASO, 1:APYGARDA, 2:RINKEJU]
            is_catboost = 'catboost' in name.lower()
            if is_catboost:
                # NUCLEAR OPTION: Reconstruct fresh DataFrame to force Dtypes and alignment
                # Column 0: List Name (str), Column 1: District Name (str), Column 2: Voter Count (float)
                data_to_use = pd.DataFrame({
                    'SARASO_PAVADINIMAS': X_raw.iloc[:, 0].astype(str).values,
                    'APYGARDOS_PAVADINIMAS': X_raw.iloc[:, 1].astype(str).values,
                    'RINKEJU_SKAICIUS': pd.to_numeric(X_raw.iloc[:, 2], errors='coerce').fillna(0).astype(float).values
                })
            else:
                data_to_use = X_enc

            if data_to_use is None: data_to_use = X
                
            preds = self.models[name].predict(data_to_use)
            preds = np.clip(preds, 0, 100)
            final_preds += preds * redist_weights[name]
        
        return np.clip(final_preds, 0, 100)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
