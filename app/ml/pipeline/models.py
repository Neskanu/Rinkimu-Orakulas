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
        params.setdefault('n_estimators', 100) # Kiek medziu bus sukurta
        params.setdefault('random_state', 42) # Atsitiktinumo sėkla, kad modelis nesikeistų
        params.setdefault('n_jobs', 2) # Kiek procesoriaus branduoliu bus naudojama
        self.model = RandomForestRegressor(**params)
    
    def train(self, X, y, **kwargs): # Gauname kintamuosius X ir y, juose yra informacija apie apylinkes ir balsu skaicius
        self.model.fit(X, y) # Apmokome modeli
    
    def predict(self, X): 
        return self.model.predict(X) # Gražiname prognozes

class NNModel(BaseModel):
    def __init__(self, **params):
        params.setdefault('hidden_layer_sizes', (64, 32)) # 2 sluoksniai po 64 ir 32 neuronus
        params.setdefault('max_iter', 500) # Maksimalus epochų skaičius, po tiek iteracijų modelis sustos
        params.setdefault('random_state', 42) # Atsitiktinumo sėkla, kad modelis nesikeistų
        self.model = Pipeline([
            ('scaler', StandardScaler(with_mean=False)), # Standartizuoja duomenis, kad modelis galetu geriau ismokti daryti prognozes
            ('mlp', MLPRegressor(**params)) # Neuroninis tinklas, jį naudoju nes jis gerai apdoroja kompleksinius ryšius tarp savybiu, bet jis yra gana letai apmokomas 
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
        self.models = models or {}
        self.weights = weights or {
            'rf': 0.20, 'nn': 0.20, 'catboost': 0.20,
            'xgboost': 0.15, 'lgbm': 0.15, 'elasticnet': 0.10
        }

    def predict(self, X):
        if not self.models:
            return np.zeros(len(X))
        
        import joblib
        import os
        
        # 1. Užkrauname produkcinį transformatorių
        preprocessor_path = 'app/ml/models/preprocessor.joblib'
        preprocessor = None
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        
        # 2. Pasiruošiame bazinius duomenis (CatBoost modeliui reikia tiesiog tekstų ir skaičių)
        cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
        num_features = ['RINKEJU_SKAICIUS']
        
        X_base = X[cat_features + num_features].copy()
        X_base[cat_features] = X_base[cat_features].astype(str).fillna('Unknown')
        for col in num_features:
            X_base[col] = pd.to_numeric(X_base[col], errors='coerce').fillna(0)
        
        # 3. Sukuriame užkoduotą matricą (Skirta Scikit-Learn ir XGBoost)
        X_encoded = preprocessor.transform(X_base) if preprocessor else None

        # Susinormalizuojame svorius
        total_w = sum(self.weights.values())
        norm_weights = {k.lower(): v/total_w for k, v in self.weights.items()}
        
        available_models = [m for m in self.models.keys() if m.lower() in norm_weights]
        if not available_models: return np.zeros(len(X))

        actual_total_w = sum(norm_weights[m.lower()] for m in available_models)
        redist_weights = {m: norm_weights[m.lower()]/actual_total_w for m in available_models}

        final_preds = np.zeros(len(X))

        # 4. Generuojame prognozes
        for name in available_models:
            is_catboost = 'catboost' in name.lower()
            
            if is_catboost:
                # CatBoost naudoja natyvius tekstinius laukus
                preds = self.models[name].predict(X_base)
            else:
                # Visi kiti modeliai naudoja transformatoriaus paruoštą matricą
                if X_encoded is None:
                    continue # Praleidžiame, jei transformatorius dingo
                preds = self.models[name].predict(X_encoded)
                
            preds = np.clip(preds, 0, 100)
            final_preds += preds * redist_weights[name]
        
        return np.clip(final_preds, 0, 100)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
