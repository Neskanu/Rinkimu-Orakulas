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

class BaseModel:
    def train(self, X, y, **kwargs):
        pass
    def predict(self, X):
        pass

class TreeModel(BaseModel):
    def __init__(self, **params):
        params.setdefault('n_estimators', 100)
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', -1)
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
            ('scaler', StandardScaler()),
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
        params.setdefault('verbose', -1)
        self.model = LGBMRegressor(**params)
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class PolyElasticNetModel(BaseModel):
    def __init__(self, degree=2, **params):
        params.setdefault('random_state', 42)
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('en', ElasticNet(**params))
        ])
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class EnsembleModel(BaseModel):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        # Average weighted prediction
        final_preds = None
        for mid, model in self.models.items():
            preds = np.array(model.predict(X))
            weight = self.weights.get(mid, 0)
            if final_preds is None:
                final_preds = preds * weight
            else:
                final_preds += preds * weight
        return final_preds
