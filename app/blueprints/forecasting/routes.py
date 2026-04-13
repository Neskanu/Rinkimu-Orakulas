from flask import Blueprint, render_template, request, jsonify
from app.data.database import SessionLocal
from app.data.models import EnsembleConfig
from app.ml.pipeline.processor import DataProcessor
from app.ml.pipeline.models import EnsembleModel, TreeModel, XGBModel, LGBMModel, PolyElasticNetModel, CatBoostModel
import joblib
import pandas as pd
import numpy as np
import os
from sqlalchemy import select

forecast_bp = Blueprint('forecast', __name__)

MODEL_MAP = {
    'rf': TreeModel,
    'xgboost': XGBModel,
    'lgbm': LGBMModel,
    'elasticnet': PolyElasticNetModel
}

# 2028 Archetype Mapping - Exact Training Strings
ARCHETYPES = {
    "conservative": "Tėvynės sąjunga - Lietuvos krikščionys demokratai",
    "social_democrat": "Lietuvos socialdemokratų partija",
    "liberal": "Lietuvos Respublikos liberalų sąjūdis",
    "populist": "Lietuvos valstiečių ir žaliųjų sąjunga",
    "independent": "AVERAGE" 
}

def load_ensemble():
    session = SessionLocal()
    stmt = select(EnsembleConfig).where(EnsembleConfig.is_active == True).order_by(EnsembleConfig.updated_at.desc()).limit(1)
    config = session.execute(stmt).scalar_one_or_none()
    session.close()

    models_dict = {}
    for mid in ['rf', 'xgboost', 'lgbm', 'elasticnet']:
        path = f'app/ml/models/{mid}_tuned.pkl'
        if os.path.exists(path):
            try:
                wrapped = MODEL_MAP[mid]()
                wrapped.model = joblib.load(path)
                models_dict[mid] = wrapped
            except Exception as e:
                print(f"Error loading {mid} in forecast: {e}")
    
    cb_path = 'app/ml/models/catboost_tuned.cbm'
    if os.path.exists(cb_path):
        from catboost import CatBoostRegressor
        m = CatBoostModel()
        m.model = CatBoostRegressor().load_model(cb_path)
        models_dict['catboost'] = m

    active_mids = list(models_dict.keys())
    if not active_mids: return None, []
    return EnsembleModel(models_dict, {mid: 1.0/len(active_mids) for mid in active_mids}), active_mids

@forecast_bp.route('/')
def index():
    return render_template('forecast/index.html')

@forecast_bp.route('/predict_2028', methods=['POST'])
def predict_2028():
    data = request.json
    contestants = data.get('contestants', []) 
    turnout_adj = data.get('turnout_adj', 0) / 100

    processor = DataProcessor()
    template = processor.prepare_2028_template()
    if template is None:
        return jsonify({"error": "No 2024 template found."}), 400

    ensemble, _ = load_ensemble()
    if not ensemble or 'catboost' not in ensemble.models:
        return jsonify({"error": "CatBoost model required for Oracle."}), 400

    # Ensure numeric for template
    template['VISO_DALYVAVO'] = pd.to_numeric(template['VISO_DALYVAVO'], errors='coerce').fillna(0)
    template['RINKEJU_SKAICIUS'] = pd.to_numeric(template['RINKEJU_SKAICIUS'], errors='coerce').replace(0, 1)
    template['VISO_DALYVAVO'] = template['VISO_DALYVAVO'] * (1 + turnout_adj)

    results = []
    # Variety baseline
    avg_preds_base = np.random.uniform(0.045, 0.08, len(template))

    for cont in contestants:
        name = cont['name']
        profile = cont['profile']
        if profile == 'independent':
            # Unique noise for each independent
            preds = avg_preds_base + np.random.normal(0, 0.005, len(template))
        else:
            archetype = ARCHETYPES.get(profile, ARCHETYPES['populist'])
            # IMPORTANT: Column order must exactly match the Training Set used by CatBoost
            X_raw = template[['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS', 'RINKEJU_SKAICIUS']].copy()
            X_raw['SARASO_PAVADINIMAS'] = archetype
            X_raw = X_raw[['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS', 'RINKEJU_SKAICIUS']]
            
            # Predict
            preds = ensemble.models['catboost'].predict(X_raw)
            # Add stochastic variety noise
            preds = preds + np.random.normal(0, 0.003, len(template))
            preds = np.clip(preds, 0, 1)

        cont_df = template[['APYGARDOS_PAVADINIMAS']].copy()
        cont_df['PRED_SHARE'] = preds
        cont_df['PARTY'] = name
        results.append(cont_df)

    final_df = pd.concat(results)
    
    # DISTRICT Normalization to 100% for contestants
    chart_data = []
    for dist in final_df['APYGARDOS_PAVADINIMAS'].unique():
        dist_p = final_df[final_df['APYGARDOS_PAVADINIMAS'] == dist].copy()
        
        # Mean share across all precincts in district for each party
        summary = dist_p.groupby('PARTY')['PRED_SHARE'].mean().reset_index()
        # Scale to 100% relative base
        total_p = summary['PRED_SHARE'].sum()
        summary['PRED_SHARE'] = (summary['PRED_SHARE'] / (total_p if total_p > 0 else 1)) * 100
        
        chart_data.append({
            "district": dist,
            "predictions": summary.to_dict('records')
        })

    return jsonify(chart_data)
