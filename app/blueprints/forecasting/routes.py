from flask import Blueprint, render_template, request, jsonify
from app.data.database import SessionLocal
from app.data.models import EnsembleConfig
from app.ml.pipeline.processor import DataProcessor
from app.ml.pipeline.models import (
    EnsembleModel, TreeModel, XGBModel, LGBMModel, 
    PolyElasticNetModel, CatBoostModel, NNModel,
    DeepNNModel, WideNNModel, SVRModel  # <-- PRIDĖKITE ŠIAS TRIS
)
import joblib
import pandas as pd
import numpy as np
import os
from sqlalchemy import select

forecast_bp = Blueprint('forecast', __name__)

MODEL_MAP = {
    'rf': TreeModel,
    'xgboost': XGBModel,
    'catboost': CatBoostModel,
    'nn': NNModel,
    'dnn': DeepNNModel,   # Pridėta
    'wnn': WideNNModel,   # Pridėta
    'svr': SVRModel       # Pridėta
}

# 2028 Archetype Mapping - Standartizuoti pavadinimai (sutampa su processor.py)
ARCHETYPES = {
    "conservative": "TS-LKD",
    "social_democrat": "LSDP",
    "liberal": "Liberalų sąjūdis",
    "populist": "LVŽS",
    "independent": "Nepriklausomas" 
}

def load_ensemble():
    session = SessionLocal()
    stmt = select(EnsembleConfig).where(EnsembleConfig.is_active == True).order_by(EnsembleConfig.updated_at.desc()).limit(1)
    config = session.execute(stmt).scalar_one_or_none()
    session.close()

    models_dict = {}
    for mid in ['rf', 'xgboost', 'lgbm', 'elasticnet', 'nn']:
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
    selected_model = data.get('model', 'catboost')

    # --- LIETUVIŠKO CIKLO LOGIKA ---
    # Seka: socdem -> conservative -> independent (nauja) -> social_democrat
    # Kadangi 2024 m. laimėjo Socdemai, 2028 m. boostą gauna Konservatoriai
    last_winner_profile = "social_democrat"
    cycle_boost_map = {
        "social_democrat": "conservative",
        "conservative": "independent",
        "independent": "social_democrat",
        "liberal": "social_democrat",
        "populist": "conservative"
    }
    target_boost_profile = cycle_boost_map.get(last_winner_profile)

    processor = DataProcessor()
    template = processor.prepare_2028_template()
    if template is None:
        return jsonify({"error": "No 2024 template found."}), 400

    ensemble, _ = load_ensemble()
    if not ensemble or selected_model not in ensemble.models:
        return jsonify({"error": f"Modelis '{selected_model.upper()}' nerastas."}), 400

    # Transformatoriaus krovimas kitiems modeliams nei CatBoost
    preprocessor = None
    if selected_model != 'catboost':
        preprocessor_path = 'app/ml/models/preprocessor.joblib'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        else:
            return jsonify({"error": "Nerastas preprocessor.joblib. Apmokykite modelius."}), 400

    # Duomenų paruošimas
    template['VISO_DALYVAVO'] = pd.to_numeric(template['VISO_DALYVAVO'], errors='coerce').fillna(0)
    template['RINKEJU_SKAICIUS'] = pd.to_numeric(template['RINKEJU_SKAICIUS'], errors='coerce').replace(0, 1)
    template['VISO_DALYVAVO'] = template['VISO_DALYVAVO'] * (1 + turnout_adj)

    results = []
    avg_preds_base = np.random.uniform(4.5, 8.0, len(template))

    for cont in contestants:
        name = cont['name']
        profile = cont['profile']
        
        if profile == 'independent':
            preds = avg_preds_base + np.random.normal(0, 0.5, len(template))
        else:
            archetype = ARCHETYPES.get(profile, ARCHETYPES['populist'])
            X_raw = template[['APYGARDOS_PAVADINIMAS', 'RINKEJU_SKAICIUS']].copy()
            X_raw['SARASO_PAVADINIMAS'] = archetype
            X_raw = X_raw[['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS', 'RINKEJU_SKAICIUS']]
            
            if selected_model == 'catboost':
                preds = ensemble.models[selected_model].predict(X_raw)
            else:
                X_encoded = preprocessor.transform(X_raw)
                preds = ensemble.models[selected_model].predict(X_encoded)
            
            # Atsitiktinis triukšmas (0.3%)
            preds = preds + np.random.normal(0, 0.3, len(template))

        # --- CIKLO KOREKCIJA ---
        cycle_multiplier = 1.0
        if profile == target_boost_profile:
            cycle_multiplier = 1.15 # +15% švytuoklės efektas
        elif profile == last_winner_profile:
            cycle_multiplier = 0.90 # -10% valdžios nuovargis

        preds = np.clip(preds * cycle_multiplier, 0, 100)

        cont_df = template[['APYGARDOS_PAVADINIMAS']].copy()
        cont_df['PRED_SHARE'] = preds
        cont_df['PARTY'] = name
        results.append(cont_df)

    # Normalizacija į 100% per apygardą
    final_df = pd.concat(results)
    chart_data = []
    for dist in final_df['APYGARDOS_PAVADINIMAS'].unique():
        dist_p = final_df[final_df['APYGARDOS_PAVADINIMAS'] == dist].copy()
        summary = dist_p.groupby('PARTY')['PRED_SHARE'].mean().reset_index()
        total_p = summary['PRED_SHARE'].sum()
        summary['PRED_SHARE'] = (summary['PRED_SHARE'] / (total_p if total_p > 0 else 1)) * 100
        
        chart_data.append({
            "district": dist,
            "predictions": summary.to_dict('records')
        })

    return jsonify(chart_data)