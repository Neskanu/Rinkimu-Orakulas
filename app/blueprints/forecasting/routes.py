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
    
    # Ištraukiame vartotojo pasirinktą modelį (numatytasis - catboost)
    selected_model = data.get('model', 'catboost')

    processor = DataProcessor()
    template = processor.prepare_2028_template()
    if template is None:
        return jsonify({"error": "No 2024 template found."}), 400

    ensemble, _ = load_ensemble()
    if not ensemble or selected_model not in ensemble.models:
        return jsonify({"error": f"Modelis '{selected_model.upper()}' nerastas. Pirmiausia apmokykite jį."}), 400

    # Jei tai ne CatBoost, mums būtinai reikia transformatoriaus (OneHotEncoder)
    preprocessor = None
    if selected_model != 'catboost':
        preprocessor_path = 'app/ml/models/preprocessor.joblib'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        else:
            return jsonify({"error": "Nerastas preprocessor.joblib failas. Apmokykite modelius iš naujo."}), 400

    # Ensure numeric for template
    template['VISO_DALYVAVO'] = pd.to_numeric(template['VISO_DALYVAVO'], errors='coerce').fillna(0)
    template['RINKEJU_SKAICIUS'] = pd.to_numeric(template['RINKEJU_SKAICIUS'], errors='coerce').replace(0, 1)
    template['VISO_DALYVAVO'] = template['VISO_DALYVAVO'] * (1 + turnout_adj)

    results = []
    # PATAISYTA: Bazė nepriklausomiems (4.5% - 8.0%), nes skalė yra 0-100
    avg_preds_base = np.random.uniform(4.5, 8.0, len(template))

    for cont in contestants:
        name = cont['name']
        profile = cont['profile']
        if profile == 'independent':
            # PATAISYTA: Triukšmas pritaikytas 0-100 skalei
            preds = avg_preds_base + np.random.normal(0, 0.5, len(template))
        else:
            archetype = ARCHETYPES.get(profile, ARCHETYPES['populist'])
            
            X_raw = template[['APYGARDOS_PAVADINIMAS', 'RINKEJU_SKAICIUS']].copy()
            X_raw['SARASO_PAVADINIMAS'] = archetype
            X_raw = X_raw[['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS', 'RINKEJU_SKAICIUS']]
            
            # --- MODELIO INFERENCIJOS LOGIKA ---
            if selected_model == 'catboost':
                X_raw['SARASO_PAVADINIMAS'] = X_raw['SARASO_PAVADINIMAS'].astype(str)
                X_raw['APYGARDOS_PAVADINIMAS'] = X_raw['APYGARDOS_PAVADINIMAS'].astype(str)
                preds = ensemble.models[selected_model].predict(X_raw)
            else:
                # Scikit-Learn ir XGBoost modeliams naudojame užkoduotą matricą
                X_encoded = preprocessor.transform(X_raw)
                preds = ensemble.models[selected_model].predict(X_encoded)
            
            # PATAISYTA: Triukšmas pritaikytas 0-100 skalei (0.3% svyravimas)
            preds = preds + np.random.normal(0, 0.3, len(template))
            preds = np.clip(preds, 0, 100)

        cont_df = template[['APYGARDOS_PAVADINIMAS']].copy()
        cont_df['PRED_SHARE'] = preds
        cont_df['PARTY'] = name
        results.append(cont_df)

    final_df = pd.concat(results)
    
    # DISTRICT Normalization to 100% for contestants
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