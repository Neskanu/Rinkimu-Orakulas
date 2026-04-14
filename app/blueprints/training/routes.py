from flask import Blueprint, render_template, request, jsonify
from app.data.database import SessionLocal
from app.data.models import MLModelRegistry, EnsembleConfig
from app.ml.pipeline.trainer import train_suite
import threading
import json
import os

training_bp = Blueprint('training', __name__)

@training_bp.route('/')
def index():
    with SessionLocal() as session:
        # Ištraukiame konfigūraciją ir modelius kol sesija dar atidaryta
        config = session.query(EnsembleConfig).first()
        models = session.query(MLModelRegistry).order_by(MLModelRegistry.training_date.desc()).all()
        return render_template('training/index.html', config=config, models=models)

@training_bp.route('/status', methods=['GET'])
def get_status():
    """Grąžina dabartinį modelių apmokymo progresą iš JSON failo."""
    progress_file = 'training_progress.json'
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return jsonify(data)
        except Exception:
            return jsonify({"status": "reading_error"})
    return jsonify({"status": "idle"})

@training_bp.route('/start_training', methods=['POST'])
def start_training():
    """Asinchroniškai paleidžia apmokymą naujoje gijoje."""
    data = request.get_json() or {}
    selected_models = data.get('models', ['rf', 'catboost', 'xgboost', 'lgbm', 'elasticnet', 'nn'])
    
    # PRIIMAME HIPERPARAMETRUS IŠ JŪSŲ JS FUNKCIJOS
    model_params = data.get('params', {}) 
    
    # Perduodame parametrus į giją
    thread = threading.Thread(target=train_suite, args=(selected_models, model_params))
    thread.daemon = True 
    thread.start()
    
    return jsonify({"message": "Apmokymas pradėtas fone!", "status": "started"}), 202

@training_bp.route('/update_weights', methods=['POST'])
def update_weights():
    data = request.json  # Čia gausime pvz: {"rf": 0.1, "dnn": 0.2, ...}
    
    with SessionLocal() as session:
        # Paimame pirmą (ir vienintelę) konfigūraciją
        config = session.query(EnsembleConfig).first()
        
        if not config:
            config = EnsembleConfig()
            session.add(config)
        
        # Dinamiškai atnaujiname svorius
        for model_id, weight in data.items():
            # Sukonstruojame stulpelio pavadinimą, pvz., 'rf' -> 'rf_weight'
            column_name = f"{model_id}_weight"
            
            # Patikriname, ar toks stulpelis egzistuoja duomenų modelyje
            if hasattr(config, column_name):
                setattr(config, column_name, weight)
                print(f"Updating {column_name} to {weight}")
        
        try:
            session.commit()
            return jsonify({"status": "success", "message": "Weights updated"}), 200
        except Exception as e:
            session.rollback()
            print(f"Database error while saving weights: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500