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