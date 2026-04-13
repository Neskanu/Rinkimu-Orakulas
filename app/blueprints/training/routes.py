from flask import Blueprint, render_template, request, jsonify
from app.data.database import SessionLocal
from app.data.models import MLModelRegistry, EnsembleConfig
from app.ml.pipeline.trainer import train_suite, PROGRESS_FILE
import threading
import json
import os
from datetime import datetime
from sqlalchemy import select, update

training_bp = Blueprint('training', __name__)

@training_bp.route('/')
def index():
    session = SessionLocal()
    # Get dynamic weights
    stmt = select(EnsembleConfig).where(EnsembleConfig.is_active == True).order_by(EnsembleConfig.updated_at.desc()).limit(1)
    config = session.execute(stmt).scalar_one_or_none()
    
    if not config:
        # Create default config if missing
        config = EnsembleConfig(rf_weight=0.33, nn_weight=0.33, catboost_weight=0.34)
        session.add(config)
        session.commit()
    
    # Get models for history
    stmt_models = select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc()).limit(10)
    models = session.execute(stmt_models).scalars().all()
    
    session.close()
    return render_template('training/index.html', config=config, models=models)

@training_bp.route('/start', methods=['POST'])
def start_training():
    data = request.json
    timeout = data.get('timeout', 60)
    models = data.get('models', [])
    params = data.get('params', {})
    
    # Run trainer in background thread
    thread = threading.Thread(target=train_suite, args=(models, params, timeout))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "message": f"Training started for {len(models)} models."})

@training_bp.route('/metrics')
def get_metrics():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})

@training_bp.route('/update_weights', methods=['POST'])
def update_weights():
    data = request.json
    session = SessionLocal()
    try:
        # Get active config
        stmt = select(EnsembleConfig).where(EnsembleConfig.is_active == True).order_by(EnsembleConfig.updated_at.desc()).limit(1)
        config = session.execute(stmt).scalar_one_or_none()
        
        if not config:
            config = EnsembleConfig(is_active=True)
            session.add(config)
        
        # Update weights from JSON (0-1 range)
        config.rf_weight = float(data.get('rf', 0))
        config.nn_weight = float(data.get('nn', 0))
        config.catboost_weight = float(data.get('catboost', 0))
        config.xgboost_weight = float(data.get('xgboost', 0))
        config.lgbm_weight = float(data.get('lgbm', 0))
        config.elasticnet_weight = float(data.get('elasticnet', 0))
        
        config.updated_at = datetime.now()
        session.commit()
        return jsonify({"status": "success", "message": "Ensemble weights updated successfully."})
    except Exception as e:
        session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        session.close()
