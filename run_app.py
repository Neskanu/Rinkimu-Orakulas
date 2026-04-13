from flask import Flask, render_template
from app.blueprints.dashboard.routes import dashboard_bp
from app.blueprints.forecasting.routes import forecast_bp
from app.blueprints.training.routes import training_bp
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-123')
    
    # Register Blueprints
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(forecast_bp, url_prefix='/forecast')
    app.register_blueprint(training_bp, url_prefix='/training')
    
    @app.route('/')
    def index():
        return render_template('index.html')
        
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
