from flask import Blueprint, render_template, request, jsonify
from app.data.database import SessionLocal, get_db
from app.data.repositories import ElectionRepository
from app.data.models import MLModelRegistry, EnsembleConfig
from app.ml.pipeline.processor import DataProcessor
from app.ml.pipeline.models import EnsembleModel
from sqlalchemy import select
import json
import plotly
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import os

dashboard_bp = Blueprint('dashboard', __name__)

def get_vote_col(df):
    for col in ['BALSU_SKAICIUS', 'BALSU_VISO']:
        if col in df.columns:
            return col
    return None

def load_model_instance(model_name="Ensemble"):
    """Load a specific model instance or the ensemble."""
    base_path = 'app/ml/models'
    
    # Consolidate configuration to use SHORT IDs consistently across the app
    model_configs = {
        'rf': ['rf_tuned.pkl', 'rf_model.pkl'],
        'nn': ['nn_model.pt', 'nn_model.pkl'],
        'catboost': ['catboost_tuned.cbm', 'catboost_v1.cbm'],
        'xgboost': ['xgboost_tuned.pkl', 'xgboost_model.pkl'],
        'lgbm': ['lgbm_tuned.pkl', 'lgbm_model.pkl'],
        'elasticnet': ['elasticnet_tuned.pkl', 'elasticnet_model.pkl']
    }

    if model_name == "Ensemble":
        models = {}
        for name, opts in model_configs.items():
            for filename in opts:
                path = os.path.join(base_path, filename)
                if os.path.exists(path):
                    try:
                        if filename.endswith('.cbm'):
                            from catboost import CatBoostRegressor
                            m = CatBoostRegressor()
                            m.load_model(path)
                            models[name] = m
                        else:
                            models[name] = joblib.load(path)
                        break
                    except Exception as e:
                        print(f"Failed to load {name}: {e}")
        
        # Load weights from DB
        with next(get_db()) as db:
            config = db.query(EnsembleConfig).filter_by(is_active=True).first()
            if not config:
                config = EnsembleConfig()
            
            weights = {
                'RandomForest': config.rf_weight,
                'NeuralNetwork': config.nn_weight,
                'CatBoost': config.catboost_weight,
                'XGBoost': config.xgboost_weight,
                'LightGBM': config.lgbm_weight,
                'ElasticNet': config.elasticnet_weight
            }
            
        return EnsembleModel(models, weights), "Ensemble"

    # Load individual model
    if model_name in model_configs:
        for filename in model_configs[model_name]:
            path = os.path.join(base_path, filename)
            if os.path.exists(path):
                try:
                    if path.endswith('.cbm'):
                        from catboost import CatBoostRegressor
                        m = CatBoostRegressor()
                        m.load_model(path)
                        return m, model_name
                    else:
                        return joblib.load(path), model_name
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                
    return None, None
                
    return None, None

def run_inference(test_df, model_name="Ensemble"):
    """Run model inference and return result DataFrame."""
    model, label = load_model_instance(model_name)
    if model is None:
        return None, "No model found"

    features = ['RINKEJU_SKAICIUS', 'SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
    X = test_df[features].copy()

    # Pre-process numeric
    X['RINKEJU_SKAICIUS'] = pd.to_numeric(X['RINKEJU_SKAICIUS'], errors='coerce').fillna(0)

    # Model-specific preprocessing
    model_type = label.lower().replace(" ", "").replace("_", "")
    
    if "catboost" in model_type or "ensemble" in model_type:
        # CatBoost uses raw, Ensemble handles its own internal branching
        preds = model.predict(X)
    else:
        # Standard sklearn-style models (RF, XGB, NN, etc.) need encoding
        import pickle
        from scipy.sparse import hstack
        encoder_path = 'app/ml/models/feature_encoder.pkl'
        
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            
            cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
            num_features = ['RINKEJU_SKAICIUS']
            X_cat = encoder.transform(X[cat_features])
            X_enc = hstack([X[num_features].values, X_cat])
            preds = model.predict(X_enc)
        else:
            print(f"⚠️  CRITICAL: {encoder_path} missing. Falling back to raw data (this will likely fail). PLEASE RETRAIN MODELS.")
            preds = model.predict(X)

    result_df = test_df[['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS',
                         'SARASO_PAVADINIMAS', 'VOTE_SHARE']].copy()
    
    # Ensure 0-100 scale
    result_df['PREDICTED'] = np.clip(preds, 0, 100) 
    return result_df, label


@dashboard_bp.route('/')
def index():
    session = SessionLocal()
    year = request.args.get('year', 2024, type=int)
    district = request.args.get('district')
    precinct = request.args.get('precinct')

    repo = ElectionRepository(session)
    df = repo.get_dataframe_for_ml(year)
    districts = repo.get_districts(year)
    precincts = repo.get_precincts(year, district) if district else []
    session.close()

    if df.empty:
        return render_template('dashboard/index.html', plot_json=None, pie_json=None,
                               year=year, districts=districts, precincts=precincts,
                               metrics={})

    vote_col = get_vote_col(df)
    for col in [vote_col, 'VISO_DALYVAVO', 'RINKEJU_SKAICIUS']:
        if col and col in df.columns:
            # FIX 1: Strip out European number formatting (spaces and commas) first
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.')
            
            # Convert to pure floats, safely replacing failures with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').replace({np.nan: 0})

    if district:
        df = df[df['APYGARDOS_PAVADINIMAS'] == district]
    if precinct:
        df = df[df['APYLINKES_PAVADINIMAS'] == precinct]

    if vote_col not in df.columns or df.empty:
        return render_template('dashboard/index.html', plot_json=None, pie_json=None,
                               year=year, districts=districts, precincts=precincts,
                               metrics={})

    agg_df = df.groupby('SARASO_PAVADINIMAS')[vote_col].sum().reset_index()
    total_votes = agg_df[vote_col].sum()
    
    # FIX 2: Safeguard to prevent the 7.14% equal-slice bug. 
    # If total_votes is 0, it means no valid numeric data exists for this filter.
    if total_votes == 0:
        return render_template('dashboard/index.html', plot_json=None, pie_json=None,
                               year=year, districts=districts, precincts=precincts,
                               metrics={})

    # Now perfectly safe to calculate real percentages
    agg_df['SHARE_PCT'] = (agg_df[vote_col] / total_votes) * 100

    # Bar Chart (Top 10)
    top_10 = agg_df.sort_values(by='SHARE_PCT', ascending=True).tail(10).copy()
    top_10['DISPLAY_NAME'] = top_10['SARASO_PAVADINIMAS'].apply(
        lambda x: (x[:35] + '...') if len(x) > 38 else x)

    # Add text parameter to place percentages directly on the bars, pass full name and votes for tooltips
    fig_bar = px.bar(top_10, x='SHARE_PCT', y='DISPLAY_NAME', orientation='h',
                     title=f"Actual Results: Top 10 Parties ({year})",
                     labels={'SHARE_PCT': 'Vote Share (%)', 'DISPLAY_NAME': 'Party'},
                     color='SHARE_PCT', color_continuous_scale='Viridis',
                     text=top_10['SHARE_PCT'].apply(lambda x: f"{x:.1f}%"),
                     custom_data=['SARASO_PAVADINIMAS', vote_col])

    # Push text outside bars and format the popout tooltip
    fig_bar.update_traces(
        textposition='outside',
        hovertemplate='<b>%{customdata[0]}</b><br>Votes: %{customdata[1]:,}<br>Share: %{x:.2f}%<extra></extra>'
    )
    
    # Extend the X-axis range to prevent the outside text from clipping on the right edge
    max_share = top_10['SHARE_PCT'].max()
    fig_bar.update_layout(
        template='plotly_white', 
        margin=dict(l=250, r=50), 
        yaxis={'categoryorder': 'total ascending'}, 
        xaxis=dict(range=[0, max_share * 1.15]), # 15% buffer space for the text label
        coloraxis_showscale=False
    )
    bar_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # --- PIE CHART ---
    major_pie = agg_df[agg_df['SHARE_PCT'] >= 1.5].copy()
    other_share = agg_df[agg_df['SHARE_PCT'] < 1.5]['SHARE_PCT'].sum()
    other_votes = agg_df[agg_df['SHARE_PCT'] < 1.5][vote_col].sum()
    
    if other_share > 0:
        new_row = pd.DataFrame({'SARASO_PAVADINIMAS': ['Kitos partijos'],
                                vote_col: [other_votes], 'SHARE_PCT': [other_share]})
        major_pie = pd.concat([major_pie, new_row], ignore_index=True)

    # THE FIX: Force BOTH columns to pure floats so Plotly cannot fall back to counting rows
    major_pie['SHARE_PCT'] = major_pie['SHARE_PCT'].astype(float)
    major_pie[vote_col] = major_pie[vote_col].astype(float)

    # Bind values directly to SHARE_PCT to guarantee perfect alignment with the Bar Chart
    fig_pie = px.pie(major_pie, values='SHARE_PCT', names='SARASO_PAVADINIMAS',
                     title="National Vote Distribution", hole=0.4,
                     custom_data=[vote_col])
    
    # Format the tooltip so it reads the absolute votes from customdata and the percentage from value
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>Votes: %{customdata[0]:,}<br>Share: %{value:.2f}%<extra></extra>'
    )
    
    fig_pie.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        template='plotly_white',
        margin=dict(t=50, b=100, l=20, r=20)
    )
    pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    # Fetch metrics from registry for display
    metrics = {}
    session_metrics = SessionLocal()
    db_metrics = session_metrics.execute(
        select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc())
    ).scalars().all()
    
    # Group by model name, take latest
    for m in db_metrics:
        if m.model_name not in metrics:
            metrics[m.model_name] = {
                "mae": m.mae, 
                "type": m.model_type,
                "date": m.training_date
            }
    session_metrics.close()

    return render_template('dashboard/index.html', plot_json=bar_json, pie_json=pie_json,
                           year=year, districts=districts, precincts=precincts,
                           selected_district=district, selected_precinct=precinct,
                           metrics=metrics)


@dashboard_bp.route('/backtest')
def backtest():
    session = SessionLocal()
    district_filter = request.args.get('district')
    precinct_filter = request.args.get('precinct')
    selected_model = request.args.get('model', 'Ensemble')

    repo = ElectionRepository(session)
    districts = repo.get_districts(2024)
    precincts = repo.get_precincts(2024, district_filter) if district_filter else []

    # Get available models for the dropdown
    available_models = [
        "Ensemble", "RandomForest", "NeuralNetwork", 
        "CatBoost", "XGBoost", "LightGBM", "ElasticNet"
    ]
    
    # Fetch metrics from registry for display
    metrics = {}
    db_metrics = session.execute(
        select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc())
    ).scalars().all()
    
    # Group by model name, take latest
    for m in db_metrics:
        if m.model_name not in metrics:
            metrics[m.model_name] = {
                "mae": m.mae,
                "type": m.model_type,
                "date": m.training_date
            }

    session.close()

    # Load 2024 test data via DataProcessor
    processor = DataProcessor()
    _, test_df = processor.prepare_training_data()

    scatter_json = None
    comp_json = None
    district_data = []
    error_msg = None

    if test_df is not None and not test_df.empty:
        result_df, label = run_inference(test_df, selected_model)

        if result_df is not None:
            # Apply filters
            if district_filter:
                result_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == district_filter]
            if precinct_filter:
                result_df = result_df[result_df['APYLINKES_PAVADINIMAS'] == precinct_filter]

            if not result_df.empty:
                # --- Scatter: Actual vs Predicted ---
                # Sample for scatter to keep it performant
                scatter_df = result_df.copy()
                if len(scatter_df) > 5000:
                    scatter_df = scatter_df.sample(5000)

                fig_scatter = px.scatter(
                    scatter_df, x='VOTE_SHARE', y='PREDICTED',
                    hover_data=['SARASO_PAVADINIMAS', 'APYLINKES_PAVADINIMAS'],
                    title=f"Actual vs Predicted Vote Share — {label}",
                    labels={'VOTE_SHARE': 'Actual Share (%)', 'PREDICTED': 'Predicted Share (%)'},
                    opacity=0.4,
                    color_discrete_sequence=['#6366f1']
                )
                fig_scatter.add_shape(type='line', x0=0, y0=0, x1=100, y1=100,
                                      line=dict(color='red', dash='dash', width=1))
                fig_scatter.update_layout(template='plotly_white', xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 100]))
                scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

                # --- Bar: Top 15 Comparison ---
                party_agg = result_df.groupby('SARASO_PAVADINIMAS').agg({
                    'VOTE_SHARE': 'mean',
                    'PREDICTED': 'mean'
                }).reset_index()
                
                comp_df = party_agg.sort_values('VOTE_SHARE', ascending=False).head(15).copy()
                comp_df = comp_df.melt(id_vars=['SARASO_PAVADINIMAS'],
                                        value_vars=['VOTE_SHARE', 'PREDICTED'],
                                        var_name='Type', value_name='Share')
                
                fig_comp = px.bar(
                    comp_df, x='Share', y='SARASO_PAVADINIMAS', color='Type',
                    orientation='h', barmode='group',
                    title=f"Actual vs Predicted Top Parties ({district_filter or 'National'})",
                    labels={'Share': 'Vote Share (%)', 'SARASO_PAVADINIMAS': 'Party'},
                    color_discrete_map={'VOTE_SHARE': '#6366f1', 'PREDICTED': '#f59e0b'}
                )
                fig_comp.update_layout(template='plotly_white', margin=dict(l=200))
                comp_json = json.dumps(fig_comp, cls=plotly.utils.PlotlyJSONEncoder)

                # --- District breakdown ---
                for dist in sorted(result_df['APYGARDOS_PAVADINIMAS'].unique()):
                    dist_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == dist]
                    dist_party = dist_df.groupby('SARASO_PAVADINIMAS').agg({
                        'VOTE_SHARE': 'mean',
                        'PREDICTED': 'mean'
                    }).reset_index().sort_values('VOTE_SHARE', ascending=False)

                    # Determine winners for the UI labels
                    top_actual = dist_party.iloc[0]['SARASO_PAVADINIMAS'] if not dist_party.empty else 'N/A'
                    top_pred_df = dist_party.sort_values('PREDICTED', ascending=False)
                    top_predicted = top_pred_df.iloc[0]['SARASO_PAVADINIMAS'] if not top_pred_df.empty else 'N/A'
                    winner_match = (top_actual == top_predicted)

                    # Sub-precincts
                    precincts_data = [] # Fixed: was prec_data in initialization
                    for prec in sorted(dist_df['APYLINKES_PAVADINIMAS'].unique()):
                        p_df = dist_df[dist_df['APYLINKES_PAVADINIMAS'] == prec]
                        prec_parties = p_df.groupby('SARASO_PAVADINIMAS').agg({
                            'VOTE_SHARE': 'mean',
                            'PREDICTED': 'mean'
                        }).reset_index().sort_values('VOTE_SHARE', ascending=False).head(5)
                        
                        precincts_data.append({
                            'name': prec,
                            'parties': prec_parties.to_dict('records')
                        })

                    district_data.append({
                        'name': dist,
                        'top_actual': top_actual,
                        'top_predicted': top_predicted,
                        'winner_match': winner_match,
                        'parties': dist_party.head(5).to_dict('records'),
                        'precincts': precincts_data
                    })
        else:
            error_msg = "Model inference failed. Please train a model first."
    else:
        error_msg = "No 2024 test data available."

    return render_template(
        'dashboard/backtest.html',
        scatter_json=scatter_json,
        comp_json=comp_json,
        metrics=metrics,
        available_models=available_models,
        selected_model=selected_model,
        districts=districts,
        precincts=precincts,
        district_data=district_data,
        selected_district=district_filter,
        selected_precinct=precinct_filter,
        error=error_msg
    )