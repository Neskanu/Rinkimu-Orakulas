from flask import Blueprint, render_template, request, jsonify
from app.data.database import SessionLocal
from app.data.repositories import ElectionRepository
from app.data.models import MLModelRegistry
from app.ml.pipeline.processor import DataProcessor
from sqlalchemy import select
import json
import plotly
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os

dashboard_bp = Blueprint('dashboard', __name__)

def get_vote_col(df):
    for col in ['BALSU_SKAICIUS', 'BALSU_VISO']:
        if col in df.columns:
            return col
    return None

def load_best_model():
    """Load the best available model for inference. Prefers CatBoost."""
    # Try CatBoost first (handles categorical natively)
    cb_path = 'app/ml/models/catboost_tuned.cbm'
    if not os.path.exists(cb_path):
        cb_path = 'app/ml/models/catboost_v1.cbm'
    if os.path.exists(cb_path):
        try:
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(cb_path)
            return m, 'catboost'
        except Exception as e:
            print(f"CatBoost load failed: {e}")

    # Fallback to RF
    rf_path = 'app/ml/models/rf_tuned.pkl'
    if not os.path.exists(rf_path):
        rf_path = 'app/ml/models/rf_model.pkl'
    if os.path.exists(rf_path):
        try:
            with open(rf_path, 'rb') as f:
                m = pickle.load(f)
            return m, 'rf'
        except Exception as e:
            print(f"RF load failed: {e}")
    return None, None

def run_inference_on_2024(test_df):
    """Run model inference on 2024 data and return DataFrame with PREDICTED column."""
    model, model_type = load_best_model()
    if model is None:
        return None, "No trained model found"

    features = ['RINKEJU_SKAICIUS', 'SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
    
    for col in ['RINKEJU_SKAICIUS']:
        # Replaced fillna(0) to strictly avoid the pandas DataFrame downcasting FutureWarning
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').replace({np.nan: 0})

    X = test_df[features].copy()

    if model_type == 'catboost':
        preds = model.predict(X)
    else:
        # Encode for sklearn models
        all_enc = pd.get_dummies(X, drop_first=True)
        
        # FIX: Ensure prediction matrix exactly matches the features the RF model expects
        # Note: model.feature_names_in_ requires sklearn 1.0+ and the model must have been trained with feature names.
        if hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
            all_enc = all_enc.reindex(columns=expected_cols, fill_value=0)
            
        preds = model.predict(all_enc.values.astype(np.float32))

    result_df = test_df[['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS',
                         'SARASO_PAVADINIMAS', 'VOTE_SHARE']].copy()
    
    # FIX: Clipped to 100 instead of 1 to preserve percentage scales
    result_df['PREDICTED'] = np.clip(preds, 0, 100) 
    return result_df, model_type


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
                               year=year, districts=districts, precincts=precincts)

    vote_col = get_vote_col(df)
    for col in [vote_col, 'VISO_DALYVAVO', 'RINKEJU_SKAICIUS']:
        if col:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if district:
        df = df[df['APYGARDOS_PAVADINIMAS'] == district]
    if precinct:
        df = df[df['APYLINKES_PAVADINIMAS'] == precinct]

    if vote_col not in df.columns or df.empty:
        return render_template('dashboard/index.html', plot_json=None, pie_json=None,
                               year=year, districts=districts, precincts=precincts)

    agg_df = df.groupby('SARASO_PAVADINIMAS')[vote_col].sum().reset_index()
    total_votes = agg_df[vote_col].sum()
    agg_df['SHARE_PCT'] = (agg_df[vote_col] / (total_votes if total_votes > 0 else 1)) * 100

    # Bar Chart (Top 10)
    top_10 = agg_df.sort_values(by='SHARE_PCT', ascending=True).tail(10).copy()
    top_10['DISPLAY_NAME'] = top_10['SARASO_PAVADINIMAS'].apply(
        lambda x: (x[:35] + '...') if len(x) > 38 else x)

    fig_bar = px.bar(top_10, x='SHARE_PCT', y='DISPLAY_NAME', orientation='h',
                     title=f"Actual Results: Top 10 Parties ({year})",
                     labels={'SHARE_PCT': 'Vote Share (%)', 'DISPLAY_NAME': 'Party'},
                     color='SHARE_PCT', color_continuous_scale='Viridis')
    fig_bar.update_layout(template='plotly_white', margin=dict(l=250),
                          yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
    bar_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # Pie Chart
    major_pie = agg_df[agg_df['SHARE_PCT'] >= 1.5].copy()
    other_share = agg_df[agg_df['SHARE_PCT'] < 1.5]['SHARE_PCT'].sum()
    if other_share > 0:
        new_row = pd.DataFrame({'SARASO_PAVADINIMAS': ['Kitos partijos'],
                                vote_col: [0], 'SHARE_PCT': [other_share]})
        major_pie = pd.concat([major_pie, new_row], ignore_index=True)

    fig_pie = px.pie(major_pie, values='SHARE_PCT', names='SARASO_PAVADINIMAS',
                     title="National Vote Distribution", hole=0.4)
    fig_pie.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        template='plotly_white')
    pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('dashboard/index.html', plot_json=bar_json, pie_json=pie_json,
                           year=year, districts=districts, precincts=precincts,
                           selected_district=district, selected_precinct=precinct)


@dashboard_bp.route('/backtest')
def backtest():
    session = SessionLocal()
    district_filter = request.args.get('district')
    precinct_filter = request.args.get('precinct')

    repo = ElectionRepository(session)
    districts = repo.get_districts(2024)
    precincts = repo.get_precincts(2024, district_filter) if district_filter else []

    # Fetch ALL trained models
    models = session.execute(
        select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc())
    ).scalars().all()
    session.close()

    # Load 2024 test data via DataProcessor (same pipeline as training)
    processor = DataProcessor()
    _, test_df = processor.prepare_training_data()

    scatter_json = None
    comp_json = None
    district_data = []
    model_type_label = "None"
    error_msg = None

    if test_df is not None and not test_df.empty:
        result_df, model_type_label = run_inference_on_2024(test_df)

        if result_df is not None:
            # Apply filters
            if district_filter:
                result_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == district_filter]
            if precinct_filter:
                result_df = result_df[result_df['APYLINKES_PAVADINIMAS'] == precinct_filter]

            if not result_df.empty:
                # --- Scatter: Actual vs Predicted per party-precinct ---
                party_agg = result_df.groupby('SARASO_PAVADINIMAS').agg(
                    ACTUAL=('VOTE_SHARE', 'mean'),
                    PREDICTED=('PREDICTED', 'mean')
                ).reset_index()

                fig_scatter = px.scatter(
                    party_agg, x='ACTUAL', y='PREDICTED',
                    hover_name='SARASO_PAVADINIMAS',
                    trendline='ols',
                    title=f"Actual vs Predicted Vote Share — {model_type_label.upper()} model",
                    labels={'ACTUAL': 'Actual Vote Share', 'PREDICTED': 'Predicted Vote Share'},
                    color_discrete_sequence=['#6366f1']
                )
                # Perfect-prediction reference line
                max_val = max(party_agg['ACTUAL'].max(), party_agg['PREDICTED'].max())
                fig_scatter.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                                      line=dict(color='red', dash='dash', width=1))
                fig_scatter.update_layout(template='plotly_white')
                scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

                # --- Bar: Actual vs Predicted national top parties ---
                comp_df = party_agg.sort_values('ACTUAL', ascending=False).head(15).copy()
                comp_df = comp_df.melt(id_vars=['SARASO_PAVADINIMAS'],
                                        value_vars=['ACTUAL', 'PREDICTED'],
                                        var_name='Type', value_name='Share')
                fig_comp = px.bar(
                    comp_df, x='Share', y='SARASO_PAVADINIMAS', color='Type',
                    orientation='h', barmode='group',
                    title=f"Actual vs Predicted — Top 15 Parties ({district_filter or 'National'})",
                    labels={'Share': 'Vote Share', 'SARASO_PAVADINIMAS': 'Party'},
                    color_discrete_map={'ACTUAL': '#6366f1', 'PREDICTED': '#f59e0b'}
                )
                fig_comp.update_layout(template='plotly_white', margin=dict(l=200))
                comp_json = json.dumps(fig_comp, cls=plotly.utils.PlotlyJSONEncoder)

                # --- District breakdown list ---
                for dist in sorted(result_df['APYGARDOS_PAVADINIMAS'].unique()):
                    dist_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == dist]

                    # Per-district party summary
                    dist_party = dist_df.groupby('SARASO_PAVADINIMAS').agg(
                        ACTUAL=('VOTE_SHARE', 'mean'),
                        PREDICTED=('PREDICTED', 'mean')
                    ).reset_index().sort_values('ACTUAL', ascending=False)

                    # Winner
                    top_actual = dist_party.iloc[0]['SARASO_PAVADINIMAS'] if not dist_party.empty else '-'
                    top_predicted = dist_party.sort_values('PREDICTED', ascending=False).iloc[0]['SARASO_PAVADINIMAS'] if not dist_party.empty else '-'
                    winner_match = top_actual == top_predicted

                    # Per-precinct breakdown within district
                    precincts_data = []
                    for prec in sorted(dist_df['APYLINKES_PAVADINIMAS'].unique()):
                        prec_df = dist_df[dist_df['APYLINKES_PAVADINIMAS'] == prec]
                        prec_parties = prec_df.groupby('SARASO_PAVADINIMAS').agg(
                            ACTUAL=('VOTE_SHARE', 'mean'),
                            PREDICTED=('PREDICTED', 'mean')
                        ).reset_index().sort_values('ACTUAL', ascending=False).head(8)
                        precincts_data.append({
                            'name': prec,
                            'parties': prec_parties.to_dict('records')
                        })

                    district_data.append({
                        'name': dist,
                        'top_actual': top_actual[:40],
                        'top_predicted': top_predicted[:40],
                        'winner_match': winner_match,
                        'parties': dist_party.head(8).to_dict('records'),
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
        models=models,
        districts=districts,
        precincts=precincts,
        district_data=district_data,
        model_type_label=model_type_label,
        selected_district=district_filter,
        selected_precinct=precinct_filter,
        error_msg=error_msg
    )
