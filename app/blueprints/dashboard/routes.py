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
    """Identify the specific party vote column, strictly avoiding precinct-level totals."""
    party_vote_columns = [
        'BALSU_SKAICIUS',          
        'BALSAI_UZ_SARASA',        
        'PADUOTI_BALSAI',          
        'GALI_BALSAI',             
        'BALSU_VISO'               
    ]
    for col in party_vote_columns:
        if col in df.columns:
            return col
    return None

def load_model_by_id(model_id=None):
    """Load a model from the registry. If model_id is None, use the one with lowest MAE."""
    session = SessionLocal()
    from sqlalchemy import select
    if model_id:
        model_entry = session.get(MLModelRegistry, model_id)
    else:
        # Get the latest trained model with the lowest MAE
        model_entry = session.execute(
            select(MLModelRegistry).order_by(MLModelRegistry.mae.asc())
        ).scalars().first()
    
    if not model_entry:
        session.close()
        return None, None, None
    
    path = model_entry.file_path
    model_type = model_entry.model_type
    session.close()
    
    if not os.path.exists(path):
        return None, None, None
        
    try:
        if model_type == 'catboost':
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(path)
            return m, 'catboost', model_entry.id
        else:
            with open(path, 'rb') as f:
                m = pickle.load(f)
            return m, model_type, model_entry.id
    except Exception:
        return None, None, None

def run_inference_on_2024(test_df, model_id=None):
    """Run inference using the specified model."""
    model, model_type, actual_id = load_model_by_id(model_id)
    
    if model is None:
        return None, "No valid model found", None

    cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
    num_features = ['RINKEJU_SKAICIUS']
    features = cat_features + num_features
    
    for col in num_features:
        # Strictly avoid the pandas DataFrame downcasting FutureWarning
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').replace({np.nan: 0})

    # Ensure categorical features are strings (critical for CatBoost)
    for col in cat_features:
        test_df[col] = test_df[col].astype(str).replace('nan', 'Unknown')
    
    X = test_df[features].copy()

    if model_type == 'catboost':
        # CatBoost expects categorical features to be strings
        X[cat_features] = X[cat_features].astype(str)
        preds = model.predict(X)
    else:
        all_enc = pd.get_dummies(X, drop_first=True)
        if hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
            all_enc = all_enc.reindex(columns=expected_cols, fill_value=0)
        preds = model.predict(all_enc.values.astype(np.float32))

    result_df = test_df[['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS',
                         'SARASO_PAVADINIMAS', 'VOTE_SHARE']].copy()
    
    result_df['PREDICTED'] = np.clip(preds, 0, 100) 
    return result_df, model_type, actual_id


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
        if col and col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.')
            # Strictly avoid the pandas DataFrame downcasting FutureWarning
            df[col] = pd.to_numeric(df[col], errors='coerce').replace({np.nan: 0})

    if district:
        df = df[df['APYGARDOS_PAVADINIMAS'] == district]
    if precinct:
        df = df[df['APYLINKES_PAVADINIMAS'] == precinct]

    if vote_col not in df.columns or df.empty:
        return render_template('dashboard/index.html', plot_json=None, pie_json=None,
                               year=year, districts=districts, precincts=precincts)

    agg_df = df.groupby('SARASO_PAVADINIMAS')[vote_col].sum().reset_index()
    total_votes = agg_df[vote_col].sum()
    
    if total_votes == 0:
        return render_template('dashboard/index.html', plot_json=None, pie_json=None,
                               year=year, districts=districts, precincts=precincts)

    agg_df['SHARE_PCT'] = (agg_df[vote_col] / total_votes) * 100

    # --- Bar Chart ---
    top_10 = agg_df.sort_values(by='SHARE_PCT', ascending=True).tail(10).copy()
    top_10['DISPLAY_NAME'] = top_10['SARASO_PAVADINIMAS'].apply(
        lambda x: (x[:35] + '...') if len(x) > 38 else x)

    fig_bar = px.bar(top_10, x='SHARE_PCT', y='DISPLAY_NAME', orientation='h',
                     title=f"Actual Results: Top 10 Parties ({year})",
                     labels={'SHARE_PCT': 'Vote Share (%)', 'DISPLAY_NAME': 'Party'},
                     color='SHARE_PCT', color_continuous_scale='Viridis',
                     text=top_10['SHARE_PCT'].apply(lambda x: f"{x:.1f}%"),
                     custom_data=['SARASO_PAVADINIMAS', vote_col])

    fig_bar.update_traces(
        textposition='outside',
        hovertemplate='<b>%{customdata[0]}</b><br>Votes: %{customdata[1]:,.0f}<br>Share: %{x:.2f}%<extra></extra>'
    )
    
    max_share = top_10['SHARE_PCT'].max()
    fig_bar.update_layout(
        template='plotly_white', 
        margin=dict(l=250, r=50), 
        yaxis={'categoryorder': 'total ascending'}, 
        xaxis=dict(range=[0, max_share * 1.15]), 
        coloraxis_showscale=False
    )
    bar_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # --- Pie Chart (Safe Method) ---
    pie_data = []
    other_votes = 0.0
    other_share = 0.0

    for _, row in agg_df.iterrows():
        if row['SHARE_PCT'] >= 1.5:
            pie_data.append({
                'Party': str(row['SARASO_PAVADINIMAS']), 
                'Votes': float(row[vote_col]), 
                'Share': float(row['SHARE_PCT'])
            })
        else:
            other_votes += float(row[vote_col])
            other_share += float(row['SHARE_PCT'])
            
    if other_share > 0:
        pie_data.append({
            'Party': 'Kitos partijos', 
            'Votes': float(other_votes), 
            'Share': float(other_share)
        })

    clean_pie_df = pd.DataFrame(pie_data)

    fig_pie = px.pie(clean_pie_df, values='Share', names='Party',
                     title="National Vote Distribution", hole=0.4,
                     custom_data=['Votes'])
    
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>Votes: %{customdata[0]:,.0f}<br>Share: %{value:.2f}%<extra></extra>'
    )
    
    fig_pie.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        template='plotly_white',
        margin=dict(t=50, b=100, l=20, r=20)
    )
    pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('dashboard/index.html', plot_json=bar_json, pie_json=pie_json,
                           year=year, districts=districts, precincts=precincts,
                           selected_district=district, selected_precinct=precinct)

@dashboard_bp.route('/backtest')
def backtest():
    session = SessionLocal()
    district_filter = request.args.get('district')
    precinct_filter = request.args.get('precinct')
    model_id = request.args.get('model', type=int)

    repo = ElectionRepository(session)
    districts = repo.get_districts(2024)
    precincts = repo.get_precincts(2024, district_filter) if district_filter else []

    models = session.execute(
        select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc())
    ).scalars().all()
    session.close()

    # Add labels for the template dropdown
    for m in models:
        m.label = f"{m.model_type.upper()} ({m.training_date.strftime('%H:%M')})"

    processor = DataProcessor()
    _, test_df = processor.prepare_training_data()

    scatter_json = None
    comp_json = None
    district_data = []
    model_type_label = "None"
    error_msg = None

    if test_df is not None and not test_df.empty:
        result_df, model_type_label, model_id = run_inference_on_2024(test_df, model_id)

        if result_df is not None:
            if district_filter:
                result_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == district_filter]
            if precinct_filter:
                result_df = result_df[result_df['APYLINKES_PAVADINIMAS'] == precinct_filter]

            if not result_df.empty:
                # Optimized aggregation for the visualization logic
                party_agg = result_df.groupby('SARASO_PAVADINIMAS').agg(
                    ACTUAL=('VOTE_SHARE', 'mean'),
                    PREDICTED=('PREDICTED', 'mean')
                ).reset_index().sort_values('ACTUAL', ascending=False)

                if not party_agg.empty:
                    import plotly.express as px
                    import plotly.utils
                    import json

                    fig_scatter = px.scatter(
                        party_agg, x='ACTUAL', y='PREDICTED',
                        hover_name='SARASO_PAVADINIMAS',
                        title=f"Actual vs Predicted Vote Share — {model_type_label.upper()} model",
                        labels={'ACTUAL': 'Actual Vote Share', 'PREDICTED': 'Predicted Vote Share'},
                        color_discrete_sequence=['#6366f1']
                    )
                    max_val = max(party_agg['ACTUAL'].max(), party_agg['PREDICTED'].max())
                    fig_scatter.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                                          line=dict(color='red', dash='dash', width=1))
                    fig_scatter.update_layout(template='plotly_white')
                    scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

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

                    for dist in sorted(result_df['APYGARDOS_PAVADINIMAS'].unique()):
                        dist_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == dist]

                        dist_party = dist_df.groupby('SARASO_PAVADINIMAS').agg(
                            ACTUAL=('VOTE_SHARE', 'mean'),
                            PREDICTED=('PREDICTED', 'mean')
                        ).reset_index().sort_values('ACTUAL', ascending=False)

                        top_actual = dist_party.iloc[0]['SARASO_PAVADINIMAS'] if not dist_party.empty else '-'
                        top_predicted = dist_party.sort_values('PREDICTED', ascending=False).iloc[0]['SARASO_PAVADINIMAS'] if not dist_party.empty else '-'
                        winner_match = top_actual == top_predicted

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
        available_models=models,
        selected_model=model_id,
        districts=districts,
        precincts=precincts,
        district_data=district_data,
        model_type_label=model_type_label,
        selected_district=district_filter,
        selected_precinct=precinct_filter,
        error_msg=error_msg
    )
   
