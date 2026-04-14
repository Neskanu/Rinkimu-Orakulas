import joblib
from flask import Blueprint, render_template, request, jsonify
from app.data.database import SessionLocal
from app.data.repositories import ElectionRepository
from app.data.models import MLModelRegistry, EnsembleConfig
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
# NAUJA: In-memory kešas (atmintis) greitam krovimui
TEST_DF_CACHE = None
INFERENCE_CACHE = {}

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
            # Naudojame mūsų Wrapper klasę iš naujosios architektūros
            from app.ml.pipeline.models import CatBoostModel
            m = CatBoostModel()
            m.load_model(path)
            return m, 'catboost', model_entry.id
        else:
            # Naudojame joblib visiems kitiems scikit-learn/xgboost modeliams
            m = joblib.load(path)
            return m, model_type, model_entry.id
    except Exception as e:
        print(f"KLAIDA KRAUNANT MODELĮ: {e}") # Pridedame klaidų spausdinimą diagnozei
        return None, None, None

def run_inference_on_2024(test_df, model_id=None):
    """Run inference using the specified model and ColumnTransformer."""
    model, model_type, actual_id = load_model_by_id(model_id)
    
    if model is None:
        return None, "No valid model found", None

    cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
    num_features = ['RINKEJU_SKAICIUS']
    features = cat_features + num_features
    
    for col in num_features:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').replace({np.nan: 0})

    for col in cat_features:
        test_df[col] = test_df[col].astype(str).replace('nan', 'Unknown')
    
    X = test_df[features].copy()

    try:
        if model_type == 'catboost':
            X[cat_features] = X[cat_features].astype(str)
            preds = model.predict(X)
        else:
            # NAUJA LOGIKA: Naudojame išsaugotą transformatorių vietoj pd.get_dummies
            preprocessor_path = 'app/ml/models/preprocessor.joblib'
            if not os.path.exists(preprocessor_path):
                return None, "Nerastas preprocessor.joblib failas", None
                
            preprocessor = joblib.load(preprocessor_path)
            X_encoded = preprocessor.transform(X)
            
            # Mūsų Wrapper klasių .predict() metodai dabar gaus teisingo formato matricą
            preds = model.predict(X_encoded)

        result_df = test_df[['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS',
                             'SARASO_PAVADINIMAS', 'VOTE_SHARE']].copy()
        
        result_df['PREDICTED'] = np.clip(preds, 0, 100) 
        return result_df, model_type, actual_id
        
    except Exception as e:
        print(f"INFERENCE KLAIDA: {e}")
        import traceback
        traceback.print_exc()
        return None, "Prediction failed", None


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

    # --- Unified Color Map & Truncation ---
    premium_palette = px.colors.qualitative.Prism
    def get_party_color_map(parties):
        return {party: premium_palette[i % len(premium_palette)] for i, party in enumerate(parties)}

    # Truncate names to 50 chars as requested
    agg_df['DISPLAY_NAME_FULL'] = agg_df['SARASO_PAVADINIMAS'].apply(
        lambda x: (x[:47] + '...') if len(x) > 50 else x)

    chart_df = agg_df.sort_values(by='SHARE_PCT', ascending=False).head(15).copy()
    color_map = get_party_color_map(chart_df['DISPLAY_NAME_FULL'].unique())

    # --- Bar Chart ---
    top_10 = chart_df.head(10).copy().iloc[::-1]
    
    fig_bar = px.bar(top_10, x='SHARE_PCT', y='DISPLAY_NAME_FULL', orientation='h',
                     title=f"Actual Results: Top 10 Parties ({year})",
                     labels={'SHARE_PCT': 'Vote Share (%)', 'DISPLAY_NAME_FULL': 'Party'},
                     color='DISPLAY_NAME_FULL', 
                     color_discrete_map=color_map,
                     text=top_10['SHARE_PCT'].apply(lambda x: f"{x:.1f}%"),
                     custom_data=['SARASO_PAVADINIMAS', vote_col])

    fig_bar.update_traces(
        textposition='outside',
        marker_line_width=0,
        hovertemplate='<b>%{customdata[0]}</b><br>Votes: %{customdata[1]:,.0f}<br>Share: %{x:.2f}%<extra></extra>'
    )

    max_share = top_10['SHARE_PCT'].max()
    fig_bar.update_layout(
        template='plotly_white',
        margin=dict(l=10, r=40, t=50, b=20),
        yaxis={'categoryorder': 'total ascending', 'showgrid': False, 'showticklabels': True, 'title': '', 'side': 'right'},
        xaxis=dict(range=[0, max_share * 1.25], showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        showlegend=False,
        font=dict(family="Outfit, sans-serif")
    )
    bar_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # --- Optimized Pie Chart ---
    pie_data = []
    other_votes = 0.0
    other_share = 0.0
    
    top_party_names = chart_df['DISPLAY_NAME_FULL'].tolist()
    
    for _, row in agg_df.iterrows():
        name = row['DISPLAY_NAME_FULL']
        if name in top_party_names:
            pie_data.append({
                'Party': name,
                'Votes': float(row[vote_col]),
                'Share': float(row['SHARE_PCT'])
            })
        else:
            other_votes += float(row[vote_col])
            other_share += float(row['SHARE_PCT'])

    if other_share > 0:
        pie_data.append({
            'Party': 'Other Parties',
            'Votes': float(other_votes),
            'Share': float(other_share)
        })
        color_map['Other Parties'] = '#cbd5e1'

    clean_pie_df = pd.DataFrame(pie_data).sort_values('Share', ascending=False)
    total_votes_int = int(clean_pie_df['Votes'].sum())

    fig_pie = px.pie(
        clean_pie_df,
        values='Share',
        names='Party',
        title="Vote Distribution",
        hole=0.45,
        color='Party',
        color_discrete_map=color_map,
        custom_data=['Votes']
    )

    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent',
        textfont=dict(size=12, color='white', family="Outfit"),
        hovertemplate='<b>%{label}</b><br>Votes: %{customdata[0]:,.0f}<br>Share: %{value:.2f}%<extra></extra>',
        marker=dict(line=dict(color='white', width=2))
    )

    fig_pie.update_layout(
        annotations=[
            dict(
                text=f"<span style='font-size:12px;color:#64748b'>TOTAL</span><br><b style='font-size:16px'>{total_votes_int:,}</b>",
                x=0.5, y=0.5, showarrow=False, xanchor="center", yanchor="middle"
            )
        ],
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=11, family="Outfit"),
            itemclick=False,
            itemdoubleclick=False
        ),
        template='plotly_white',
        # Push domains to the LEFT to ensure pie is on the left and legend has room on the right
        grid=dict(rows=1, columns=1),
        margin=dict(t=50, b=20, l=10, r=150),
        title=dict(font=dict(size=18, family="Outfit"), x=0.05),
        font=dict(family="Outfit, sans-serif")
    )

    pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('dashboard/index.html', plot_json=bar_json, pie_json=pie_json,
                           year=year, districts=districts, precincts=precincts,
                           selected_district=district, selected_precinct=precinct)

@dashboard_bp.route('/backtest')
def backtest():
    district_filter = request.args.get('district')
    precinct_filter = request.args.get('precinct')
    
    # 1. SAUGUS MODELIO ID GAVIMAS
    raw_model_id = request.args.get('model')
    model_id = int(raw_model_id) if raw_model_id and raw_model_id.isdigit() else None

    # Inicializuojame kintamuosius
    scatter_json = None
    comp_json = None
    district_data = []
    model_type_label = "None"
    error_msg = None

    global TEST_DF_CACHE, INFERENCE_CACHE

    # 2. RANDAME GERIAUSIĄ MODELĮ
    if not model_id:
        with SessionLocal() as session:
            best_model = session.execute(
                select(MLModelRegistry).order_by(MLModelRegistry.mae.asc())
            ).scalars().first()
            if best_model:
                model_id = best_model.id
    
    cache_key = str(model_id) if model_id else "unknown"

    # 3. DUOMENŲ PARUOŠIMAS 
    if TEST_DF_CACHE is None:
        processor = DataProcessor()
        prep_res = processor.prepare_training_data()
        TEST_DF_CACHE = prep_res[1] if prep_res else None

    test_df = TEST_DF_CACHE
    base_result_df = None

    if test_df is not None and not test_df.empty:
        # 4. NAUJA KEŠO STRUKTŪRA (Saugome ne tik DataFrame, bet ir paruoštą HTML informaciją)
        if cache_key in INFERENCE_CACHE:
            cached = INFERENCE_CACHE[cache_key]
            base_result_df = cached['df'].copy()
            model_type_label = cached['label']
            model_id = cached['id']
        else:
            base_result_df, model_type_label, actual_id = run_inference_on_2024(test_df, model_id)
            model_id = actual_id
            if base_result_df is not None:
                INFERENCE_CACHE[cache_key] = {
                    'df': base_result_df.copy(),
                    'label': model_type_label,
                    'id': actual_id,
                    'national_data': None,
                    'scatter_json': None,
                    'comp_json': None
                }

        if base_result_df is not None:
            result_df = base_result_df.copy()
            
            # Tikriname, ar pasirinkta visa Lietuva
            is_national = not district_filter and not precinct_filter
            
            if district_filter:
                result_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == district_filter]
            if precinct_filter:
                result_df = result_df[result_df['APYLINKES_PAVADINIMAS'] == precinct_filter]

            if not result_df.empty:
                # SUPER GREITAS UŽKROVIMAS: Jei tai visa Lietuva ir jau skaičiavome - imam iš atminties!
                if is_national and INFERENCE_CACHE[cache_key].get('national_data') is not None:
                    district_data = INFERENCE_CACHE[cache_key]['national_data']
                    scatter_json = INFERENCE_CACHE[cache_key]['scatter_json']
                    comp_json = INFERENCE_CACHE[cache_key]['comp_json']
                else:
                    # Skaičiuojame iš naujo (Tai įvyks tik 1 kartą arba kai pasirenkamas konkretus miestas)
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
                        fig_scatter.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color='red', dash='dash', width=1))
                        fig_scatter.update_layout(template='plotly_white')
                        scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

                        party_agg['SHORT_NAME'] = party_agg['SARASO_PAVADINIMAS'].apply(lambda x: x[:20] + '...' if len(str(x)) > 20 else x)
                        comp_df = party_agg.head(12).melt(id_vars=['SARASO_PAVADINIMAS', 'SHORT_NAME'], value_vars=['ACTUAL', 'PREDICTED'])
                        fig_bar = px.bar(
                            comp_df, x='value', y='SHORT_NAME', color='variable',
                            hover_name='SARASO_PAVADINIMAS',
                            barmode='group', orientation='h', title="Top 12 partijų palyginimas",
                            template='plotly_white', color_discrete_map={'ACTUAL': '#6366f1', 'PREDICTED': '#f59e0b'}
                        )
                        fig_bar.update_layout(
                            yaxis={'categoryorder':'total ascending', 'title': '', 'tickmode': 'linear', 'side': 'right'}, 
                            xaxis={'title': 'Balsų dalis (%)'},
                            legend=dict(title="", orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                            margin=dict(l=10, r=150, t=50, b=0) 
                        )
                        comp_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

                    # SUNKUSIS CIKLAS (1900 Apylinkių)
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
                            ).reset_index().sort_values('ACTUAL', ascending=False)
                            
                            top_act_prec = prec_parties.iloc[0]['SARASO_PAVADINIMAS'] if not prec_parties.empty else '-'
                            top_prd_prec = prec_parties.sort_values('PREDICTED', ascending=False).iloc[0]['SARASO_PAVADINIMAS'] if not prec_parties.empty else '-'
                            winner_match_prec = top_act_prec == top_prd_prec

                            precincts_data.append({
                                'name': prec,
                                'top_actual': top_act_prec,
                                'top_predicted': top_prd_prec,
                                'winner_match': winner_match_prec,
                                'parties': prec_parties.head(10).to_dict('records')
                            })

                        district_data.append({
                            'name': dist,
                            'top_actual': top_actual[:40],
                            'top_predicted': top_predicted[:40],
                            'winner_match': winner_match,
                            'parties': dist_party.head(8).to_dict('records'),
                            'precincts': precincts_data
                        })

                    # 5. IŠSAUGOME SUNKAUS CIKLO REZULTATUS KEŠE (Jei tai visos Lietuvos filtras)
                    if is_national:
                        INFERENCE_CACHE[cache_key]['national_data'] = district_data
                        INFERENCE_CACHE[cache_key]['scatter_json'] = scatter_json
                        INFERENCE_CACHE[cache_key]['comp_json'] = comp_json
        else:
            error_msg = "Model inference failed. Please train a model first."
    else:
        error_msg = "No 2024 test data available."

    # 6. Traukiame DB informaciją TIK PRIEŠ PAT renderinant HTML
    
    with SessionLocal() as session:
        repo = ElectionRepository(session)
        districts = repo.get_districts(2024)
        precincts = repo.get_precincts(2024, district_filter) if district_filter else []

        models = session.execute(
            select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc())
        ).scalars().all()
        
        # PRIDĖTA: Ansamblio konfigūracijos užkrovimas
        config = session.query(EnsembleConfig).first()

        for m in models:
            m.label = f"{m.model_type.upper()} ({m.training_date.strftime('%H:%M')})"

        # Šablonas generuojamas kol sesija atidaryta
        return render_template(
            'dashboard/backtest.html',
            scatter_json=scatter_json,
            comp_json=comp_json,
            available_models=models,
            config=config,             # PRIDĖTA: Perduodame ansamblį į naršyklę
            selected_model=model_id,
            districts=districts,
            precincts=precincts,
            district_data=district_data,
            model_type_label=model_type_label,
            selected_district=district_filter,
            selected_precinct=precinct_filter,
            error_msg=error_msg
        )