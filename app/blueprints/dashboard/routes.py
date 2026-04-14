import joblib
import json
import os
import plotly
import plotly.express as px
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, jsonify
from sqlalchemy import select

from app.data.database import SessionLocal
from app.data.repositories import ElectionRepository
from app.data.models import MLModelRegistry, EnsembleConfig
from app.ml.pipeline.processor import DataProcessor

# BŪTINA: Importuojame Wrapper klases, kad joblib galėtų atidaryti SVR, NN ir kitus modelius
from app.ml.pipeline.models import (
    TreeModel, XGBModel, LGBMModel, 
    PolyElasticNetModel, CatBoostModel, NNModel,
    DeepNNModel, WideNNModel, SVRModel
)

dashboard_bp = Blueprint('dashboard', __name__)

TEST_DF_CACHE = None
INFERENCE_CACHE = {}

def get_vote_col(df):
    party_vote_columns = [
        'BALSU_SKAICIUS', 'BALSAI_UZ_SARASA', 'PADUOTI_BALSAI', 'GALI_BALSAI', 'BALSU_VISO'
    ]
    for col in party_vote_columns:
        if col in df.columns:
            return col
    return None

def fix_lt_encoding_global(val):
    if pd.isna(val): return val
    text = str(val)
    try:
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    fixes = {
        'ā€“': '-', 'â€“': '-', '–': '-',
        'Å«': 'ū', 'Åª': 'Ū', 'Å¡': 'š', 'Å ': 'Š',
        'Å¾': 'ž', 'Å½': 'Ž', 'Å³': 'ų', 'Å²': 'Ų',
        'Ä—': 'ė', 'Ä–': 'Ė', 'Ä¯': 'į', 'Ä®': 'Į',
        'Ä…': 'ą', 'Ä„': 'Ą', 'Ä': 'č', 'ÄŒ': 'Č',
        'Ä™': 'ę', 'Ä˜': 'Ę'
    }
    for bad, good in fixes.items():
        if bad in text: text = text.replace(bad, good)
    return text.replace('SÅ«duvos', 'Sūduvos').replace('KÄ™stuÄ io', 'Kęstučio')

def load_model_by_id(model_id=None):
    session = SessionLocal()
    if model_id:
        model_entry = session.get(MLModelRegistry, model_id)
    else:
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
            m = CatBoostModel()
            m.load_model(path)
            return m, 'catboost', model_entry.id
        else:
            m = joblib.load(path)
            return m, model_type, model_entry.id
    except Exception as e:
        print(f"KLAIDA KRAUNANT MODELĮ: {e}")
        return None, None, None

def run_inference_on_2024(test_df, model_id=None):
    """Vykdo prognozę naudodamas konkretų modelį arba visą ansamblį."""
    
    # --- ANSAMBLIO LOGIKA ---
    if model_id == 'ensemble':
        try:
            from app.blueprints.forecasting.routes import load_ensemble
            model, active_mids = load_ensemble()
            model_type = "ensemble"
            actual_id = "ensemble"
        except ImportError as e:
            print(f"Importo klaida: {e}")
            return None, "Nepavyko rasti forecasting modulio", None
        
        if not model:
            return None, "Ansamblio failas nerastas", None
    else:
        # Krauname įprastą modelį iš DB registro
        model, model_type, actual_id = load_model_by_id(model_id)
    
    if model is None: 
        return None, "No valid model found", None
    else:
        # Krauname įprastą modelį iš DB registro
        model, model_type, actual_id = load_model_by_id(model_id)
    # --- LOGIKOS PABAIGA ---

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
        elif model_type == 'ensemble':
            preds = model.predict(test_df) # Ansamblis pats viduje naudoja preprocessor.joblib
        else:
            import warnings
            preprocessor_path = 'app/ml/models/preprocessor.joblib'
            if not os.path.exists(preprocessor_path):
                return None, "Nerastas preprocessor.joblib failas", None
                
            preprocessor = joblib.load(preprocessor_path)
            X_encoded = preprocessor.transform(X)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = model.predict(X_encoded)

        result_df = test_df[['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS',
                             'SARASO_PAVADINIMAS', 'VOTE_SHARE']].copy()
        result_df['PREDICTED'] = np.clip(preds, 0, 100) 
        return result_df, model_type, actual_id
        
    except Exception as e:
        print(f"INFERENCE KLAIDA: {e}")
        return None, "Prediction failed", None


@dashboard_bp.route('/')
def index():
    session = SessionLocal()
    year = request.args.get('year', 2024, type=int)
    
    # 1. Paimame RAW filtrus iškviesti duombazei
    raw_district = request.args.get('district')
    raw_precinct = request.args.get('precinct')
    
    # 2. Sukuriame IŠVALYTUS filtrus, kuriuos naudosime filtruoti DataFrame
    district = fix_lt_encoding_global(raw_district) if raw_district else None
    precinct = fix_lt_encoding_global(raw_precinct) if raw_precinct else None

    repo = ElectionRepository(session)
    df = repo.get_dataframe_for_ml(year)
    
    # Išvalome DataFrame
    if not df.empty:
        for col in ['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS', 'SARASO_PAVADINIMAS']:
            if col in df.columns:
                df[col] = df[col].apply(fix_lt_encoding_global)

    # Į Dropdown siunčiame išvalytus tekstus, bet traukiame iš duombazės pagal RAW
    districts = [fix_lt_encoding_global(d) for d in repo.get_districts(year)]
    precincts = [fix_lt_encoding_global(p) for p in (repo.get_precincts(year, raw_district) if raw_district else [])]
    session.close()

    if df.empty:
        return render_template('dashboard/index.html', plot_json=None, pie_json=None,
                               year=year, districts=districts, precincts=precincts)

    vote_col = get_vote_col(df)
    for col in [vote_col, 'VISO_DALYVAVO', 'RINKEJU_SKAICIUS']:
        if col and col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').replace({np.nan: 0})

    # Filtruojame su IŠVALYTAIS filtrais
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

    premium_palette = px.colors.qualitative.Prism
    def get_party_color_map(parties):
        return {party: premium_palette[i % len(premium_palette)] for i, party in enumerate(parties)}

    agg_df['DISPLAY_NAME_FULL'] = agg_df['SARASO_PAVADINIMAS'].apply(lambda x: (x[:47] + '...') if len(x) > 50 else x)
    chart_df = agg_df.sort_values(by='SHARE_PCT', ascending=False).head(15).copy()
    color_map = get_party_color_map(chart_df['DISPLAY_NAME_FULL'].unique())

    top_10 = chart_df.head(10).copy().iloc[::-1]
    fig_bar = px.bar(top_10, x='SHARE_PCT', y='DISPLAY_NAME_FULL', orientation='h',
                     title=f"Top 10 {year} partijų",
                     labels={'SHARE_PCT': 'Balsų dalis (%)', 'DISPLAY_NAME_FULL': 'Partija'},
                     color='DISPLAY_NAME_FULL', color_discrete_map=color_map,
                     text=top_10['SHARE_PCT'].apply(lambda x: f"{x:.1f}%"),
                     custom_data=['SARASO_PAVADINIMAS', vote_col])

    fig_bar.update_traces(textposition='outside', marker_line_width=0,
        hovertemplate='<b>%{customdata[0]}</b><br>Balsai: %{customdata[1]:,.0f}<br>Dalis: %{x:.2f}%<extra></extra>')

    max_share = top_10['SHARE_PCT'].max()
    fig_bar.update_layout(
        template='plotly_white', margin=dict(l=10, r=40, t=50, b=20),
        yaxis={'categoryorder': 'total ascending', 'showgrid': False, 'showticklabels': True, 'title': '', 'side': 'right'},
        xaxis=dict(range=[0, max_share * 1.25], showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        showlegend=False, font=dict(family="Outfit, sans-serif")
    )
    bar_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    pie_data = []
    other_votes, other_share = 0.0, 0.0
    top_party_names = chart_df['DISPLAY_NAME_FULL'].tolist()
    
    for _, row in agg_df.iterrows():
        name = row['DISPLAY_NAME_FULL']
        if name in top_party_names:
            pie_data.append({'Party': name, 'Votes': float(row[vote_col]), 'Share': float(row['SHARE_PCT'])})
        else:
            other_votes += float(row[vote_col])
            other_share += float(row['SHARE_PCT'])

    if other_share > 0:
        pie_data.append({'Party': 'Other Parties', 'Votes': float(other_votes), 'Share': float(other_share)})
        color_map['Other Parties'] = '#cbd5e1'

    clean_pie_df = pd.DataFrame(pie_data).sort_values('Share', ascending=False)
    total_votes_int = int(clean_pie_df['Votes'].sum())

    fig_pie = px.pie(clean_pie_df, values='Share', names='Party', title="Balsų pasiskirstymas",
                     hole=0.45, color='Party', color_discrete_map=color_map, custom_data=['Votes'])

    fig_pie.update_traces(textposition='inside', textinfo='percent', textfont=dict(size=12, color='white', family="Outfit"),
        hovertemplate='<b>%{label}</b><br>Dalis: %{value:.2f}%<extra></extra>',
        marker=dict(line=dict(color='white', width=2)))

    fig_pie.update_layout(
        annotations=[dict(text=f"<span style='font-size:12px;color:#64748b'>TOTAL</span><br><b style='font-size:16px'>{total_votes_int:,}</b>",
                          x=0.5, y=0.5, showarrow=False, xanchor="center", yanchor="middle")],
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, font=dict(size=11, family="Outfit")),
        template='plotly_white', grid=dict(rows=1, columns=1), margin=dict(t=50, b=20, l=10, r=150),
        title=dict(font=dict(size=18, family="Outfit"), x=0.05), font=dict(family="Outfit, sans-serif")
    )
    pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('dashboard/index.html', plot_json=bar_json, pie_json=pie_json,
                           year=year, districts=districts, precincts=precincts,
                           selected_district=district, selected_precinct=precinct)

@dashboard_bp.route('/backtest')
def backtest():
    raw_district = request.args.get('district')
    raw_precinct = request.args.get('precinct')
    
    district_filter = fix_lt_encoding_global(raw_district) if raw_district else None
    precinct_filter = fix_lt_encoding_global(raw_precinct) if raw_precinct else None
    
    raw_model_id = request.args.get('model')
    
    # 1. Nustatome model_id (tekstas arba skaičius)
    if raw_model_id == 'ensemble':
        model_id = 'ensemble'
    else:
        model_id = int(raw_model_id) if raw_model_id and raw_model_id.isdigit() else None

    # 2. Surandame geriausią modelį, jei nieko nepasirinkta
    if not model_id:
        with SessionLocal() as session:
            best_model = session.execute(select(MLModelRegistry).order_by(MLModelRegistry.mae.asc())).scalars().first()
            if best_model: model_id = best_model.id
    
    # 3. Svarbu: cache_key visada paverčiam į string
    cache_key = str(model_id)

    scatter_json, comp_json = None, None
    district_data = []
    model_type_label, error_msg = "None", None

    global TEST_DF_CACHE, INFERENCE_CACHE

    if not model_id:
        with SessionLocal() as session:
            best_model = session.execute(select(MLModelRegistry).order_by(MLModelRegistry.mae.asc())).scalars().first()
            if best_model: model_id = best_model.id
    
    cache_key = str(model_id) if model_id else "unknown"

    if TEST_DF_CACHE is None:
        processor = DataProcessor()
        prep_res = processor.prepare_training_data()
        TEST_DF_CACHE = prep_res[1] if prep_res else None

    test_df = TEST_DF_CACHE
    base_result_df = None

    if test_df is not None and not test_df.empty:
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
                    'df': base_result_df.copy(), 'label': model_type_label, 'id': actual_id,
                    'national_data': None, 'scatter_json': None, 'comp_json': None
                }

        if base_result_df is not None:
            result_df = base_result_df.copy()
            is_national = not district_filter and not precinct_filter
            
            if district_filter: result_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == district_filter]
            if precinct_filter: result_df = result_df[result_df['APYLINKES_PAVADINIMAS'] == precinct_filter]

            if not result_df.empty:
                if is_national and INFERENCE_CACHE[cache_key].get('national_data') is not None:
                    district_data = INFERENCE_CACHE[cache_key]['national_data']
                    scatter_json = INFERENCE_CACHE[cache_key]['scatter_json']
                    comp_json = INFERENCE_CACHE[cache_key]['comp_json']
                else:
                    party_agg = result_df.groupby('SARASO_PAVADINIMAS').agg(
                        ACTUAL=('VOTE_SHARE', 'mean'), PREDICTED=('PREDICTED', 'mean')
                    ).reset_index().sort_values('ACTUAL', ascending=False)

                    if not party_agg.empty:
                        fig_scatter = px.scatter(
                            party_agg, x='ACTUAL', y='PREDICTED', hover_name='SARASO_PAVADINIMAS',
                            title=f"Tikros vs Prognozuotos balsų dalys — {model_type_label.upper()} model",
                            labels={'ACTUAL': 'Tikra balsų dalis', 'PREDICTED': 'Prognozuota balsų dalis'},
                            color_discrete_sequence=['#6366f1']
                        )
                        max_val = max(party_agg['ACTUAL'].max(), party_agg['PREDICTED'].max())
                        fig_scatter.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color='red', dash='dash', width=1))
                        fig_scatter.update_layout(template='plotly_white')
                        scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

                        party_agg['SHORT_NAME'] = party_agg['SARASO_PAVADINIMAS'].apply(lambda x: x[:20] + '...' if len(str(x)) > 20 else x)
                        comp_df = party_agg.head(12).melt(id_vars=['SARASO_PAVADINIMAS', 'SHORT_NAME'], value_vars=['ACTUAL', 'PREDICTED'])
                        fig_bar = px.bar(
                            comp_df, x='value', y='SHORT_NAME', color='variable', hover_name='SARASO_PAVADINIMAS',
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

                    for dist in sorted(result_df['APYGARDOS_PAVADINIMAS'].unique()):
                        dist_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == dist]
                        dist_party = dist_df.groupby('SARASO_PAVADINIMAS').agg(
                            ACTUAL=('VOTE_SHARE', 'mean'), PREDICTED=('PREDICTED', 'mean')
                        ).reset_index().sort_values('ACTUAL', ascending=False)

                        top_actual = dist_party.iloc[0]['SARASO_PAVADINIMAS'] if not dist_party.empty else '-'
                        top_predicted = dist_party.sort_values('PREDICTED', ascending=False).iloc[0]['SARASO_PAVADINIMAS'] if not dist_party.empty else '-'
                        
                        precincts_data = []
                        for prec in sorted(dist_df['APYLINKES_PAVADINIMAS'].unique()):
                            prec_df = dist_df[dist_df['APYLINKES_PAVADINIMAS'] == prec]
                            prec_parties = prec_df.groupby('SARASO_PAVADINIMAS').agg(
                                ACTUAL=('VOTE_SHARE', 'mean'), PREDICTED=('PREDICTED', 'mean')
                            ).reset_index().sort_values('ACTUAL', ascending=False)
                            
                            top_act_prec = prec_parties.iloc[0]['SARASO_PAVADINIMAS'] if not prec_parties.empty else '-'
                            top_prd_prec = prec_parties.sort_values('PREDICTED', ascending=False).iloc[0]['SARASO_PAVADINIMAS'] if not prec_parties.empty else '-'
                            
                            precincts_data.append({
                                'name': prec, 'top_actual': top_act_prec, 'top_predicted': top_prd_prec,
                                'winner_match': top_act_prec == top_prd_prec, 'parties': prec_parties.head(10).to_dict('records')
                            })

                        district_data.append({
                            'name': dist, 'top_actual': top_actual[:40], 'top_predicted': top_predicted[:40],
                            'winner_match': top_actual == top_predicted, 'parties': dist_party.head(8).to_dict('records'),
                            'precincts': precincts_data
                        })

                    if is_national:
                        INFERENCE_CACHE[cache_key]['national_data'] = district_data
                        INFERENCE_CACHE[cache_key]['scatter_json'] = scatter_json
                        INFERENCE_CACHE[cache_key]['comp_json'] = comp_json
        else:
            error_msg = "Model inference failed. Please train a model first."
    else:
        error_msg = "No 2024 test data available."

    with SessionLocal() as session:
        repo = ElectionRepository(session)
        districts = [fix_lt_encoding_global(d) for d in repo.get_districts(2024)]
        precincts = [fix_lt_encoding_global(p) for p in (repo.get_precincts(2024, raw_district) if raw_district else [])]

        models = session.execute(select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc())).scalars().all()
        config = session.query(EnsembleConfig).first()

        for m in models:
            m.label = f"{m.model_type.upper()} ({m.training_date.strftime('%H:%M')})"

        return render_template(
            'dashboard/backtest.html',
            scatter_json=scatter_json, comp_json=comp_json, available_models=models,
            config=config, selected_model=model_id, districts=districts, precincts=precincts,
            district_data=district_data, model_type_label=model_type_label,
            selected_district=district_filter, selected_precinct=precinct_filter, error_msg=error_msg
        )