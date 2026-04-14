import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.utils
import joblib
from flask import Blueprint, render_template, request, jsonify
from sqlalchemy import select

from app.data.database import SessionLocal
from app.data.models import MLModelRegistry, EnsembleConfig
from app.data.repositories import ElectionRepository
from app.ml.pipeline.processor import DataProcessor

dashboard_bp = Blueprint('dashboard', __name__)

def load_model_by_id(model_id=None):
    """Saugiai užkrauna modelį iš registro pagal ID arba geriausią MAE."""
    session = SessionLocal()
    try:
        if model_id:
            model_entry = session.get(MLModelRegistry, model_id)
        else:
            # Paimame naujausią modelį su mažiausia paklaida
            model_entry = session.execute(
                select(MLModelRegistry).order_by(MLModelRegistry.mae.asc())
            ).scalars().first()
        
        if not model_entry or not os.path.exists(model_entry.file_path):
            return None, None, None
            
        path = model_entry.file_path
        model_type = model_entry.model_type
        
        if model_type == 'catboost':
            from app.ml.pipeline.models import CatBoostModel
            m = CatBoostModel()
            m.load_model(path)
        else:
            m = joblib.load(path)
            
        return m, model_type, model_entry.id
    except Exception as e:
        print(f"Klaida kraunant modelį: {e}")
        return None, None, None
    finally:
        session.close()

def run_inference_on_2024(test_df, model_id=None):
    """Vykdo prognozę su ColumnTransformer ir taiko politinio ciklo koeficientus."""
    model, model_type, actual_id = load_model_by_id(model_id)
    if model is None:
        return None, "No valid model found", None

    X = test_df.copy()
    
    # 1. Partijų pavadinimų standartizavimas (kad modelis atpažintų mokymosi metu matytus vardus)
    party_replacements = {
        r'(?i).*Tėvynės sąjunga.*': 'TS-LKD',
        r'(?i).*liberalų sąjūdis.*': 'Liberalų sąjūdis',
        r'(?i).*socialdemokratų partija.*': 'LSDP',
        r'(?i).*valstiečių.*': 'LVŽS',
        r'(?i)^Darbo partija.*': 'Darbo partija',
        r'(?i).*lenkų rinkimų akcija.*': 'LLRA-KŠS',
        r'(?i).*Vardan Lietuvos.*': 'Demokratai Vardan Lietuvos',
        r'(?i).*Nemuno aušra.*': 'Nemuno aušra'
    }
    X['SARASO_PAVADINIMAS'] = X['SARASO_PAVADINIMAS'].replace(party_replacements, regex=True)

    cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
    num_features = ['RINKEJU_SKAICIUS']
    
    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    for col in cat_features:
        X[col] = X[col].astype(str).replace('nan', 'Unknown')

    X_features = X[cat_features + num_features]

    # 2. Inferencija
    try:
        if model_type == 'catboost':
            raw_preds = model.predict(X_features)
        else:
            preprocessor_path = 'app/ml/models/preprocessor.joblib'
            if not os.path.exists(preprocessor_path):
                return None, "Preprocessor not found", None
            preprocessor = joblib.load(preprocessor_path)
            X_encoded = preprocessor.transform(X_features)
            raw_preds = model.predict(X_encoded)
            
        # 3. LIETUVIŠKO CIKLO LOGIKA (Švytuoklė)
        # LSDP +15%, TS-LKD -10%
        multipliers = X['SARASO_PAVADINIMAS'].apply(
            lambda x: 1.15 if x == 'LSDP' else (0.90 if x == 'TS-LKD' else 1.0)
        )
        
        result_df = test_df[['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS', 
                             'SARASO_PAVADINIMAS', 'VOTE_SHARE']].copy()
        result_df['PREDICTED'] = np.clip(raw_preds * multipliers, 0, 100)
        
        return result_df, model_type, actual_id
    except Exception as e:
        print(f"Inference error: {e}")
        return None, f"Prediction error: {str(e)}", None

@dashboard_bp.route('/')
def index():
    """Pagrindinis Dashboard puslapis."""
    # Nuskaitome filtrus iš URL
    year = request.args.get('year', default=2024, type=int)
    district_filter = request.args.get('district')
    precinct_filter = request.args.get('precinct')
    
    plot_json = None
    pie_json = None
    districts = []
    precincts = []

    with SessionLocal() as session:
        repo = ElectionRepository(session)
        
        # Ištraukiame apygardas ir apylinkes pasirinktiems metams
        districts = repo.get_districts(year)
        if district_filter:
            precincts = repo.get_precincts(year, district_filter)

        # Ištraukiame balsavimo rezultatus
        results_df = repo.get_national_results(year)
        
        if results_df is not None and not results_df.empty:
            # Pritaikius filtrus
            if district_filter:
                results_df = results_df[results_df['APYGARDOS_PAVADINIMAS'] == district_filter]
            if precinct_filter:
                results_df = results_df[results_df['APYLINKES_PAVADINIMAS'] == precinct_filter]

            # Agreguojame duomenis grafikams
            party_results = results_df.groupby('SARASO_PAVADINIMAS')['VOTE_SHARE'].mean().reset_index()
            party_results = party_results.sort_values('VOTE_SHARE', ascending=False)
            # Sukuriame naują stulpelį su trumpu pavadinimu
            party_results['SHORT_NAME'] = party_results['SARASO_PAVADINIMAS'].apply(
                lambda x: x[:20] + '...' if len(str(x)) > 20 else x
            )
            
            # Generuojame Bar Chart
            # Naudojame prekės ženklo spalvas (tamsiai purpurinė perėjanti į šviesią)
            brand_colors = ['#2E2573', '#453A95', '#5E51B8', '#7B6FDB', '#9B90FA', '#BDB5FE', '#DCD8FF', '#94A3B8']

            if not party_results.empty:
                # 1. Bar Chart (Stulpelinė diagrama)
                fig_bar = px.bar(
                    party_results.head(15), 
                    x='VOTE_SHARE', 
                    y='SHORT_NAME',
                    orientation='h',
                    title=f"{year} m. Rezultatai (Top 15)",
                    template='plotly_white',
                    color='SARASO_PAVADINIMAS', # Spalviname pagal partiją, o ne pagal skaičių
                    color_discrete_sequence=brand_colors # Naudojame jūsų paletę
                )
                
                # Bar Chart dizaino tobulinimas
                fig_bar.update_layout(
                    yaxis={'categoryorder':'total ascending', 'title': ''}, # Paslepiame Y ašies pavadinimą
                    xaxis={'title': 'Balsų dalis (%)'},
                    showlegend=False, # IŠJUNGIAME legendą, nes partijų pavadinimai jau yra ant Y ašies
                    margin=dict(l=0, r=20, t=50, b=0), # Sumažiname tuščius kraštus
                    font=dict(color='#475569', size=10) # Teksto spalva (Tailwind slate-600)
                )
                plot_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

                # 2. Pie Chart (Skritulinė diagrama)
                fig_pie = px.pie(
                    party_results.head(8), 
                    values='VOTE_SHARE', 
                    names='SARASO_PAVADINIMAS',
                    title=f"{year} m. Jėgų Pasiskirstymas (Top 8)",
                    hole=0.4, # Padaro "Donut" stiliaus grafiką (moderniau)
                    template='plotly_white',
                    color_discrete_sequence=brand_colors
                )
                
                # Pie Chart dizaino tobulinimas
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent', # Viduje rodome tik procentus
                    insidetextorientation='horizontal'
                )
                fig_pie.update_layout(
                    margin=dict(l=0, r=0, t=50, b=0),
                    font=dict(color='#475569'),
                    # Sumažiname ir perkeliame legendą į apačią
                    legend=dict(
                        orientation="h", # Horizontali legenda
                        yanchor="top",
                        y=-0.1, # Nuleidžiame žemiau grafiko
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10), # Sumažintas šriftas
                        title="" # Paslepiame legendos antraštę
                    )
                )
                pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        'dashboard/index.html',
        year=year,
        districts=districts,
        precincts=precincts,
        selected_district=district_filter,
        selected_precinct=precinct_filter,
        plot_json=plot_json,
        pie_json=pie_json
    )

@dashboard_bp.route('/backtest')
def backtest():
    """Atsigręžtinio testo (2024 m.) vizualizacija."""
    # Inicializuojame kintamuosius, kad išvengtume NameError
    scatter_json = None
    comp_json = None
    district_data = []
    model_type_label = "None"
    error_msg = None
    
    district_filter = request.args.get('district')
    precinct_filter = request.args.get('precinct')
    model_id = request.args.get('model')

    with SessionLocal() as session:
        # Traukiame filtrų duomenis
        repo = ElectionRepository(session)
        districts = repo.get_districts(2024)
        precincts = repo.get_precincts(2024, district_filter) if district_filter else []

        # Traukiame modelius ir ansamblio konfigūraciją
        models = session.execute(
            select(MLModelRegistry).order_by(MLModelRegistry.training_date.desc())
        ).scalars().all()
    
        # Ši eilutė KRITINĖ:
        config = session.query(EnsembleConfig).first()

        for m in models:
            m.label = f"{m.model_type.upper()} ({m.training_date.strftime('%H:%M')})"

        config = session.query(EnsembleConfig).first()
        
        # Ruošiame duomenis prognozei
        processor = DataProcessor()
        # Šis metodas gražina (train_df, test_df). Mums reikia test_df (2024 m.)
        prep_res = processor.prepare_training_data()
        test_df = prep_res[1] if prep_res else None

        if test_df is not None and not test_df.empty:
            result_df, model_type_label, actual_id = run_inference_on_2024(test_df, model_id)
            model_id = actual_id 

            if result_df is not None:
                # Filtruojame rezultatus pagal vartotojo pasirinkimą
                if district_filter:
                    result_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == district_filter]
                if precinct_filter:
                    result_df = result_df[result_df['APYLINKES_PAVADINIMAS'] == precinct_filter]

                # Agreguojame grafikams
                party_agg = result_df.groupby('SARASO_PAVADINIMAS').agg(
                    ACTUAL=('VOTE_SHARE', 'mean'),
                    PREDICTED=('PREDICTED', 'mean')
                ).reset_index().sort_values('ACTUAL', ascending=False)

                if not party_agg.empty:
                    # 1. Scatter Chart
                    fig_scatter = px.scatter(
                        party_agg, x='ACTUAL', y='PREDICTED', hover_name='SARASO_PAVADINIMAS',
                        title=f"Actual vs Predicted — {model_type_label.upper()}",
                        labels={'ACTUAL': 'Tikra dalis %', 'PREDICTED': 'Prognozuota %'},
                        template='plotly_white'
                    )
                    max_v = max(party_agg['ACTUAL'].max(), party_agg['PREDICTED'].max())
                    fig_scatter.add_shape(type='line', x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color='red', dash='dash'))
                    scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

                    # 2. Bar Comparison
                    comp_df = party_agg.head(12).melt(id_vars='SARASO_PAVADINIMAS', value_vars=['ACTUAL', 'PREDICTED'])
                    fig_bar = px.bar(
                        comp_df, x='value', y='SARASO_PAVADINIMAS', color='variable',
                        barmode='group', orientation='h', title="Top 12 partijų palyginimas",
                        template='plotly_white', color_discrete_map={'ACTUAL': '#6366f1', 'PREDICTED': '#f59e0b'}
                    )
                    comp_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

                # 3. Ruošiame duomenis lentelėms
                for dist in sorted(result_df['APYGARDOS_PAVADINIMAS'].unique()):
                    dist_df = result_df[result_df['APYGARDOS_PAVADINIMAS'] == dist]

                    # Sugrupuojame rezultatus apygardoje
                    dist_party = dist_df.groupby('SARASO_PAVADINIMAS').agg(
                        ACTUAL=('VOTE_SHARE', 'mean'),
                        PREDICTED=('PREDICTED', 'mean')
                    ).reset_index()

                    if not dist_party.empty:
                        # 1. Randame tikrąjį nugalėtoją
                        top_actual_row = dist_party.sort_values('ACTUAL', ascending=False).iloc[0]
                        top_actual = str(top_actual_row['SARASO_PAVADINIMAS']).strip()
                        
                        # 2. Randame prognozuotą nugalėtoją
                        top_predicted_row = dist_party.sort_values('PREDICTED', ascending=False).iloc[0]
                        top_predicted = str(top_predicted_row['SARASO_PAVADINIMAS']).strip()
                        
                        # 3. PALYGINIMAS (Pridedame .lower() ir .strip() saugumui)
                        winner_match = top_actual.lower() == top_predicted.lower()
                        
                        # Detalios apylinkės (jei reikia)
                        precincts_data = []
                        # ... jūsų apylinkių ciklas ...

                        district_data.append({
                            'name': dist,
                            'top_actual': top_actual,
                            'top_predicted': top_predicted,
                            'winner_match': winner_match, # Čia dabar bus teisingas True/False
                            'parties': dist_party.sort_values('ACTUAL', ascending=False).head(5).to_dict('records')
                        })
                error_msg = "Nepavyko įvykdyti prognozės."
        else:
            error_msg = "Nerasti 2024 m. testiniai duomenys."

        return render_template(
            'dashboard/backtest.html',
            scatter_json=scatter_json,
            comp_json=comp_json,
            available_models=models,
            config=config,
            selected_model=model_id,
            districts=districts,
            precincts=precincts,
            district_data=district_data,
            model_type_label=model_type_label,
            selected_district=district_filter,
            selected_precinct=precinct_filter,
            error_msg=error_msg
        )