import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from app.data.repositories import ElectionRepository
from app.data.database import SessionLocal

class DataProcessor:
    def __init__(self):
        self.session = SessionLocal()
        self.repo = ElectionRepository(self.session)

    def fix_lt_encoding(self, df):
        """Universali funkcija koduotės klaidoms taisyti visame DataFrame."""
        def fix_val(val):
            if pd.isna(val): 
                return val
            text = str(val)
            try:
                text = text.encode('latin1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
                
            fixes = {
                'ā€“': '-', 'â€“': '-', '–': '-',
                'Å«': 'ū', 'Åª': 'Ū',
                'Å¡': 'š', 'Å ': 'Š',
                'Å¾': 'ž', 'Å½': 'Ž',
                'Å³': 'ų', 'Å²': 'Ų',
                'Ä—': 'ė', 'Ä–': 'Ė',
                'Ä¯': 'į', 'Ä®': 'Į',
                'Ä…': 'ą', 'Ä„': 'Ą',
                'Ä': 'č', 'ÄŒ': 'Č',
                'Ä™': 'ę', 'Ä˜': 'Ę'
            }
            for bad, good in fixes.items():
                if bad in text:
                    text = text.replace(bad, good)
            return text.replace('SÅ«duvos', 'Sūduvos').replace('KÄ™stuÄ io', 'Kęstučio')

        name_cols = ['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS', 'SARASO_PAVADINIMAS']
        for col in name_cols:
            if col in df.columns:
                df[col] = df[col].apply(fix_val)
        return df

    def prepare_training_data(self, test_year=2024):
        """Prepares a unified dataset for training and testing."""
        years = [2012, 2016, 2020, 2024]
        dfs = []
        for y in years:
            df = self.repo.get_dataframe_for_ml(y)
            if not df.empty:
                df['YEAR'] = y
                dfs.append(df)
        
        if not dfs:
            return None
            
        full_df = pd.concat(dfs, ignore_index=True)

        # 1. KODUOTĖS TAISYMAS (Sutvarko visas raides prieš skaičiavimus)
        full_df = self.fix_lt_encoding(full_df)

        # 2. Partijų pavadinimų normalizavimas
        party_replacements = {
            r'(?i).*Tėvynės sąjunga.*': 'TS-LKD',
            r'(?i).*liberalų sąjūdis.*': 'Liberalų sąjūdis',
            r'(?i).*socialdemokratų partija.*': 'LSDP',
            r'(?i).*valstiečių.*': 'LVŽS',
            r'(?i)^Darbo partija.*': 'Darbo partija',
            r'(?i).*lenkų rinkimų akcija.*': 'LLRA-KŠS',
            r'(?i).*Tvarka ir teisingumas.*': 'Tvarka ir teisingumas',
            r'(?i).*Vardan Lietuvos.*': 'Demokratai Vardan Lietuvos',
            r'(?i).*Laisvės partija.*': 'Laisvės partija',
            r'(?i).*Nemuno aušra.*': 'Nemuno aušra',
            r'(?i).*Regionų partija.*': 'Lietuvos regionų partija'
        }
        
        if 'SARASO_PAVADINIMAS' in full_df.columns:
            full_df['SARASO_PAVADINIMAS'] = full_df['SARASO_PAVADINIMAS'].replace(party_replacements, regex=True)
        
        # Saugus skaičių konvertavimas
        full_df['BALSU_VISO'] = pd.to_numeric(full_df['BALSU_VISO'], errors='coerce').fillna(0)
        full_df['VISO_DALYVAVO'] = pd.to_numeric(full_df['VISO_DALYVAVO'], errors='coerce').fillna(0)
        full_df['RINKEJU_SKAICIUS'] = pd.to_numeric(full_df['RINKEJU_SKAICIUS'], errors='coerce').fillna(0)
        
        full_df = full_df[full_df['VISO_DALYVAVO'] > 0].copy()
        full_df['VOTE_SHARE'] = (full_df['BALSU_VISO'] / full_df['VISO_DALYVAVO']) * 100
        full_df['VOTE_SHARE'] = full_df['VOTE_SHARE'].clip(0, 100)
        
        train_df = full_df[full_df['YEAR'] < 2024].copy()
        test_df = full_df[full_df['YEAR'] == 2024].copy()
        
        train_df = train_df.dropna(subset=['VOTE_SHARE'])
        test_df = test_df.dropna(subset=['VOTE_SHARE'])
        
        return train_df, test_df

    def prepare_2028_template(self):
        """Creates a prediction template for 2028 using 2024 as the feature base."""
        df_2024 = self.repo.get_dataframe_for_ml(2024)
        if df_2024.empty:
            return None
            
        # 3. KODUOTĖS TAISYMAS 2028 METŲ ŠABLONUI
        df_2024 = self.fix_lt_encoding(df_2024)
        
        template = df_2024[['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS', 'RINKEJU_SKAICIUS', 'VISO_DALYVAVO']].drop_duplicates()
        return template

    def fit_and_save_preprocessor(self, train_df, save_path='app/ml/models/preprocessor.joblib'):
        """Sukuria, apmoko ir išsaugo produkcinį scikit-learn transformatorių."""
        cat_features = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
        num_features = ['RINKEJU_SKAICIUS']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
            ],
            remainder='drop'
        )
        
        preprocessor.fit(train_df)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(preprocessor, save_path)
        
        return preprocessor

    def encode_features_prod(self, df, preprocessor_path='app/ml/models/preprocessor.joblib'):
        """Transforms data using the saved preprocessor for inference."""
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Transformatorius nerastas: {preprocessor_path}. Pirmiausia apmokykite modelį.")
            
        preprocessor = joblib.load(preprocessor_path)
        encoded_data = preprocessor.transform(df)
        return encoded_data

    def close(self):
        """Explicitly close the database session."""
        if hasattr(self, 'session'):
            self.session.close()

    def __del__(self):
        try:
            self.session.close()
        except:
            pass