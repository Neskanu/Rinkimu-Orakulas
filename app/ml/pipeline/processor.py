import pandas as pd
import numpy as np
from app.data.repositories import ElectionRepository
from app.data.database import SessionLocal

class DataProcessor:
    def __init__(self):
        self.session = SessionLocal()
        self.repo = ElectionRepository(self.session)

    def prepare_training_data(self, test_year=2024):
        """Prepares a unified dataset for training and testing."""
        years = [2012, 2016, 2020, 2024]
        dfs = []
        for y in years:
            df = self.repo.get_dataframe_for_ml(y)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return None
            
        full_df = pd.concat(dfs, ignore_index=True)

        # Clean encoding artifacts in names
        name_cols = ['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS', 'SARASO_PAVADINIMAS']
        for col in name_cols:
            if col in full_df.columns:
                full_df[col] = full_df[col].astype(str).str.replace('ā€“', '-', regex=False)
                full_df[col] = full_df[col].str.replace('SÅ«duvos', 'Sūduvos', regex=False)
        
        # Force numeric types before math to prevent 'garbage' training targets
        full_df['BALSU_VISO'] = pd.to_numeric(full_df['BALSU_VISO'], errors='coerce').fillna(0)
        full_df['VISO_DALYVAVO'] = pd.to_numeric(full_df['VISO_DALYVAVO'], errors='coerce').fillna(0)
        full_df['RINKEJU_SKAICIUS'] = pd.to_numeric(full_df['RINKEJU_SKAICIUS'], errors='coerce').fillna(0)
        
        # Drop rows where VISO_DALYVAVO is 0 — we cannot calculate a meaningful
        # vote share without a valid total-participants count. This removes the
        # entire 2012 dataset (which has all-zero VISO_DALYVAVO) and any broken
        # precinct rows in other years.
        full_df = full_df[full_df['VISO_DALYVAVO'] > 0].copy()
        
        # Calculate Vote Share (0-100 scale) — safe now that denominator > 0
        full_df['VOTE_SHARE'] = (full_df['BALSU_VISO'] / full_df['VISO_DALYVAVO']) * 100
        
        # Safety clamp: vote share must be 0-100%
        full_df['VOTE_SHARE'] = full_df['VOTE_SHARE'].clip(0, 100)
        
        # Train-Test Split: Strictly enforce 2024 as hold-out
        train_df = full_df[full_df['YEAR'] < 2024].copy()
        test_df = full_df[full_df['YEAR'] == 2024].copy()
        
        # Drop rows where target is NaN
        train_df = train_df.dropna(subset=['VOTE_SHARE'])
        test_df = test_df.dropna(subset=['VOTE_SHARE'])
        
        return train_df, test_df

    def prepare_2028_template(self):
        """Creates a prediction template for 2028 using 2024 as the feature base."""
        df_2024 = self.repo.get_dataframe_for_ml(2024)
        if df_2024.empty:
            return None
            
        template = df_2024[['APYGARDOS_NR', 'APYGARDOS_PAVADINIMAS', 'APYLINKES_NR', 'APYLINKES_PAVADINIMAS', 'RINKEJU_SKAICIUS', 'VISO_DALYVAVO']].drop_duplicates()
        return template

    def encode_features(self, df):
        """One-hot encodes categorical features using sparse matrices for efficiency."""
        df = df.copy()
        categorical_cols = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
        
        # Convert to category type first (speeds up get_dummies)
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                
        encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
        return encoded_df

    def close(self):
        """Explicitly close the database session."""
        if hasattr(self, 'session'):
            self.session.close()

    def __del__(self):
        # Fallback closure (non-blocking)
        try:
            self.session.close()
        except:
            pass
