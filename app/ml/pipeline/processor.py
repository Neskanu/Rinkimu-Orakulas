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
        
        # Calculate Vote Share
        full_df['VOTE_SHARE'] = full_df['BALSU_VISO'] / full_df['VISO_DALYVAVO'].replace(0, 1)
        
        # Create a unique ID for precincts: District_NR + Precinct_NR
        full_df['PRECINCT_ID'] = full_df['APYGARDOS_NR'].astype(str) + "_" + full_df['APYLINKES_NR'].astype(str)
        
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
        """One-hot encodes categorical features."""
        df = df.copy()
        categorical_cols = ['SARASO_PAVADINIMAS', 'APYGARDOS_PAVADINIMAS']
        encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return encoded_df

    def __del__(self):
        self.session.close()
