from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.data.models import *
import pandas as pd
import numpy as np
import ftfy

def fix_encoding(val):
    if not isinstance(val, str):
        return val
    val = ftfy.fix_text(val)
    try:
        raw = val.encode('utf-8')
        fixed_bytes = bytearray()
        i = 0
        while i < len(raw):
            if (i + 3 < len(raw)
                    and raw[i] == 0xC3 and 0x80 <= raw[i+1] <= 0xBF
                    and raw[i+2] == 0xC2 and 0x80 <= raw[i+3] <= 0xBF):
                b1 = (raw[i+1] | 0x40) 
                b2 = raw[i+3]
                fixed_bytes.append(b1)
                fixed_bytes.append(b2)
                i += 4
            else:
                fixed_bytes.append(raw[i])
                i += 1
        return fixed_bytes.decode('utf-8', errors='replace')
    except Exception:
        return val


class ElectionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_all_multi_mandate_raw(self, year: int, district: str = None, precinct: str = None):
        table_map = {
            2024: Daugiamandates2024,
            2020: Daugiamandates2020,
            2016: Daugiamandates2016,
            2012: Daugiamandates2012
        }
        model = table_map.get(year)
        if not model:
            return None
        
        stmt = select(model)
        if district:
            stmt = stmt.where(model.APYGARDOS_PAVADINIMAS == district)
        if precinct:
            stmt = stmt.where(model.APYLINKES_PAVADINIMAS == precinct)
            
        result = self.session.execute(stmt)
        return result.scalars().all()

    def get_districts(self, year: int):
        table_map = {2024: Daugiamandates2024, 2020: Daugiamandates2020, 2016: Daugiamandates2016, 2012: Daugiamandates2012}
        model = table_map.get(year)
        if not model: return []
        stmt = select(model.APYGARDOS_PAVADINIMAS).distinct().order_by(model.APYGARDOS_PAVADINIMAS)
        results = self.session.execute(stmt).scalars().all()
        return [fix_encoding(r) for r in results]

    def get_precincts(self, year: int, district: str):
        table_map = {2024: Daugiamandates2024, 2020: Daugiamandates2020, 2016: Daugiamandates2016, 2012: Daugiamandates2012}
        model = table_map.get(year)
        if not model or not district: return []
        stmt = select(model.APYLINKES_PAVADINIMAS).where(model.APYGARDOS_PAVADINIMAS == district).distinct().order_by(model.APYLINKES_PAVADINIMAS)
        results = self.session.execute(stmt).scalars().all()
        if not results:
            try:
                raw_district = district.encode('utf-8').decode('latin-1')
                stmt2 = select(model.APYLINKES_PAVADINIMAS).where(model.APYGARDOS_PAVADINIMAS == raw_district).distinct().order_by(model.APYLINKES_PAVADINIMAS)
                results = self.session.execute(stmt2).scalars().all()
            except Exception:
                pass
        return [fix_encoding(r) for r in results]

    def get_dataframe_for_ml(self, year: int, district: str = None, precinct: str = None):
        raw_data = self.get_all_multi_mandate_raw(year, district, precinct)
        if not raw_data:
            return pd.DataFrame()
        
        data = []
        for row in raw_data:
            d = {c.name: getattr(row, c.name) for c in row.__table__.columns}
            data.append(d)
        
        df = pd.DataFrame(data)
        df.columns = [c[4:] if c.startswith('col_') else c for c in df.columns]
        
        for col in ['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS', 'SARASO_PAVADINIMAS']:
            if col in df.columns:
                df[col] = df[col].apply(fix_encoding)
        
        numeric_cols = [
            'RINKEJU_SKAICIUS', 'VISO_DALYVAVO', 'BALSADEZEJE_GALIOJANTYS', 
            'BALSADEZEJE_NEGALIOJANTYS', 'BALSU_BALSADEZEJE', 'BALSU_PASTU', 'BALSU_VISO', 'BALSU_SKAICIUS'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # 1. Clean strings BEFORE conversion to catch European spaces/commas
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.')
                # 2. Convert and strictly avoid the fillna() FutureWarning
                df[col] = pd.to_numeric(df[col], errors='coerce').replace({np.nan: 0})
        
        df['YEAR'] = year
        
        if 'BALSU_VISO' in df.columns and 'VISO_DALYVAVO' in df.columns:
            df['VOTE_SHARE'] = df['BALSU_VISO'] / df.groupby('APYLINKES_PAVADINIMAS')['BALSU_VISO'].transform('sum')
            # Avoid fillna() here as well
            df['VOTE_SHARE'] = df['VOTE_SHARE'].replace({np.nan: 0})
            
        return df

class ModelRepository:
    def __init__(self, session: Session):
        self.session = session

    def register_model(self, model_entry: MLModelRegistry):
        self.session.add(model_entry)
        self.session.commit()
        return model_entry

    def get_latest_model(self, model_type: str):
        stmt = select(MLModelRegistry).where(MLModelRegistry.model_type == model_type).order_by(MLModelRegistry.training_date.desc()).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()