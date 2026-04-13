from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.data.models import *
import pandas as pd

import ftfy

def fix_encoding(val):
    """Fix Lithuanian mojibake (mixed double-encoded UTF-8 in SQLite).
    Two passes:
    1. ftfy fixes common garbled sequences (em-dashes, quotation marks, etc.)
    2. Byte-level scan repairs remaining double-encoded UTF-8 sequences
       (e.g. Ã…« -> ū, caused by UTF-8 bytes read as Latin-1 then re-encoded)
    """
    if not isinstance(val, str):
        return val
    # Pass 1: ftfy (handles most common garbling)
    val = ftfy.fix_text(val)
    # Pass 2: detect and fix double-encoded multi-byte sequences
    # Encode back to UTF-8 bytes and attempt to re-decode any latin-1 misread sequences
    try:
        raw = val.encode('utf-8')
        # Scan for double-encoded patterns: C3 8x C2 xx  (latin-1 of a 2-byte UTF-8 sequence)
        fixed_bytes = bytearray()
        i = 0
        while i < len(raw):
            # Detect: C3 8? C2 ?? -> these 4 bytes are a double-encoded 2-byte UTF-8 sequence
            if (i + 3 < len(raw)
                    and raw[i] == 0xC3 and 0x80 <= raw[i+1] <= 0xBF
                    and raw[i+2] == 0xC2 and 0x80 <= raw[i+3] <= 0xBF):
                # Reconstruct original bytes: (raw[i+1] ^ 0x40) and raw[i+3]
                b1 = (raw[i+1] | 0x40)  # undo latin-1 encoding of the high byte
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
        # district coming from form may already be fixed; try both raw and encoded
        stmt = select(model.APYLINKES_PAVADINIMAS).where(model.APYGARDOS_PAVADINIMAS == district).distinct().order_by(model.APYLINKES_PAVADINIMAS)
        results = self.session.execute(stmt).scalars().all()
        if not results:
            # Try the raw (mojibake) version of the district name for DB lookup
            try:
                raw_district = district.encode('utf-8').decode('latin-1')
                stmt2 = select(model.APYLINKES_PAVADINIMAS).where(model.APYGARDOS_PAVADINIMAS == raw_district).distinct().order_by(model.APYLINKES_PAVADINIMAS)
                results = self.session.execute(stmt2).scalars().all()
            except Exception:
                pass
        return [fix_encoding(r) for r in results]

    def get_dataframe_for_ml(self, year: int, district: str = None, precinct: str = None):
        """Returns a cleaned pandas DataFrame for ML processing with calculated vote shares."""
        raw_data = self.get_all_multi_mandate_raw(year, district, precinct)
        if not raw_data:
            return pd.DataFrame()
        
        # Convert to list of dicts
        data = []
        for row in raw_data:
            d = {c.name: getattr(row, c.name) for c in row.__table__.columns}
            data.append(d)
        
        df = pd.DataFrame(data)
        
        # Cleanup Column Names (if col_ prefix was added)
        df.columns = [c[4:] if c.startswith('col_') else c for c in df.columns]
        
        # Fix mojibake encoding on all string/name columns
        for col in ['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS', 'SARASO_PAVADINIMAS']:
            if col in df.columns:
                df[col] = df[col].apply(fix_encoding)
        
        # Convert numeric columns from string to float/int
        numeric_cols = [
            'RINKEJU_SKAICIUS', 'VISO_DALYVAVO', 'BALSADEZEJE_GALIOJANTYS', 
            'BALSADEZEJE_NEGALIOJANTYS', 'BALSU_BALSADEZEJE', 'BALSU_PASTU', 'BALSU_VISO'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['YEAR'] = year
        
        # Calculate Vote Share
        if 'BALSU_VISO' in df.columns and 'VISO_DALYVAVO' in df.columns:
            # We group by precinct (APYLINKES_PAVADINIMAS) to get the correct denominator for that location
            # or just use the sum of all BALSU_VISO in the current selection if it's filtered
            df['VOTE_SHARE'] = df['BALSU_VISO'] / df.groupby('APYLINKES_PAVADINIMAS')['BALSU_VISO'].transform('sum')
            df['VOTE_SHARE'] = df['VOTE_SHARE'].fillna(0)
            
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
