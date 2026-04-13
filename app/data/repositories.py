import pandas as pd
from sqlalchemy import select
from app.data.models import Election, District, Precinct, Participant, VoteResult

class ElectionRepository:
    def __init__(self, session):
        self.session = session

    def get_districts(self, year: int):
        """Grąžina unikalius apygardų pavadinimus nurodytiems metams."""
        stmt = select(District.name)\
            .join(Election)\
            .where(Election.year == year)\
            .distinct()
        return self.session.scalars(stmt).all()

    def get_precincts(self, year: int, district_name: str):
        """Grąžina unikalius apylinkių pavadinimus konkrečiai apygardai."""
        stmt = select(Precinct.name)\
            .join(District).join(Election)\
            .where(Election.year == year, District.name == district_name)\
            .distinct()
        return self.session.scalars(stmt).all()

    def get_dataframe_for_ml(self, year: int):
        """
        Iš relacinių lentelių suformuoja vieną plokščią DataFrame,
        kurio stulpeliai atitinka senąją schemą (ML modeliui ir grafikams).
        """
        stmt = select(
            Participant.name.label('SARASO_PAVADINIMAS'),
            District.name.label('APYGARDOS_PAVADINIMAS'),
            Precinct.name.label('APYLINKES_PAVADINIMAS'),
            Precinct.registered_voters.label('RINKEJU_SKAICIUS'),
            Precinct.total_participated.label('VISO_DALYVAVO'),
            VoteResult.votes_total.label('BALSU_VISO')
        ).select_from(VoteResult)\
         .join(Participant).join(Precinct).join(District).join(Election)\
         .where(Election.year == year)
        
        # Sukuriame Pandas DataFrame tiesiai iš SQLAlchemy užklausos
        df = pd.read_sql(stmt, self.session.bind)
        return df