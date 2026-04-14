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

    def get_national_results(self, year: int) -> pd.DataFrame:
        """Ištraukia visus nurodytų metų rinkimų rezultatus į Pandas DataFrame grafikams."""
        
        stmt = (
            select(
                District.name.label("APYGARDOS_PAVADINIMAS"),
                Precinct.name.label("APYLINKES_PAVADINIMAS"),
                Participant.name.label("SARASO_PAVADINIMAS"),
                VoteResult.votes_total,
                Precinct.total_participated
            )
            .select_from(VoteResult)
            .join(Precinct, VoteResult.precinct_id == Precinct.id)
            .join(District, Precinct.district_id == District.id)
            .join(Election, District.election_id == Election.id)
            .join(Participant, VoteResult.participant_id == Participant.id)
            .where(Election.year == year)
        )

        results = self.session.execute(stmt).all()
        
        if not results:
            return pd.DataFrame()

        # Paverčiame į DataFrame
        df = pd.DataFrame(results)
        
        # Suskaičiuojame procentinę balsų dalį (VOTE_SHARE)
        # Apsauga nuo dalybos iš nulio apylinkėse be balsuotojų
        df['VOTE_SHARE'] = 0.0
        mask = df['total_participated'] > 0
        df.loc[mask, 'VOTE_SHARE'] = (df.loc[mask, 'votes_total'] / df.loc[mask, 'total_participated']) * 100

        return df