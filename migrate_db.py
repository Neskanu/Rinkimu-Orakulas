import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.data.models import Base, Election, District, Precinct, Participant, VoteResult, ElectionType
import os

# 1. Prisijungimai prie senos ir naujos bazės
OLD_DB = "sqlite:///rinkimai.db"
NEW_DB = "sqlite:///rinkimai_v2.db"

old_engine = create_engine(OLD_DB)
new_engine = create_engine(NEW_DB)

def migrate():
    print("Kuriama nauja relacinė duomenų bazės struktūra...")
    Base.metadata.create_all(new_engine)
    
    with Session(new_engine) as session:
        # Pavyzdys: Migruojame 2024 m. daugiamandatės duomenis
        print("Skaitomi seni 2024 m. duomenys...")
        try:
            df = pd.read_sql_table('daugiamandates_2024', old_engine)
        except ValueError:
            print("Klaida: Nerasta lentelė 'daugiamandates_2024' senojoje bazėje.")
            return

        # Išvalome nereikalingus tarpus ar NaN reikšmes
        df['RINKEJU_SKAICIUS'] = pd.to_numeric(df['RINKEJU_SKAICIUS'], errors='coerce').fillna(0)
        df['VISO_DALYVAVO'] = pd.to_numeric(df['VISO_DALYVAVO'], errors='coerce').fillna(0)
        df['BALSU_VISO'] = pd.to_numeric(df['BALSU_VISO'], errors='coerce').fillna(0)

        # 1. Sukuriame Rinkimų įrašą
        election = Election(
            year=2024, 
            type=ElectionType.PARLIAMENT_MULTI, 
            name="2024 m. Seimo Rinkimai (Daugiamandatė)"
        )
        session.add(election)
        session.commit() # Commit, kad gautume election.id
        
        print("Generuojamos apygardos, apylinkės ir dalyviai...")
        
        # 2. Žodynai greitam ID suradimui, kad nereikėtų daryti tūkstančių SQL užklausų
        district_map = {}
        precinct_map = {}
        participant_map = {}
        
        # 3. Ištraukiame ir išsaugome unikalius Dalyvius (Partijas)
        unique_parties = df['SARASO_PAVADINIMAS'].dropna().unique()
        for party_name in unique_parties:
            p = Participant(election_id=election.id, name=party_name, is_individual=False)
            session.add(p)
            session.flush()
            participant_map[party_name] = p.id
            
        # 4. Ištraukiame ir išsaugome unikalias Apygardas
        unique_districts = df['APYGARDOS_PAVADINIMAS'].dropna().unique()
        for dist_name in unique_districts:
            d = District(election_id=election.id, name=dist_name)
            session.add(d)
            session.flush()
            district_map[dist_name] = d.id
            
        # 5. Ištraukiame unikalias Apylinkes (apylinkės pavadinimas + apygarda)
        # Naudojame drop_duplicates, nes vienoje apylinkėje balsavo už daug partijų, 
        # bet rinkėjų skaičius apylinkėje yra tas pats.
        precincts_df = df.drop_duplicates(subset=['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS'])
        for _, row in precincts_df.iterrows():
            dist_id = district_map[row['APYGARDOS_PAVADINIMAS']]
            prec = Precinct(
                district_id=dist_id,
                name=row['APYLINKES_PAVADINIMAS'],
                registered_voters=int(row['RINKEJU_SKAICIUS']),
                total_participated=int(row['VISO_DALYVAVO'])
            )
            session.add(prec)
            session.flush()
            precinct_map[(row['APYGARDOS_PAVADINIMAS'], row['APYLINKES_PAVADINIMAS'])] = prec.id
            
        # 6. Surenkame pačius balsavimo rezultatus
        print("Įkeliami balsavimo rezultatai. Tai gali užtrukti kelias sekundes...")
        results_to_insert = []
        for _, row in df.iterrows():
            dist_name = row['APYGARDOS_PAVADINIMAS']
            prec_name = row['APYLINKES_PAVADINIMAS']
            party_name = row['SARASO_PAVADINIMAS']
            
            if pd.isna(dist_name) or pd.isna(prec_name) or pd.isna(party_name):
                continue
                
            prec_id = precinct_map.get((dist_name, prec_name))
            part_id = participant_map.get(party_name)
            
            if prec_id and part_id:
                results_to_insert.append(VoteResult(
                    precinct_id=prec_id,
                    participant_id=part_id,
                    votes_total=int(row['BALSU_VISO'])
                ))
        
        # Optimizuotas masinis įkėlimas (Bulk Insert)
        session.bulk_save_objects(results_to_insert)
        session.commit()
        print(f"Migracija baigta sėkmingai! Įkelta rezultatų: {len(results_to_insert)}")

if __name__ == "__main__":
    migrate()