import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.data.models import Base, Election, District, Precinct, Participant, VoteResult, ElectionType

OLD_DB = "sqlite:///rinkimai.db"
NEW_DB = "sqlite:///rinkimai_v2.db"

old_engine = create_engine(OLD_DB)
new_engine = create_engine(NEW_DB)

# Lentelių, kurias norime perkelti, sąrašas
tables_to_migrate = {
    2012: 'daugiamandates_2012',
    2016: 'daugiamandates_2016',
    2020: 'daugiamandates_2020',
    2024: 'daugiamandates_2024'
}

def migrate_all():
    print("🧹 Išvaloma ir perkuriami duomenų bazės (V2) pamatai...")
    Base.metadata.drop_all(new_engine)  # Ištrina senus likučius
    Base.metadata.create_all(new_engine) # Sukuria šviežias lenteles
    
    with Session(new_engine) as session:
        for year, table_name in tables_to_migrate.items():
            print(f"\n⏳ Skaitomi {year} m. duomenys iš '{table_name}'...")
            try:
                df = pd.read_sql_table(table_name, old_engine)
            except ValueError:
                print(f"⚠️ Klaida: Nerasta lentelė '{table_name}'. Praleidžiama.")
                continue

            # --- PRIDĖKITE ŠIAS TRIS EILUTES ČIA ---
            # Jei istorinėje lentelėje (pvz., 2012 m.) nėra apylinkių, sukuriame suminę apylinkę
            if 'APYLINKES_PAVADINIMAS' not in df.columns:
                df['APYLINKES_PAVADINIMAS'] = "Apygardos suminiai rezultatai"
            # --------------------------------------

            # Išvalome nereikalingus tarpus ar NaN reikšmes
            if 'RINKEJU_SKAICIUS' in df.columns:
                df['RINKEJU_SKAICIUS'] = pd.to_numeric(df['RINKEJU_SKAICIUS'], errors='coerce').fillna(0)
            if 'VISO_DALYVAVO' in df.columns:
                df['VISO_DALYVAVO'] = pd.to_numeric(df['VISO_DALYVAVO'], errors='coerce').fillna(0)
            if 'BALSU_VISO' in df.columns:
                df['BALSU_VISO'] = pd.to_numeric(df['BALSU_VISO'], errors='coerce').fillna(0)

            # 1. Sukuriame Rinkimų įrašą
            election = Election(
                year=year, 
                type=ElectionType.PARLIAMENT_MULTI, 
                name=f"{year} m. Seimo Rinkimai (Daugiamandatė)"
            )
            session.add(election)
            session.commit()
            
            district_map = {}
            precinct_map = {}
            participant_map = {}
            
            print(f"   ⚙️ Generuojami dalyviai ir apygardos ({year})...")
            # Unikalios partijos
            for party_name in df['SARASO_PAVADINIMAS'].dropna().unique():
                p = Participant(election_id=election.id, name=party_name, is_individual=False)
                session.add(p)
                session.flush()
                participant_map[party_name] = p.id
                
            # Unikalios apygardos
            for dist_name in df['APYGARDOS_PAVADINIMAS'].dropna().unique():
                d = District(election_id=election.id, name=dist_name)
                session.add(d)
                session.flush()
                district_map[dist_name] = d.id
                
            # Unikalios apylinkės
            precincts_df = df.drop_duplicates(subset=['APYGARDOS_PAVADINIMAS', 'APYLINKES_PAVADINIMAS'])
            for _, row in precincts_df.iterrows():
                dist_id = district_map.get(row['APYGARDOS_PAVADINIMAS'])
                if not dist_id: continue
                
                # 2012 m. duomenyse gali trūkti VISO_DALYVAVO, todėl naudojame .get() su fallback
                voters = int(row.get('RINKEJU_SKAICIUS', 0))
                participated = int(row.get('VISO_DALYVAVO', 0))
                
                prec = Precinct(
                    district_id=dist_id,
                    name=row['APYLINKES_PAVADINIMAS'],
                    registered_voters=voters,
                    total_participated=participated
                )
                session.add(prec)
                session.flush()
                precinct_map[(row['APYGARDOS_PAVADINIMAS'], row['APYLINKES_PAVADINIMAS'])] = prec.id
                
            print(f"   💾 Įkeliami balsavimo rezultatai ({year}). Tai gali užtrukti...")
            results_to_insert = []
            for _, row in df.iterrows():
                dist_name = row['APYGARDOS_PAVADINIMAS']
                prec_name = row['APYLINKES_PAVADINIMAS']
                party_name = row['SARASO_PAVADINIMAS']
                
                if pd.isna(dist_name) or pd.isna(prec_name) or pd.isna(party_name):
                    continue
                    
                prec_id = precinct_map.get((dist_name, prec_name))
                part_id = participant_map.get(party_name)
                votes = int(row.get('BALSU_VISO', 0))
                
                if prec_id and part_id:
                    results_to_insert.append(VoteResult(
                        precinct_id=prec_id,
                        participant_id=part_id,
                        votes_total=votes
                    ))
            
            session.bulk_save_objects(results_to_insert)
            session.commit()
            print(f"   ✅ {year} metai baigti. Įrašų: {len(results_to_insert)}")

    print("\n🎉 VISA MIGRACIJA BAIGTA!")

if __name__ == "__main__":
    migrate_all()