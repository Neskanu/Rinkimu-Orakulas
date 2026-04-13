from app.data.database import SessionLocal
from app.data.repositories import ElectionRepository
import pandas as pd

def run_debug():
    print("Fetching data from SQLite...")
    session = SessionLocal()
    repo = ElectionRepository(session)
    df = repo.get_dataframe_for_ml(2024)
    session.close()

    if df.empty:
        print("❌ Dataframe is empty! No 2024 data found.")
        return

    print(f"✅ Loaded {len(df)} rows.")
    print(f"📊 Available columns: {df.columns.tolist()}\n")

    # 1. Figure out which column we are using
    vote_col = None
    for col in ['BALSU_SKAICIUS', 'BALSAI_UZ_SARASA', 'PADUOTI_BALSAI', 'BALSU_VISO']:
        if col in df.columns:
            vote_col = col
            break

    print(f"🎯 Selected Vote Column: {vote_col}\n")

    # 2. Simulate the cleaning logic
    if df[vote_col].dtype == object:
        df[vote_col] = df[vote_col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.')
    
    df[vote_col] = pd.to_numeric(df[vote_col], errors='coerce').fillna(0)

    # 3. Simulate the grouping
    agg_df = df.groupby('SARASO_PAVADINIMAS')[vote_col].sum().reset_index()
    total_votes = agg_df[vote_col].sum()
    agg_df['SHARE_PCT'] = (agg_df[vote_col] / total_votes) * 100

    # 4. Print the final results
    print(f"📈 Total National Votes Calculated: {total_votes:,.0f}\n")
    print("🏆 Top 5 Parties by calculated sum:")
    print(agg_df.sort_values(by='SHARE_PCT', ascending=False).head(5)[['SARASO_PAVADINIMAS', vote_col, 'SHARE_PCT']])
    
    print("\n⚠️ Checking for identical sums (If these are all the same, this is our bug):")
    print(agg_df[vote_col].value_counts().head(5))

if __name__ == "__main__":
    run_debug()