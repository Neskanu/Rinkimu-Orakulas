import sqlite3
import os
import re

def fix_mojibake(text):
    if not text:
        return text
    
    # Step 1: Fix known "terminal" mojibake sequences (broken bytes)
    # These are signatures identified from hex analysis
    signatures = {
        'Å\ufffd': 'š',      # š / Š (cased in step 4)
        'Ä\u00a8': 'č',      # č
        'Ä\u2026': 'ą',      # ą
        'Ä\u00c6': 'į',      # į (from 'ateitÄÆ')
        'Ä\u0099': 'ę',      # ę
        'Ä\u008d': 'č',      # č (alternative mapping)
        'Ä\u0097': 'ė',      # ė
        'Ä\u00af': 'į',      # į
        'ā\u20ac\u02db': '„', # Double-quoted left
        'ā\u20ac\ufffd': '“', # Double-quoted right
        'ā\u20ac\u02dc': '“', # Alternative
    }
    
    for old, new in signatures.items():
        text = text.replace(old, new)

    # Step 2: Try standard double-encoding recovery for remaining chars (Ž, ų, etc.)
    try:
        # Only attempt if there's evidence of mojibake (typical Latin-1 chars)
        if any(c in text for c in 'ÅÄÖÜ'):
            fixed = text.encode('cp1252').decode('utf-8')
            # If the output is clean, take it
            if '\ufffd' not in fixed:
                text = fixed
    except Exception:
        try:
            fixed = text.encode('iso-8859-1').decode('utf-8')
            if '\ufffd' not in fixed:
                text = fixed
        except Exception:
            pass

    # Step 3: Specific remnants fix
    remnants = {
        'Ä™': 'ę',
        'Ä—': 'ė',
        'Ä¯': 'į',
        'Å¡': 'š',
        'Å\xa0': 'Š',
        'Å³': 'ų',
        'Å²': 'Ų',
        'Å¾': 'ž',
        'Å½': 'Ž',
        'sÄ\u2026junga': 'sąjunga', # Catch-all for sÄ…junga -> sąjunga
        'DEMOKRATÅ²': 'DEMOKRATŲ',
        'SÄ…JUNGA': 'SĄJUNGA'
    }
    for old, new in remnants.items():
        text = text.replace(old, new)

    # Step 4: Contextual Casing for 'š' (since Å\ufffd maps to both Š and š)
    # If 'š' is at the start of a word, it should be 'Š'
    def capitalize_lithuanian(match):
        return match.group(0).upper()
    
    text = re.sub(r'(^|[\s\-\(])š', capitalize_lithuanian, text)
    
    return text

def clean_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_') and row[0] != 'alembic_version']

    print(f"Refining characters in database: {db_path}")
    
    total_updated = 0
    for table in tables:
        print(f"  Processing table: {table}")
        cursor.execute(f"PRAGMA table_info(\"{table}\");")
        columns = cursor.fetchall()
        
        text_columns = [col[1] for col in columns if 'TEXT' in col[2].upper() or col[2] == '']
        if not text_columns: continue

        quoted_columns = [f'"{col}"' for col in text_columns]
        cursor.execute(f'SELECT rowid, {", ".join(quoted_columns)} FROM "{table}"')
        rows = cursor.fetchall()
        
        for row in rows:
            rowid = row[0]
            original_values = row[1:]
            fixed_values = [fix_mojibake(val) if isinstance(val, str) else val for val in original_values]
            
            if fixed_values != list(original_values):
                set_clause = ", ".join([f'"{col}" = ?' for col in text_columns])
                cursor.execute(f'UPDATE "{table}" SET {set_clause} WHERE rowid = ?', (*fixed_values, rowid))
                total_updated += 1
                
        # Optional: Print samples periodically
        if total_updated % 1000 == 0 and total_updated > 0:
            print(f"    {total_updated} records updated so far...")

    conn.commit()
    conn.close()
    print(f"Database cleaning completed. {total_updated} total rows updated.")

if __name__ == "__main__":
    db_file = 'rinkimai.db'
    if os.path.exists(db_file):
        clean_database(db_file)
    else:
        print(f"Database not found: {db_file}")
