import sqlite3
import os

def fix_mojibake(text):
    if not text:
        return text
    try:
        # Common pattern: UTF-8 bytes read as CP1252 or ISO-8859-1
        fixed = text.encode('cp1252').decode('utf-8')
    except Exception:
        try:
            fixed = text.encode('iso-8859-1').decode('utf-8')
        except Exception:
            fixed = text
    
    # Second pass for characters where 0x8D was mapped to 0xA8 or similar
    # or other specific remnants
    replacements = {
        '\u0128': 'č',  # Ĩ -> č
        '\u0129': 'Č',  # Ĩ-like? Check
        'Ä\x8d': 'č',
        'Ä\x8c': 'Č',
        'Ä™': 'ę',
        'Ä™': 'Ę',
        'Ä—': 'ė',
        'Ä–': 'Ė',
        'Ä¯': 'į',
        'Ä®': 'Į',
        'Å¡': 'š',
        'Å\xa0': 'Š',
        'Å³': 'ų',
        'Å²': 'Ų',
        'Å¾': 'ž',
        'Å½': 'Ž',
    }
    for old, new in replacements.items():
        fixed = fixed.replace(old, new)
    return fixed

def clean_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]

    print(f"Cleaning database: {db_path}")
    
    for table in tables:
        print(f"Processing table: {table}")
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        
        # Only process columns that are likely to contain text
        text_columns = [col[1] for col in columns if 'TEXT' in col[2].upper() or col[2] == '']
        
        if not text_columns:
            continue

        quoted_columns = [f'"{col}"' for col in text_columns]
        cursor.execute(f'SELECT rowid, {", ".join(quoted_columns)} FROM "{table}"')
        rows = cursor.fetchall()
        
        updates = []
        for row in rows:
            rowid = row[0]
            original_values = row[1:]
            fixed_values = [fix_mojibake(val) if isinstance(val, str) else val for val in original_values]
            
            if fixed_values != list(original_values):
                updates.append((fixed_values, rowid))

        if updates:
            print(f"  Updating {len(updates)} rows in {table}...")
            for fixed_vals, rowid in updates:
                set_clause = ", ".join([f'"{col}" = ?' for col in text_columns])
                cursor.execute(f'UPDATE "{table}" SET {set_clause} WHERE rowid = ?', (*fixed_vals, rowid))
            
    conn.commit()
    conn.close()
    print("Database cleaning completed.")

if __name__ == "__main__":
    db_file = 'rinkimai.db'
    if os.path.exists(db_file):
        # Create a backup first
        backup_file = db_file + '.bak'
        import shutil
        if not os.path.exists(backup_file):
            shutil.copy2(db_file, backup_file)
            print(f"Backup created: {backup_file}")
        
        clean_database(db_file)
    else:
        print(f"Database not found: {db_file}")
