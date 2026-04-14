import sys, ftfy
sys.path.insert(0, '.')
from app.data.repositories import fix_encoding
import sqlite3

c = sqlite3.connect('rinkimai.db')
rows = c.execute("SELECT DISTINCT APYGARDOS_PAVADINIMAS FROM daugiamandates_2024").fetchall()
problem_chars = set()
for r in rows:
    raw = r[0]
    fixed = fix_encoding(raw)
    raw_hex = raw.encode('utf-8').hex()
    fix_hex = fixed.encode('utf-8').hex()
    if raw_hex != fix_hex:
        print(f"FIXED: {repr(raw)} -> {repr(fixed)}")
        
print("\nAll district names after fix:")
all_fixed = sorted(set(fix_encoding(r[0]) for r in rows))
for d in all_fixed:
    print(" ", d)
c.close()

