import sqlite3
from collections import defaultdict
import pandas as pd

conn = sqlite3.connect('mlflow.db')
cur = conn.cursor()

# Run 112b8d1059bb449398b3087edbf86347 (25 epochs)
run_id = '112b8d1059bb449398b3087edbf86347'
cur.execute("""
    SELECT step, key, value 
    FROM metrics 
    WHERE run_uuid=?
    ORDER BY step, key
""", (run_id,))

rows = cur.fetchall()
data = defaultdict(dict)
for step, key, val in rows:
    data[step][key] = val

df = pd.DataFrame.from_dict(data, orient='index').sort_index()
print("Epochs logged:", len(df))
interesting_cols = [c for c in df.columns if 'val_' in c]
print(df[interesting_cols].to_string())
