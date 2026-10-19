import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path('src').absolute()))
import feature_builder as fb

df_ab, df_pr = fb._load_raw_data(Path('src/output_dl_seed99999'))
print("Absolventen-Spalten:", df_ab.columns.tolist())
print("\nBeispiel-Datensatz Absolventen:")
print(df_ab.head(1).T)
