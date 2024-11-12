import pandas as pd
import json

# Funzione per caricare i file JSON
def carica_file_json(file_paths):
    dfs = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                matches = json.load(f)
                df_temp = pd.DataFrame(matches)
                
                # Aggiungi una colonna con il nome del file per tracciare la provenienza dei dati
                df_temp['File'] = file_path
                
                dfs.append(df_temp)
        except Exception as e:
            print(f"Errore nel caricamento del file {file_path}: {e}")
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else None

# Funzione per controllare i dati mancanti
def controllo_dati_mancanti(df):
    if df is None:
        print("Nessun dato disponibile per il controllo.")
        return

    # Sostituire valori vuoti, None, o NaN con NaN, ma escludere i valori 0
    df.replace([None, '', pd.NA], pd.NA, inplace=True)

    # Identificare i dati mancanti
    missing_data = df.isnull().sum()

    # Filtra solo le colonne con dati mancanti
    missing_data = missing_data[missing_data > 0]
    if missing_data.empty:
        print("Nessun dato mancante nel DataFrame.")
        return

    # Report dei dati mancanti
    print("\nReport dei dati mancanti:")
    print("-" * 50)
    
    for col in missing_data.index:
        print(f"\nColonna: {col}")
        print(f"Tipologia di dato: {df[col].dtype}")
        print(f"Numero di valori mancanti: {missing_data[col]}")
        print(f"Percentuale di valori mancanti: {missing_data[col] / len(df) * 100:.2f}%")
        
        # Esempi di righe con valori mancanti, includendo anche il nome del file
        missing_examples = df[df[col].isnull()][['Date', 'HomeTeam', 'AwayTeam', col, 'File']]
        print(f"Esempio di righe con valori mancanti nella colonna {col}:")
        print(missing_examples.head())

# Lista dei file JSON da importare
files = [
    './data/IT/B/ITB-2021.json',
    './data/IT/B/ITB-2122.json',
    './data/IT/B/ITB-2223.json',
    './data/IT/B/ITB-2324.json'
]

# Caricamento dei dati
df = carica_file_json(files)

# Verifica dei dati mancanti
controllo_dati_mancanti(df)
