import pandas as pd

# Leggi il file JSON e carica i dati in un DataFrame
file_path = 'I1.json'  # Cambia il percorso se necessario

# Carica i dati JSON nel DataFrame
df = pd.read_json(file_path)

# Mostra le prime righe del DataFrame per verificare la corretta importazione
print(df.head())
