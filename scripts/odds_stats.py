import pandas as pd

# Funzione per calcolare la probabilità implicita e il margine per ciascun bookmaker e salvare in un DataFrame
def calcolare_probabilita_e_margine_per_bookmaker(df, nome_squadra):
    # Filtra i dati per le partite della squadra selezionata (sia in casa che in trasferta)
    partite_casa = df[df['HomeTeam'] == nome_squadra]
    partite_trasferta = df[df['AwayTeam'] == nome_squadra]
    partite = pd.concat([partite_casa, partite_trasferta])

    # Colonne dei bookmaker per ogni tipo di risultato
    bookmaker_columns = {
        'Bet365': ['B365H', 'B365D', 'B365A'],
        'Bet&Win': ['BWH', 'BWD', 'BWA'],
        'Interwetten': ['IWH', 'IWD', 'IWA'],
        'Pinnacle': ['PSH', 'PSD', 'PSA'],
        'William Hill': ['WHH', 'WHD', 'WHA'],
        'VC Bet': ['VCH', 'VCD', 'VCA']
    }

    # Lista per raccogliere i dati del DataFrame
    risultati = []

    # Itera sui bookmaker e calcola le probabilità e il margine per ciascuno
    for bookmaker, colonne in bookmaker_columns.items():
        prob_vittoria = []
        prob_pareggio = []
        prob_sconfitta = []
        
        for _, partita in partite.iterrows():
            try:
                quota_vittoria = pd.to_numeric(partita[colonne[0]], errors='coerce')
                quota_pareggio = pd.to_numeric(partita[colonne[1]], errors='coerce')
                quota_sconfitta = pd.to_numeric(partita[colonne[2]], errors='coerce')
                
                # Calcola la probabilità implicita per ogni quota del bookmaker corrente
                if pd.notna(quota_vittoria):  
                    prob_vittoria.append(1 / quota_vittoria)
                if pd.notna(quota_pareggio):
                    prob_pareggio.append(1 / quota_pareggio)
                if pd.notna(quota_sconfitta):
                    prob_sconfitta.append(1 / quota_sconfitta)
            except TypeError:
                continue

        # Calcola la media delle probabilità per i tre risultati del bookmaker
        media_vittoria = sum(prob_vittoria) / len(prob_vittoria) if prob_vittoria else 0
        media_pareggio = sum(prob_pareggio) / len(prob_pareggio) if prob_pareggio else 0
        media_sconfitta = sum(prob_sconfitta) / len(prob_sconfitta) if prob_sconfitta else 0

        # Calcolo del margine del bookmaker
        margine = (media_vittoria + media_pareggio + media_sconfitta - 1) * 100

        # Converti le probabilità in percentuali
        media_vittoria_percent = media_vittoria * 100
        media_pareggio_percent = media_pareggio * 100
        media_sconfitta_percent = media_sconfitta * 100

        # Aggiungi i risultati alla lista
        risultati.append({
            'Bookmaker': bookmaker,
            'Vittoria (%)': round(media_vittoria_percent, 2),
            'Pareggio (%)': round(media_pareggio_percent, 2),
            'Sconfitta (%)': round(media_sconfitta_percent, 2),
            'Margine (%)': round(margine, 2)
        })

    # Crea un DataFrame dai risultati
    df_risultati = pd.DataFrame(risultati)
    return df_risultati

# Esempio di utilizzo della funzione
file_path = 'I1.json'  # Cambia il percorso se necessario
df = pd.read_json(file_path)

nome_squadra = 'Empoli'  # Cambia con il nome della squadra desiderata
df_risultati = calcolare_probabilita_e_margine_per_bookmaker(df, nome_squadra)
print(df_risultati)
