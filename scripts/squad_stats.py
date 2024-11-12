import pandas as pd

# Leggi il file JSON e carica i dati in un DataFrame
file_path = 'I1.json'  # Cambia il percorso se necessario
df = pd.read_json(file_path)

# Controllo per valori nulli o non numerici
df = df.dropna(subset=['FTR', 'FTHG', 'FTAG', 'HS', 'HST', 'HC', 'HF', 'HY', 'HR', 'AS', 'AST', 'AC', 'AF', 'AY', 'AR'])
df[['FTHG', 'FTAG', 'HS', 'HST', 'HC', 'HF', 'HY', 'HR', 'AS', 'AST', 'AC', 'AF', 'AY', 'AR']] = df[['FTHG', 'FTAG', 'HS', 'HST', 'HC', 'HF', 'HY', 'HR', 'AS', 'AST', 'AC', 'AF', 'AY', 'AR']].apply(pd.to_numeric, errors='coerce')

# Funzione per calcolare le statistiche per una specifica squadra
def calcolare_statistiche_per_squadra(nome_squadra, df):
    # Filtro i dati per la squadra selezionata
    partite_casa = df[df['HomeTeam'] == nome_squadra]
    partite_trasferta = df[df['AwayTeam'] == nome_squadra]
    
    # Statistiche per la squadra di casa
    vittorie_casa = len(partite_casa[partite_casa['FTR'] == 'H'])
    vittorie_trasferta = len(partite_trasferta[partite_trasferta['FTR'] == 'A'])
    
    # Calcolo dei pareggi separando quelli in casa e in trasferta
    pareggi_casa = len(partite_casa[partite_casa['FTR'] == 'D'])
    pareggi_trasferta = len(partite_trasferta[partite_trasferta['FTR'] == 'D'])
    pareggi = pareggi_casa + pareggi_trasferta
    
    # Gol segnati e subiti
    gol_casa = partite_casa['FTHG'].sum()
    gol_subiti_casa = partite_casa['FTAG'].sum()
    gol_trasferta = partite_trasferta['FTAG'].sum()
    gol_subiti_trasferta = partite_trasferta['FTHG'].sum()

    # Tiri totali
    tiri_totali_casa = partite_casa['HS'].sum()
    tiri_totali_trasferta = partite_trasferta['AS'].sum()

    # Media tiri
    media_tiri_casa = partite_casa['HS'].mean() if len(partite_casa) > 0 else 0
    media_tiri_trasferta = partite_trasferta['AS'].mean() if len(partite_trasferta) > 0 else 0

    # Tiri in porta
    tiri_in_porta_casa = partite_casa['HST'].sum()
    tiri_in_porta_trasferta = partite_trasferta['AST'].sum()

    # Media dei tiri in porta
    media_tiri_in_porta_casa = partite_casa['HST'].mean() if len(partite_casa) > 0 else 0
    media_tiri_in_porta_trasferta = partite_trasferta['AST'].mean() if len(partite_trasferta) > 0 else 0

    # Calci d'angolo
    calci_angolo_casa = partite_casa['HC'].sum()
    calci_angolo_trasferta = partite_trasferta['AC'].sum()

    # Falli
    falli_casa = partite_casa['HF'].sum()
    falli_trasferta = partite_trasferta['AF'].sum()

    # Cartellini gialli e rossi
    cartellini_gialli_rossi_casa = (partite_casa['HY'] * 10 + partite_casa['HR'] * 25).sum()
    cartellini_gialli_rossi_trasferta = (partite_trasferta['AY'] * 10 + partite_trasferta['AR'] * 25).sum()

    # Creazione di un dizionario con le statistiche in italiano
    stats = {
        'Nome': nome_squadra,  # Aggiunto il campo Nome
        'Vittorie in casa': vittorie_casa,
        'Vittorie in trasferta': vittorie_trasferta,
        'Pareggi': pareggi,
        'Media gol in casa': gol_casa / len(partite_casa) if len(partite_casa) > 0 else 0,
        'Media gol subiti in casa': gol_subiti_casa / len(partite_casa) if len(partite_casa) > 0 else 0,
        'Media gol segnati in trasferta': gol_trasferta / len(partite_trasferta) if len(partite_trasferta) > 0 else 0,
        'Media gol subiti in trasferta': gol_subiti_trasferta / len(partite_trasferta) if len(partite_trasferta) > 0 else 0,
        'Tiri totali in casa': tiri_totali_casa,
        'Tiri totali in trasferta': tiri_totali_trasferta,
        'Media tiri in casa': media_tiri_casa,
        'Media tiri in trasferta': media_tiri_trasferta,
        'Tiri in porta in casa': tiri_in_porta_casa,
        'Tiri in porta in trasferta': tiri_in_porta_trasferta,
        'Media tiri in porta in casa': media_tiri_in_porta_casa,
        'Media tiri in porta in trasferta': media_tiri_in_porta_trasferta,
        'Calci d\'angolo in casa': calci_angolo_casa,
        'Calci d\'angolo in trasferta': calci_angolo_trasferta,
        'Falli in casa': falli_casa,
        'Falli in trasferta': falli_trasferta,
        'Cartellini gialli e rossi in casa': cartellini_gialli_rossi_casa,
        'Cartellini gialli e rossi in trasferta': cartellini_gialli_rossi_trasferta
    }
    
    # Restituisce il dizionario delle statistiche
    return stats

# Esempio di utilizzo della funzione per la Juventus
nome_squadra = 'Empoli'  # Cambia con il nome della squadra desiderata
stats = calcolare_statistiche_per_squadra(nome_squadra, df)

# Mostra le statistiche per la squadra in formato colonna
for stat, valore in stats.items():
    print(f"{stat}: {valore}")
