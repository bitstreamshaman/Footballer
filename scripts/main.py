import pandas as pd
import json

# Funzione per leggere i dati dai file JSON
def carica_dati_da_file(nome_file):
    with open(nome_file, 'r', encoding='utf-8') as file:
        dati = json.load(file)
    return dati['matches']

# Carica i dati da tutti i file JSON
matches_seriea = carica_dati_da_file('seriea23-24.json')
matches_serieb = carica_dati_da_file('serieb23-24.json')
matches_coppa = carica_dati_da_file('coppa23-24.json')

# Funzione per determinare il vincitore di una partita
def determina_vincitore(match):
    # Verifica che esistano i risultati
    score = match.get('score', {}).get('ft', [None, None])
    if score[0] is None or score[1] is None:
        return None  # Ignora il match se non ha il punteggio finale
    
    score_1, score_2 = score
    
    # Determina il vincitore o il pareggio
    if score_1 > score_2:
        return match['team1']
    elif score_2 > score_1:
        return match['team2']
    else:
        return 'Draw'  # Pareggio

# Crea una lista con i vincitori per ogni competizione, ignorando i match senza vincitore
winners_seriea = [determina_vincitore(match) for match in matches_seriea if determina_vincitore(match) is not None]
winners_serieb = [determina_vincitore(match) for match in matches_serieb if determina_vincitore(match) is not None]
winners_coppa = [determina_vincitore(match) for match in matches_coppa if determina_vincitore(match) is not None]

# Calcola la frequenza di vittorie per ogni squadra in ogni competizione
freq_seriea = pd.Series(winners_seriea).value_counts(normalize=True)
freq_serieb = pd.Series(winners_serieb).value_counts(normalize=True)
freq_coppa = pd.Series(winners_coppa).value_counts(normalize=True)

# Unisci le frequenze in un unico dizionario di probabilità (media ponderata) e riempi i NaN con 0
probabilità_vittoria = (freq_seriea.add(freq_serieb, fill_value=0) + freq_coppa.add(freq_seriea, fill_value=0)) / 3
probabilità_vittoria = probabilità_vittoria.fillna(0)

# Funzione per stimare il vincitore tra due squadre
def stima_vincitore(squadra_1, squadra_2, prob_vittoria):
    prob_1 = prob_vittoria.get(squadra_1, 0)
    prob_2 = prob_vittoria.get(squadra_2, 0)

    print(f"Probabilità di vittoria per {squadra_1}: {prob_1:.2f}")
    print(f"Probabilità di vittoria per {squadra_2}: {prob_2:.2f}")

    if prob_1 > prob_2:
        return f"Potenziale vincitore: {squadra_1}"
    elif prob_2 > prob_1:
        return f"Potenziale vincitore: {squadra_2}"
    else:
        return "Le probabilità di vittoria sono uguali!"

# Esempio di input di due squadre
squadra_1 = 'Hellas Verona FC'  # Sostituisci con la tua squadra di interesse
squadra_2 = 'AS Roma'  # Sostituisci con la tua squadra di interesse

# Determina il vincitore
risultato = stima_vincitore(squadra_1, squadra_2, probabilità_vittoria)
print(risultato)
