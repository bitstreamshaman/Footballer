import pandas as pd
import numpy as np
from scipy.stats import poisson

# Funzione per calcolare la media dei goal in casa e in trasferta per una squadra
def calcola_media_goal(df, squadra):
    goal_casa = df[df['HomeTeam'] == squadra]['FTHG'].mean()  # Goal segnati in casa
    goal_trasferta = df[df['AwayTeam'] == squadra]['FTAG'].mean()  # Goal segnati in trasferta
    return goal_casa, goal_trasferta

# Funzione per calcolare la matrice di probabilità per i risultati esatti tra due squadre
def calcola_matrice_risultati(df, squadra_casa, squadra_trasferta, max_goal=5):
    # Calcola le medie dei goal per entrambe le squadre
    media_goal_casa, _ = calcola_media_goal(df, squadra_casa)
    _, media_goal_trasferta = calcola_media_goal(df, squadra_trasferta)
    
    # Stampa delle medie calcolate
    print(f"Media goal segnati in casa da {squadra_casa}: {media_goal_casa}")
    print(f"Media goal segnati in trasferta da {squadra_trasferta}: {media_goal_trasferta}")

    # Inizializza la matrice dei risultati
    matrice_risultati = pd.DataFrame(index=range(max_goal + 1), columns=range(max_goal + 1))
    
    # Calcola la probabilità di ogni combinazione di risultati usando la distribuzione di Poisson
    for goal_casa in range(max_goal + 1):
        for goal_trasferta in range(max_goal + 1):
            prob_casa = poisson.pmf(goal_casa, media_goal_casa)
            prob_trasferta = poisson.pmf(goal_trasferta, media_goal_trasferta)
            probabilita_percentuale = prob_casa * prob_trasferta * 100  # Converti in percentuale
            matrice_risultati.loc[goal_casa, goal_trasferta] = round(probabilita_percentuale, 2)  # Arrotonda a due decimali

    # Etichetta per la matrice dei risultati
    matrice_risultati.index.name = f"Goal {squadra_casa}"
    matrice_risultati.columns.name = f"Goal {squadra_trasferta}"
    
    return matrice_risultati

# Carica i dati
file_path = 'I1.json'  # Cambia con il percorso del tuo file JSON
df = pd.read_json(file_path)

# Esempio di utilizzo per Juventus e Inter
squadra_casa = 'Inter'
squadra_trasferta = 'Juventus'
matrice_risultati = calcola_matrice_risultati(df, squadra_casa, squadra_trasferta)

# Visualizza la matrice dei risultati
print("Matrice di probabilità dei risultati (in %):")
print(matrice_risultati)
