# Import delle librerie necessarie
import pandas as pd
import json
import numpy as np
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
import logging

# Configurazione del logging per tracciare il progresso del codice
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per calcolare la media dei goal segnati
def calcola_media_goal(dataframe_partite, nome_squadra):
    goal_segnati_in_casa = dataframe_partite[dataframe_partite['HomeTeam'] == nome_squadra]['FTHG']
    goal_segnati_in_trasferta = dataframe_partite[dataframe_partite['AwayTeam'] == nome_squadra]['FTAG']
    
    if len(goal_segnati_in_casa) == 0 or len(goal_segnati_in_trasferta) == 0:
        media_goal_casa = dataframe_partite['FTHG'].mean()
        media_goal_trasferta = dataframe_partite['FTAG'].mean()
        logging.warning(f"Non ci sono dati sufficienti per la squadra {nome_squadra}. Utilizzando media globale.")
    else:
        media_goal_casa = goal_segnati_in_casa.mean()
        media_goal_trasferta = goal_segnati_in_trasferta.mean()
    
    return media_goal_casa, media_goal_trasferta

# Funzione per calcolare le probabilità di esiti utilizzando Poisson
def calcola_probabilita_esiti(dataframe_partite, nome_squadra_casa, nome_squadra_trasferta, max_goal=5):
    media_goal_casa, _ = calcola_media_goal(dataframe_partite, nome_squadra_casa)
    _, media_goal_trasferta = calcola_media_goal(dataframe_partite, nome_squadra_trasferta)
    
    if np.isnan(media_goal_casa) or np.isnan(media_goal_trasferta):
        logging.warning(f"Dati insufficienti per {nome_squadra_casa} vs {nome_squadra_trasferta}.")
        return 33.33, 33.33, 33.33
    
    prob_vittoria_in_casa, prob_pareggio, prob_vittoria_in_trasferta = 0, 0, 0
    for goal_casa in range(max_goal + 1):
        for goal_trasferta in range(max_goal + 1):
            prob_goal_casa = poisson.pmf(goal_casa, media_goal_casa)
            prob_goal_trasferta = poisson.pmf(goal_trasferta, media_goal_trasferta)

            if goal_casa > goal_trasferta:
                prob_vittoria_in_casa += prob_goal_casa * prob_goal_trasferta
            elif goal_casa == goal_trasferta:
                prob_pareggio += prob_goal_casa * prob_goal_trasferta
            else:
                prob_vittoria_in_trasferta += prob_goal_casa * prob_goal_trasferta

    return prob_vittoria_in_casa * 100, prob_pareggio * 100, prob_vittoria_in_trasferta * 100

# Caricamento dei dati delle partite
file_path_partite = [
    './data/IT/A/ITA-2021.json',
    './data/IT/A/ITA-2122.json',
    './data/IT/A/ITA-2223.json',
    './data/IT/A/ITA-2324.json',
    './data/IT/A/ITA-2425.json',  # Nuova stagione Serie A
    './data/IT/B/ITB-2021.json',
    './data/IT/B/ITB-2122.json',
    './data/IT/B/ITB-2223.json',
    './data/IT/B/ITB-2324.json'   # Ultima stagione Serie B disponibile
]

# Unione dei dati delle partite in un unico DataFrame
lista_dataframes_partite = []
for file_path in file_path_partite:
    with open(file_path, 'r') as file:
        partite = json.load(file)
        dataframe_parziale = pd.DataFrame(partite)
        lista_dataframes_partite.append(dataframe_parziale)
dataframe_partite_completo = pd.concat(lista_dataframes_partite, ignore_index=True)

# Creazione della colonna 'Outcome'
dataframe_partite_completo['Outcome'] = dataframe_partite_completo['FTR'].apply(lambda risultato: 1 if risultato == 'H' else 0)

# Creazione delle feature con le probabilità di vittoria, pareggio, sconfitta
dataframe_partite_completo[['Probabilita_VittoriaInCasa', 'Probabilita_Pareggio', 'Probabilita_VittoriaInTrasferta']] = dataframe_partite_completo.apply(
    lambda riga: pd.Series(calcola_probabilita_esiti(dataframe_partite_completo, riga['HomeTeam'], riga['AwayTeam'])),
    axis=1
)

# Selezione delle feature e separazione variabili indipendenti (X) e target (y)
nomi_feature = [
    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'B365H', 'B365A', 'B365D', 
    'AvgH', 'AvgA', 'AvgD', 'P>2.5', 'P<2.5', 'MaxH', 'MaxA', 'MaxD', 
    'Probabilita_VittoriaInCasa', 'Probabilita_Pareggio', 'Probabilita_VittoriaInTrasferta'
]
variabili_indipendenti = dataframe_partite_completo[nomi_feature]
variabile_target = dataframe_partite_completo['Outcome']

# Gestione dei valori NaN
variabili_indipendenti = variabili_indipendenti.apply(pd.to_numeric, errors='coerce')
variabili_indipendenti.fillna(variabili_indipendenti.mean(), inplace=True)

# Divisione train-test
variabili_indipendenti_train, variabili_indipendenti_test, variabile_target_train, variabile_target_test = train_test_split(
    variabili_indipendenti, variabile_target, test_size=0.3, random_state=42)

# Standardizzazione delle feature
scaler_standard = StandardScaler()
variabili_indipendenti_train_scalate = scaler_standard.fit_transform(variabili_indipendenti_train)
variabili_indipendenti_test_scalate = scaler_standard.transform(variabili_indipendenti_test)

# Addestramento dei modelli
def addestra_modello(modello, variabili_indipendenti_train, variabile_target_train, nome_modello):
    inizio_tempo = time.time()
    modello.fit(variabili_indipendenti_train, variabile_target_train)
    fine_tempo = time.time()
    durata_addestramento = fine_tempo - inizio_tempo
    logging.info(f"{nome_modello} - Durata dell'addestramento: {durata_addestramento:.2f} secondi")
    return modello

# Definizione dei modelli
modello_regressione_logistica = LogisticRegression(solver='liblinear')
modello_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
modello_gradient_boosting = GradientBoostingClassifier()

# Addestramento
modello_regressione_logistica = addestra_modello(modello_regressione_logistica, variabili_indipendenti_train_scalate, variabile_target_train, "Regressione Logistica")
modello_random_forest = addestra_modello(modello_random_forest, variabili_indipendenti_train_scalate, variabile_target_train, "Random Forest")
modello_gradient_boosting = addestra_modello(modello_gradient_boosting, variabili_indipendenti_train_scalate, variabile_target_train, "Gradient Boosting")

# Previsione sugli esiti del test set e valutazione dell'accuratezza
previsioni_logreg = modello_regressione_logistica.predict(variabili_indipendenti_test_scalate)
previsioni_rf = modello_random_forest.predict(variabili_indipendenti_test_scalate)
previsioni_gb = modello_gradient_boosting.predict(variabili_indipendenti_test_scalate)

# Accuracy
accuratezza_logreg = accuracy_score(variabile_target_test, previsioni_logreg)
accuratezza_rf = accuracy_score(variabile_target_test, previsioni_rf)
accuratezza_gb = accuracy_score(variabile_target_test, previsioni_gb)
accuratezza_df = pd.DataFrame({
    'Modello': ['Regressione Logistica', 'Random Forest', 'Gradient Boosting'],
    'Accuratezza': [accuratezza_logreg, accuratezza_rf, accuratezza_gb]
})
accuratezza_df.sort_values(by='Accuratezza', ascending=False, inplace=True)

accuratezza_df
