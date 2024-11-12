import pandas as pd
import json
import numpy as np
from scipy.stats import poisson
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
import logging

# Configurazione del logging per tracciare il progresso del codice
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per calcolare la media dei goal segnati in casa e in trasferta da una squadra
def calcola_media_goal(dataframe_partite, nome_squadra):
    """
    Calcola la media dei goal segnati dalla squadra in casa e in trasferta.
    Se i dati non sono sufficienti, utilizza la media globale dei goal.
    """
    # Filtra i goal segnati in casa e in trasferta per la squadra
    goal_segnati_in_casa = dataframe_partite[dataframe_partite['HomeTeam'] == nome_squadra]['FTHG']
    goal_segnati_in_trasferta = dataframe_partite[dataframe_partite['AwayTeam'] == nome_squadra]['FTAG']
    
    # Se non ci sono goal in casa o in trasferta, usa la media globale
    if len(goal_segnati_in_casa) == 0 or len(goal_segnati_in_trasferta) == 0:
        media_goal_casa = dataframe_partite['FTHG'].mean()
        media_goal_trasferta = dataframe_partite['FTAG'].mean()
        logging.warning(f"Non ci sono dati sufficienti per la squadra {nome_squadra}. Utilizzando media globale.")
    else:
        media_goal_casa = goal_segnati_in_casa.mean()
        media_goal_trasferta = goal_segnati_in_trasferta.mean()
    
    return media_goal_casa, media_goal_trasferta

# Funzione per calcolare le probabilità di vittoria in casa, pareggio e vittoria in trasferta
# utilizzando la distribuzione di Poisson basata sulla media dei goal
def calcola_probabilita_esiti(dataframe_partite, nome_squadra_casa, nome_squadra_trasferta, max_goal=5):
    """
    Calcola le probabilità di vittoria in casa, pareggio e vittoria in trasferta
    utilizzando la distribuzione di Poisson per prevedere il numero di goal.
    """
    # Calcola le medie dei goal per entrambe le squadre
    media_goal_casa, _ = calcola_media_goal(dataframe_partite, nome_squadra_casa)
    _, media_goal_trasferta = calcola_media_goal(dataframe_partite, nome_squadra_trasferta)
    
    # Se non sono disponibili dati sufficienti, ritorna probabilità uguali
    if np.isnan(media_goal_casa) or np.isnan(media_goal_trasferta):
        logging.warning(f"Non è possibile calcolare la probabilità per {nome_squadra_casa} vs {nome_squadra_trasferta}. Dati insufficienti.")
        return 33.33, 33.33, 33.33
    
    prob_vittoria_in_casa = 0
    prob_pareggio = 0
    prob_vittoria_in_trasferta = 0

    # Calcola le probabilità di ogni possibile risultato
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

# Funzione per determinare l'esito effettivo della partita
def esito_effettivo(outcome):
    if outcome == 1:
        return 'VittoriaCasa'
    elif outcome == 0:
        return 'SconfittaCasa'
    else:
        return 'Pareggio'


# Caricamento dei dati delle partite
file_path_partite = [
    './data/IT/A/ITA-2021.json',
    './data/IT/A/ITA-2122.json',
    './data/IT/A/ITA-2223.json',
    './data/IT/A/ITA-2324.json',
    './data/IT/A/ITA-2425.json',  # Nuova stagione Serie A
    #'./data/IT/B/ITB-2223.json',
    #'./data/IT/B/ITB-2324.json'
    #'./data/IT/B/ITB-2021.json',
    #'./data/IT/B/ITB-2122.json',
   # Ultima stagione Serie B disponibile
]

# Unione dei dati delle partite in un unico DataFrame
lista_dataframes_partite = []
for file_path in file_path_partite:
    with open(file_path, 'r') as file:
        partite = json.load(file)
        dataframe_parziale = pd.DataFrame(partite)
        lista_dataframes_partite.append(dataframe_parziale)
dataframe_partite_completo = pd.concat(lista_dataframes_partite, ignore_index=True)

# Creazione della colonna 'Outcome' che rappresenta il risultato della partita (1 per vittoria in casa, 0 per sconfitta)
dataframe_partite_completo['Outcome'] = dataframe_partite_completo['FTR'].apply(lambda risultato: 1 if risultato == 'H' else 0)

# Selezione delle feature (variabili indipendenti) per il modello predittivo
nomi_feature = [
    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'B365H', 'B365A', 
    'B365D', 'AvgH', 'AvgA', 'AvgD', 'P>2.5', 'P<2.5', 'MaxH', 'MaxA', 'MaxD'
]

# Aggiunta delle probabilità calcolate con Poisson come nuove feature
dataframe_partite_completo[['Probabilita_VittoriaInCasa', 'Probabilita_Pareggio', 'Probabilita_VittoriaInTrasferta']] = dataframe_partite_completo.apply(
    lambda riga: pd.Series(calcola_probabilita_esiti(dataframe_partite_completo, riga['HomeTeam'], riga['AwayTeam'])),
    axis=1
)
nomi_feature.extend(['Probabilita_VittoriaInCasa', 'Probabilita_Pareggio', 'Probabilita_VittoriaInTrasferta'])

# Separazione delle features (X) e del target (y)
variabili_indipendenti = dataframe_partite_completo[nomi_feature]
variabile_target = dataframe_partite_completo['Outcome']

# Gestione dei valori non numerici e valori mancanti (NaN) nelle features
variabili_indipendenti = variabili_indipendenti.apply(pd.to_numeric, errors='coerce')
variabili_indipendenti.fillna(variabili_indipendenti.mean(), inplace=True)

# Suddivisione dei dati in training set (70%) e test set (30%)
variabili_indipendenti_train, variabili_indipendenti_test, variabile_target_train, variabile_target_test = train_test_split(
    variabili_indipendenti, variabile_target, test_size=0.3, random_state=42)

# Conservazione dei dati originali di HomeTeam e AwayTeam per il test set
dataframe_test_partite = dataframe_partite_completo.loc[variabili_indipendenti_test.index, ['HomeTeam', 'AwayTeam', 'Outcome']]

# Standardizzazione delle feature (scaling)
scaler_standard = StandardScaler()
variabili_indipendenti_train_scalate = scaler_standard.fit_transform(variabili_indipendenti_train)
variabili_indipendenti_test_scalate = scaler_standard.transform(variabili_indipendenti_test)

# Funzione per addestrare un modello e calcolare il tempo di addestramento
def addestra_modello(modello, variabili_indipendenti_train, variabile_target_train, nome_modello, grid_search=False):
    """
    Allena un modello e calcola il tempo impiegato per l'addestramento.
    Se grid_search è True, esegue la ricerca con GridSearchCV.
    """
    inizio_tempo = time.time()  # Inizio tempo

    if grid_search:
        modello.fit(variabili_indipendenti_train, variabile_target_train)
    else:
        modello.fit(variabili_indipendenti_train, variabile_target_train)
    
    fine_tempo = time.time()  # Fine tempo
    durata_addestramento = fine_tempo - inizio_tempo

    logging.info(f"{nome_modello} - Durata dell'addestramento: {durata_addestramento:.2f} secondi")
    return modello

# Definizione dei modelli
modello_regressione_logistica = LogisticRegression(solver='liblinear')
modello_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
modello_gradiant_boosting = GradientBoostingClassifier()

# Addestramento dei modelli
modello_regressione_logistica = addestra_modello(modello_regressione_logistica, variabili_indipendenti_train_scalate, variabile_target_train, "Regressione Logistica")
modello_random_forest = addestra_modello(modello_random_forest, variabili_indipendenti_train_scalate, variabile_target_train, "Random Forest")
modello_gradiant_boosting = addestra_modello(modello_gradiant_boosting, variabili_indipendenti_train_scalate, variabile_target_train, "Gradient Boosting")

# Previsione sugli esiti del test set
previsioni_logreg = modello_regressione_logistica.predict(variabili_indipendenti_test_scalate)
previsioni_rf = modello_random_forest.predict(variabili_indipendenti_test_scalate)
previsioni_gb = modello_gradiant_boosting.predict(variabili_indipendenti_test_scalate)

# Creazione del dataframe per la stampa finale
probabilita_df = dataframe_test_partite.copy()
probabilita_df['EsitoPrevisioneLogreg'] = previsioni_logreg
probabilita_df['EsitoPrevisioneRF'] = previsioni_rf
probabilita_df['EsitoPrevisioneGB'] = previsioni_gb

# Aggiungiamo la colonna per l'esito effettivo della partita
probabilita_df['EsitoEffettivo'] = dataframe_test_partite['Outcome'].apply(esito_effettivo)

# Stampa finale
print("\nProbabilità degli esiti delle partite con previsioni ed esito effettivo:")
print(probabilita_df)

# Calcolo delle accuratezze per ciascun modello
accuratezza_logreg = accuracy_score(variabile_target_test, previsioni_logreg)
accuratezza_rf = accuracy_score(variabile_target_test, previsioni_rf)
accuratezza_gb = accuracy_score(variabile_target_test, previsioni_gb)

# Creazione di un DataFrame per contenere le accuratezze
accuratezza_df = pd.DataFrame({
    'Modello': ['Regressione Logistica', 'Random Forest', 'Gradient Boosting'],
    'Accuratezza': [accuratezza_logreg, accuratezza_rf, accuratezza_gb]
})

# Ordinamento del DataFrame per accuratezza (dal migliore al peggiore)
accuratezza_df = accuratezza_df.sort_values(by='Accuratezza', ascending=False)

# Calcolo del modello combinato con majority voting per il test set
previsioni_combinate = np.array([previsioni_logreg, previsioni_rf, previsioni_gb])
previsioni_majority_voting = np.apply_along_axis(lambda x: 1 if np.sum(x) >= 2 else 0, axis=0, arr=previsioni_combinate)

# Aggiunta delle previsioni del modello combinato al DataFrame
probabilita_df['EsitoPrevisioneCombinato'] = previsioni_majority_voting

# Calcolo dell'accuratezza del modello combinato
accuratezza_comb = accuracy_score(variabile_target_test, previsioni_majority_voting)

# Creazione di un DataFrame per contenere le accuratezze
accuratezza_df = pd.DataFrame({
    'Modello': ['Regressione Logistica', 'Random Forest', 'Gradient Boosting'],
    'Accuratezza': [accuratezza_logreg, accuratezza_rf, accuratezza_gb]
})

# Se vuoi aggiungere una nuova riga per il modello combinato (majority voting)
accuratezza_df = pd.concat([accuratezza_df, pd.DataFrame([{
    'Modello': 'Majority Voting',
    'Accuratezza': accuratezza_comb
}])], ignore_index=True)

# Ordinamento del DataFrame per accuratezza (dal migliore al peggiore)
accuratezza_df = accuratezza_df.sort_values(by='Accuratezza', ascending=False)


# Stampa del DataFrame con le accuratezze aggiornate
print("\nAccuratezza dei modelli:")
print(accuratezza_df)

import pandas as pd

# Funzione che determina l'esito di una singola partita usando i modelli
def determina_esito_partita(nome_squadra_casa, nome_squadra_trasferta):
    """
    Determina l'esito della partita tra due squadre utilizzando i modelli addestrati.
    """
    # Calcolo delle probabilità per le due squadre (casa vs trasferta)
    prob_vittoria_casa, prob_pareggio, prob_vittoria_trasferta = calcola_probabilita_esiti(
        dataframe_partite_completo, nome_squadra_casa, nome_squadra_trasferta
    )
    
    # Creazione delle features per la partita corrente
    features_partita = {
        'HS': 0, 'AS': 0, 'HST': 0, 'AST': 0, 'HF': 0, 'AF': 0, 'HC': 0, 'AC': 0, 'B365H': 0, 'B365A': 0, 
        'B365D': 0, 'AvgH': 0, 'AvgA': 0, 'AvgD': 0, 'P>2.5': 0, 'P<2.5': 0, 'MaxH': 0, 'MaxA': 0, 'MaxD': 0,
        'Probabilita_VittoriaInCasa': prob_vittoria_casa,
        'Probabilita_Pareggio': prob_pareggio,
        'Probabilita_VittoriaInTrasferta': prob_vittoria_trasferta
    }
    
    # Creazione del DataFrame delle features per la partita
    features_df = pd.DataFrame([features_partita])
    
    # Standardizzazione delle features
    features_scalate = scaler_standard.transform(features_df)
    
    # Previsioni dai modelli
    previsione_logreg = modello_regressione_logistica.predict(features_scalate)[0]
    previsione_rf = modello_random_forest.predict(features_scalate)[0]
    previsione_gb = modello_gradiant_boosting.predict(features_scalate)[0]
    
    return previsione_logreg, previsione_rf, previsione_gb

# Variabili di input
nome_squadra_casa = 'Cagliari'#input("Inserisci squadra in casa: ")
nome_squadra_trasferta = 'Milan'#input("Inserisci squadra in trasferta: ")

# Lista per raccogliere i risultati delle 10 simulazioni per ciascun modello
risultati_logreg = []
risultati_rf = []
risultati_gb = []

# Esegui determina_esito_partita 10 volte per ogni modello
for _ in range(10):
    previsione_logreg, previsione_rf, previsione_gb = determina_esito_partita(nome_squadra_casa, nome_squadra_trasferta)
    
    # Aggiungi i risultati di ciascun modello alle rispettive liste
    risultati_logreg.append(previsione_logreg)
    risultati_rf.append(previsione_rf)
    risultati_gb.append(previsione_gb)

# Calcola la media dei risultati per ogni modello (la media del risultato su 10 partite)
media_logreg = sum(risultati_logreg) / len(risultati_logreg)
media_rf = sum(risultati_rf) / len(risultati_rf)
media_gb = sum(risultati_gb) / len(risultati_gb)

# Determina l'esito finale per ogni modello in base alla media
esito_logreg = 1 if media_logreg >= 0.5 else 0
esito_rf = 1 if media_rf >= 0.5 else 0
esito_gb = 1 if media_gb >= 0.5 else 0

# Calcolo del Majority Voting tra le medie dei 3 modelli
majority_voting = 1 if (esito_logreg + esito_rf + esito_gb) >= 2 else 0

# Determinazione dell'esito finale
esito_finale = 'Vittoria' if majority_voting == 1 else 'Sconfitta/Pareggio'

# Stampa dei risultati
print(f"Partita: {nome_squadra_casa} vs {nome_squadra_trasferta}")

# Stampa dei risultati di ogni modello e delle medie
print(f"\nEsiti delle simulazioni per ogni modello:")
print(f"LogReg: {media_logreg:.2f} ({'Vittoria' if esito_logreg == 1 else 'Sconfitta/Pareggio'})")
print(f"Random Forest: {media_rf:.2f} ({'Vittoria' if esito_rf == 1 else 'Sconfitta/Pareggio'})")
print(f"Gradient Boosting: {media_gb:.2f} ({'Vittoria' if esito_gb == 1 else 'Sconfitta/Pareggio'})")

# Stampa dell'esito finale basato sul Majority Voting
print(f"\nEsito finale dopo 10 simulazioni per ciascun modello (Majority Voting tra i modelli): {esito_finale}")

# Stampa del risultato finale basato sulla media del majority voting
print(f"\nEsito finale dopo 10 simulazioni (media modello combinato Majority Voting): {esito_finale}")
# Caricamento dei dati dal file ITA-2425.json

file_path_partite_nuove = './data/IT/A/ITA-2425.json'

# Caricamento del JSON e creazione del DataFrame
with open(file_path_partite_nuove, 'r') as file:
    partite_nuove = json.load(file)

dataframe_partite_nuove = pd.DataFrame(partite_nuove)

# Creazione della colonna 'Outcome' per i nuovi dati
dataframe_partite_nuove['Outcome'] = dataframe_partite_nuove['FTR'].apply(lambda risultato: 1 if risultato == 'H' else 0)

# Aggiunta delle probabilità calcolate con Poisson come nuove feature
dataframe_partite_nuove[['Probabilita_VittoriaInCasa', 'Probabilita_Pareggio', 'Probabilita_VittoriaInTrasferta']] = dataframe_partite_nuove.apply(
    lambda riga: pd.Series(calcola_probabilita_esiti(dataframe_partite_completo, riga['HomeTeam'], riga['AwayTeam'])),
    axis=1
)

# Previsioni per le nuove partite utilizzando i modelli addestrati
variabili_indipendenti_nuove = dataframe_partite_nuove[nomi_feature]

# Gestione dei valori non numerici e valori mancanti (NaN) nelle features
variabili_indipendenti_nuove = variabili_indipendenti_nuove.apply(pd.to_numeric, errors='coerce')
variabili_indipendenti_nuove.fillna(variabili_indipendenti_nuove.mean(), inplace=True)

# Standardizzazione delle nuove partite
variabili_indipendenti_nuove_scalate = scaler_standard.transform(variabili_indipendenti_nuove)

# Previsioni dei modelli per le nuove partite
previsioni_logreg_nuove = modello_regressione_logistica.predict(variabili_indipendenti_nuove_scalate)
previsioni_rf_nuove = modello_random_forest.predict(variabili_indipendenti_nuove_scalate)
previsioni_gb_nuove = modello_gradiant_boosting.predict(variabili_indipendenti_nuove_scalate)

# Creazione del DataFrame con i risultati delle nuove partite
nuovi_risultati_df = dataframe_partite_nuove.copy()
nuovi_risultati_df['EsitoPrevisioneLogreg'] = previsioni_logreg_nuove
nuovi_risultati_df['EsitoPrevisioneRF'] = previsioni_rf_nuove
nuovi_risultati_df['EsitoPrevisioneGB'] = previsioni_gb_nuove

# Aggiungiamo la colonna per l'esito effettivo della partita
nuovi_risultati_df['EsitoEffettivo'] = nuovi_risultati_df['Outcome'].apply(esito_effettivo)

# Calcolo della previsione combinata (majority voting) per le nuove partite
previsioni_combinate_nuove = np.array([previsioni_logreg_nuove, previsioni_rf_nuove, previsioni_gb_nuove])
previsioni_majority_voting_nuove = np.apply_along_axis(lambda x: 1 if np.sum(x) >= 2 else 0, axis=0, arr=previsioni_combinate_nuove)

# Aggiunta della previsione combinata al DataFrame
nuovi_risultati_df['EsitoPrevisioneCombinato'] = previsioni_majority_voting_nuove

# Stampa finale dei risultati
print("\nRisultati delle nuove partite con le previsioni dei modelli e l'esito effettivo:")
print(nuovi_risultati_df)


