import json
import pandas as pd

# Funzione per caricare i dati da un file JSON
def load_data_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Funzione per creare la griglia con i dati
def create_match_grid(data):
    # Creazione di una lista per contenere i dati delle partite
    matches_data = []
    
    for match in data["matches"]:
        round = match.get("round", "N/A")
        date = match.get("date", "N/A")
        time = match.get("time", "N/A")
        team1 = match.get("team1", "N/A")
        team2 = match.get("team2", "N/A")
        
        # Recupero dei punteggi parziali e finali in modo sicuro
        ht_score = "N/A"
        ft_score = "N/A"
        
        if "score" in match:
            score = match["score"]
            ht_score = "-".join(map(str, score.get("ht", ["N/A", "N/A"])))
            ft_score = "-".join(map(str, score.get("ft", ["N/A", "N/A"])))
        
        matches_data.append([round, date, time, team1, team2, ht_score, ft_score])

    # Creazione di un DataFrame con pandas per organizzare i dati in una tabella
    df = pd.DataFrame(matches_data, columns=["Round", "Date", "Time", "Team 1", "Team 2", "HT Score", "FT Score"])

    return df

# Carica i dati dal file JSON
filename = "serieb23-24.json"  # Sostituisci con il nome del tuo file JSON
data = load_data_from_file(filename)

# Crea la griglia
df = create_match_grid(data)

# Stampa la griglia (tabella)
print(df)
