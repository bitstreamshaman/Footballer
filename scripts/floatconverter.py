import json

# Funzione per sostituire le virgole con i punti nei numeri decimali
def replace_commas_and_convert(data):
    if isinstance(data, dict):
        # Se il valore è un dizionario, applichiamo ricorsivamente la funzione
        return {key: replace_commas_and_convert(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Se il valore è una lista, applichiamo ricorsivamente la funzione a ogni elemento
        return [replace_commas_and_convert(item) for item in data]
    elif isinstance(data, str):
        # Se il valore è una stringa, controlliamo se contiene una virgola
        try:
            # Proviamo a convertirlo in un numero
            converted_value = float(data.replace(',', '.'))
            return converted_value
        except ValueError:
            # Se non è un numero valido, lasciamo la stringa così com'è
            return data
    else:
        return data

# Carica il file JSON
with open('./data/IT/ITA-2425.json', 'r') as f:
    data = json.load(f)

# Applichiamo la funzione per sostituire le virgole con i punti e convertire in numeri
data_modified = replace_commas_and_convert(data)

# Salva il nuovo file JSON con i valori modificati
with open('./data/IT/ITA-2425.json', 'w') as f:
    json.dump(data_modified, f, indent=4)

print("I separatori decimali sono stati modificati correttamente e i numeri sono stati convertiti.")
