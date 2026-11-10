# Hedging competition

## Esecuzione

Richiede Python 3.11 o successivo. Da PowerShell, nella cartella dello script:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

I dati originali restano locali. Se il Parquet e il file Excel sono nella
cartella superiore, avviare `main.py --data-dir ..` con lo stesso interprete.

## Preprocessing

Le due funzioni sono in `preprocessing/options.py`.

- `filter_option_quotes` seleziona opzioni in vita con bid e ask validi,
  mid di almeno 0,05 USD/barile e spread relativo al massimo del 50%.
- `filter_smile_quotes` verifica tipo di opzione, strike, prezzo del future e
  limiti del settlement usato per il calcolo della volatilità implicita.

`prepare_market_data` restituisce due DataFrame. `quotes` contiene le opzioni
alla data di emissione che passano entrambi i filtri. `spot` contiene lo
storico disponibile dei futures COJ21 (Excel) e COK21 (Parquet), con le colonne
`date`, `future_ticker`, `settlement`, `bid` e `ask`. Le osservazioni identiche
vengono accorpate. Bid e ask diversi per la stessa data restano visibili.

`main.py` chiama in sequenza la preparazione dei dati, il calendario,
gli smile, la simulazione e il calcolo del collar. Il calendario usa
i giorni feriali e la simulazione usa una correlazione fissata in `main.py`.

Per verificare i filtri sui dati sintetici, usare
`python -m unittest discover -s tests -v` con l'interprete dell'ambiente.
