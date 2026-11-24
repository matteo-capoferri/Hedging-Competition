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

Il backtest include i fixing SOFR dal 3 gennaio 2020 al 26 febbraio 2021 in
`data/SOFR.csv` ([serie SOFR, New York Fed tramite FRED](https://fred.stlouisfed.org/series/SOFR)).
La cassa matura al tasso realizzato ACT/360. Per scontare il collar fino
all'ultimo fixing si assume una curva piatta all'ultimo tasso già pubblicato,
con capitalizzazione giornaliera. Per un altro periodo occorre estendere
il CSV includendo almeno il fixing disponibile all'emissione.

Le due funzioni sono in `preprocessing/options.py`.

- `filter_option_quotes` seleziona opzioni in vita con bid e ask validi,
  mid di almeno 0,05 USD/barile e spread relativo al massimo del 50%.
- `filter_smile_quotes` verifica tipo di opzione, strike, prezzo del future e
  limiti del settlement usato per il calcolo della volatilità implicita.

`prepare_market_data` restituisce due DataFrame. `quotes` contiene le opzioni
su tutte le date disponibili che passano entrambi i filtri. `spot` contiene lo
storico disponibile dei futures COJ21 (Excel) e COK21 (Parquet), con le colonne
`date`, `future_ticker`, `settlement`, `bid` e `ask`. Le osservazioni identiche
vengono accorpate. Bid o ask discordanti per la stessa data diventano mancanti;
un'operazione che richiede quel prezzo viene saltata, mantenendo la posizione precedente.

`main.py` chiama in sequenza la preparazione dei dati, il calendario,
gli smile, la simulazione e il calcolo del collar. Il calendario usa
i giorni feriali; la correlazione e' ristimata con i dati disponibili a ogni seduta.

Il backtest usa il moment matching e il suo strike zero-cost; lo strike Monte Carlo
rimane un confronto separato. Il valore iniziale MM deve essere inferiore a un
centesimo di USD in valore assoluto sul nozionale di un milione di barili.
Il patrimonio prima dell'apertura e' zero e un eventuale valore iniziale del collar
entra nel P&L. La riconciliazione giornaliera tra P&L per componenti e variazione
del patrimonio e' controllata automaticamente. All'ultima data il payoff rimane
nel valore del collar e non viene anche accreditato in cassa.

`main.py` salva in `output/backtest` il CSV giornaliero, le dieci sedute con P&L
quadratico maggiore, un riepilogo JSON e un grafico con patrimonio, P&L, P&L²
e posizioni. La destinazione si cambia con `--output-dir`.
Le diagnostiche dettagliate degli ordini (obiettivi, esecuzioni e operazioni saltate)
sono raccolte solo nei test contabili, senza appesantire il ciclo del backtest.
`check_accounting` monitora invece ogni esecuzione reale e restituisce la colonna
`accounting_error`, riepilogata anche nel report.

Per verificare pricing, sensibilita', filtri e contabilità su dati sintetici, usare
`python -m unittest discover -s tests -v` con l'interprete dell'ambiente.
