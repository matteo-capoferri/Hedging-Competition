# Hedging competition

Il progetto riconcilia le quotazioni Brent da Excel e Parquet, costruisce i
fixing del collar asiatico e stima lo strike put zero-cost all'emissione.

## Avvio in VS Code

Apri la cartella che contiene `main.py` (in questa copia è `.github-upload`).
Servono Python 3.11 o successivo e le dipendenze in `requirements.txt`.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

I comandi cercano i due file dati nella cartella del codice e poi in quella
superiore. Usa `--data-dir` per indicare una cartella diversa. Excel, Parquet e
materiali del corso sono esclusi da Git.

## Consultare i dati

```powershell
.\.venv\Scripts\python.exe inspect_market.py --date 2020-01-06 --contract COJ21
.\.venv\Scripts\python.exe inspect_market.py --as-of 2021-02-24
.\.venv\Scripts\python.exe inspect_market.py --output-dir output/market_view
```

| CSV | Contenuto |
| --- | --- |
| `futures.csv` | Una riga per data e future, con settlement, bid/ask, provenienza e stato della coppia |
| `opzioni.csv` | Quotazioni originali, con ticker, sottostante, scadenza, strike e prezzi nelle prime colonne |
| `fixing.csv` | Data, future, peso, stato, settlement osservato e contributo a `A` |
| `sofr.csv` | Osservazione SOFR grezza per data |

`--date` filtra le quotazioni e, in assenza di `--as-of`, imposta anche
la data di osservazione dei fixing. `--as-of` permette di separare le due date.
Senza filtri, l'export include la storia completa delle quotazioni disponibili.
La vista dei fixing usa la chiusura dell'ultimo fixing in assenza di una data.
In Excel, usa **Dati → Da testo/CSV** e scegli la virgola come delimitatore.

## Come funziona il calcolo

1. `market_data.py` valida le chiavi e riconcilia i settlement. Per il bid/ask
   usa una coppia realmente osservata. Se le fonti non permettono una scelta
   univoca, marca la riga come `ambiguous` e conserva il settlement.
2. `build_calendar.py` assegna a ciascun fixing un ticker e un peso.
   `MarketData` calcola il contributo già noto e conserva i fixing residui.
3. `option_analytics.py` filtra le opzioni e usa `black76.py` per ricavare le
   IV dai mid. Il pricing interpola la volatilità ATM separatamente per
   ciascun future.
4. `simulate_paths.py` simula i due futures con shock correlati. Per ogni
   scenario, `A` somma i prezzi osservati e simulati con i pesi originali.
5. `price_collar.py` cerca lo strike put che uguaglia il prezzo della call a
   strike 78. `main.py` rivaluta lo strike su un campione indipendente.

```powershell
.\.venv\Scripts\python.exe main.py --correlation 0.8
.\.venv\Scripts\python.exe first_steps.py --date 2020-01-06
.\.venv\Scripts\python.exe -m unittest discover -v
```

`first_steps.py` produce in `output/initial_analysis` la diagnostica sulle
quotazioni, le IV OTM, la copertura degli smile e la correlazione mobile.
Per il grafico delle IV in `main.py`, usa `--plot` con matplotlib installato.

## Mappa dei file Python

| File | Ruolo | Cosa fa |
| --- | --- | --- |
| `main.py` | Pricing | Coordina il pricing all'emissione. Legge i dati della seduta, prepara gli input, calcola lo strike zero-cost e lo rivaluta su una seconda simulazione. |
| `inspect_market.py` | Consultazione | Mostra futures, opzioni, SOFR e stato dei fixing. Filtra le quotazioni ed esporta quattro CSV. |
| `first_steps.py` | Diagnostica | Analizza il training ed esporta IV giornaliere, copertura degli smile, rendimenti, correlazione mobile e discordanze bid/ask. |
| `market_data.py` | Dati | Legge Excel e Parquet, riconcilia le quotazioni e risolve i fixing già osservati. Calcola il valore di `A` per ogni scenario sommando i contributi osservati e simulati. |
| `build_calendar.py` | Calendario | Assegna ticker e peso a ogni data di fixing applicando la regola di roll. Contiene il calendario provvisorio di training. |
| `option_analytics.py` | Analisi delle opzioni | Applica i filtri di qualità e liquidità, prepara i punti IV e interpola la volatilità ATM per ciascun future. |
| `black76.py` | Modello vanilla | Calcola il prezzo Black-76 non scontato e inverte la formula per ottenere la volatilità implicita. |
| `simulate_paths.py` | Simulazione | Genera i percorsi correlati dei due futures, con una colonna per fixing residuo. |
| `price_collar.py` | Payoff e strike | Valuta il collar dal vettore dei valori simulati di `A` e trova lo strike put zero-cost. |
| `plot_smiles.py` | Grafico facoltativo | Disegna i punti strike/IV quando si avvia `main.py --plot`. |
| `extract_smile.py` | Interfaccia storica | Estrae gli smile dai settlement delle opzioni. Il pricing attuale usa le IV ricavate dai mid in `option_analytics.py`. |
| `find_K.py` | Interfaccia storica | Accetta le due matrici di percorsi come nel prototipo e delega la ricerca dello strike a `price_collar.py`. |
| `get_initial_price.py` | Interfaccia storica | Estrae un settlement univoco da un gruppo di quotazioni. Il pricing attuale legge il settlement dalla tabella futures riconciliata. |
| `tests/__init__.py` | Test | Permette a `unittest discover` di trovare la suite dalla radice del repository. |
| `tests/test_first_steps.py` | Test | Controlla Black-76, inversione della IV, filtri e unione dello storico dei futures. |
| `tests/test_market_data.py` | Test | Controlla la riconciliazione bid/ask, il roll e la separazione tra fixing osservati e futuri. |
| `test_first_steps.py` | Avvio storico dei test | Mantiene il comando precedente e richiama la suite corrispondente in `tests/`. |

Per il prezzo iniziale basta `main.py`. Tieni `first_steps.py` come comando
separato per verificare il campione di training e spiegare i dati nel report.
La correlazione mobile che esporta serve alla diagnostica. Il pricer
all'emissione legge la correlazione di scenario da `--correlation`.

Il flusso di `main.py` segue l'ordine del calcolo, con le formule nei moduli
dedicati. Quando si aggiungerà il backtest, conviene separare il calcolo
riutilizzabile per una data di valutazione dalla lettura degli argomenti e
dalla stampa dei risultati.

## Assunzioni e lavoro rimanente

Il pricing usa ancora il calendario feriale provvisorio, volatilità ATM
costanti e una correlazione iniziale di scenario. Restano da verificare
il calendario ICE e la stima di correlazione, e da sviluppare la contabilità
dell'hedge e il backtest. Il SOFR grezzo resta disponibile per il futuro
conto cassa. Prima di usarlo occorre verificarne unità e convenzione.
