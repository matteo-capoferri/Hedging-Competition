# Hedging competition - prima bozza Python

## Installazione

Richiede Python 3.11 o successivo. Da PowerShell nella cartella del progetto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe first_steps.py
```

Se `py` non trova un interprete, installare Python prima di questi passaggi.
Non serve attivare l'ambiente virtuale. Le librerie sono dichiarate con intervalli
compatibili: congelare le versioni esatte prima della consegna finale.

## Cosa fa

I due file dati originali devono essere nella cartella dello script, oppure nella
cartella indicata con `--data-dir`. Dati e PDF del corso restano locali e sono
esclusi da Git; una clonazione del repository da sola non contiene questi input.

- Legge il Parquet e tutti i fogli Excel senza modificare le fonti.
- Controlla schema, duplicati e coerenza dei prezzi futures tra fonti.
- Calcola mid, spread, scadenza ACT/365F e indicatori di qualita'.
- Costruisce un pannello futures senza riempire i dati mancanti.
- Calcola rendimenti e correlazione mobile usando esclusivamente dati passati.
- Estrae IV Black-76 non scontate per le opzioni OTM COJ21/COK21 alla data scelta.
- Esporta CSV e un riepilogo JSON in `output/initial_analysis`.

Esempio su una data successiva:

```powershell
.\.venv\Scripts\python.exe first_steps.py --date 2020-06-01
```

I parametri iniziali sono in `Config`, in testa a `first_steps.py`. I filtri di
liquidita' sono esplorativi. Le righe escluse della fotografia iniziale restano
nel CSV, con i flag; `iv_mid` e' vuota per righe escluse o radici non trovate.
Le IV sono espresse in decimali: 0.30 significa 30%.

Per eseguire i controlli numerici e sui dati sintetici:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Il programma controlla che la data scelta abbia quotazioni di entrambe le gambe.
Le discrepanze tra bid/ask futures ripetuti vengono elencate in
`futures_quote_conflicts.csv`: i rendimenti usano settlement coerenti, mentre
questi spread non vengono aggregati o scelti automaticamente per un hedge.
I filtri non costituiscono un controllo completo di arbitraggio dello smile.

## Limiti e prossimi passi

1. Verificare calendario ICE, fixing e roll prima di prezzare il collar.
2. Scegliere una regola motivata per la correlazione iniziale: la storia comune
   disponibile prima dell'emissione non basta alla finestra di 20 rendimenti.
   Il codice lascia NaN e non usa dati futuri come ripiego.
3. Calibrare smile separati per COJ21 e COK21 e controllare gli arbitraggi.
4. Implementare moment matching e Monte Carlo indipendente, poi lo strike zero-cost.
5. Aggiungere contabilita' futures-style, cash, SOFR e hedge con spread.

Il SOFR del dataset sembra in punti percentuali: verificarne metadati e convenzione.
La singola colonna non e' una curva completa. Non viene applicata al pricing delle
opzioni quotate, per le quali la consegna richiede Black con r=0.
Il premio di esercizio anticipato e' assunto nullo nell'idealizzazione futures-style
senza frizioni; discutere questa ipotesi nel report.
Le date di expiration del Parquet sono scadenze delle opzioni, non un calendario
ufficiale di scadenza dei futures.

Questa bozza produce un'analisi iniziale, non risultati di pricing del collar
o performance della strategia.
