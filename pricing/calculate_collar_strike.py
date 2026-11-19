import numpy as np

from models.simulate_paths import simulate_paths
from pricing.find_K import find_K
from pricing.select_vols import select_vols

def calculate_collar_strike(price_april, price_may, smiles, date, contracts, future, time_steps, call_strike,
                            n_paths=100_000, correlation=0.8, seed=42):
    """
    INPUT:
        price_april = April futures settlement on the valuation date.
        price_may = May futures settlement on the valuation date.
        smiles = dictionary organized as smiles[contract][date], with K and IV lists.
        date = market date used for initial futures prices and smiles.
        contracts = April and May futures tickers, in this order.
        time_steps = simulation intervals in years, starting from date.
                     The last two fixing dates use the May future.
        call_strike = fixed strike of the collar call.
        n_paths = number of simulated paths; default is 100000.
        correlation = correlation between futures shocks; default is 0.8.
        seed = random seed for repeatable results; default is 42.

    OUTPUT:
        collar_strike = put strike that matches the simulated call value.

    Select daily market inputs, simulate both futures, and calculate the put strike.
    """

    smile_april = smiles[contracts[0]][date]
    smile_may = smiles[contracts[1]][date]

    uses_april = future["future_ticker"].eq(contracts[0]).to_numpy()
    forward_avg = np.where(uses_april, price_april, price_may).mean()
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, len(time_steps), 2))

    def simulate_average(basket_strike):
        vol_april, vol_may = select_vols(
            basket_strike, price_april, price_may, forward_avg,
            smile_april, smile_may,
        )

        paths_april, paths_may = simulate_paths(
            price_april,
            price_may,
            vol_april,
            vol_may,
            time_steps=time_steps,
            n_paths=n_paths,
            correlation=correlation,
            shocks=shocks,
        )

        return np.where(uses_april, paths_april, paths_may).mean(axis=1)

    return find_K(simulate_average, call_strike)


# uses_april = ... crea un array di booleani con un elemento per ogni fixing. True significa che il fixing usa Apr21, False che usa May21. Nel vostro calendario sono 18 True seguiti da 2 False.
# forward_avg = ... usa quella maschera per prendere il prezzo corrente di Apr21 18 volte e quello di May21 2 volte. Poi ne fa la media. All’emissione, nel vostro modello i futures sono martingale, quindi questo è il valore atteso della media asiatica:
# F_A=\frac{18F_{\mathrm{Apr}}+2F_{\mathrm{May}}}{20}=61{,}365.
# def simulate_average(basket_strike): definisce la funzione dentro calculate_collar_strike. Per questo può già usare forward_avg, i due smile, il calendario e gli altri parametri. basket_strike è il solo argomento che riceve a ogni chiamata. Sarà 78 per la call e il valore proposto da brentq per la put.
# relative_strike = basket_strike / forward_avg calcola la moneyness \(K/F_A\). Per la call vale \(78/61{,}365\approx 1{,}27\).
# Le due np.interp(...) leggono ciascuna volatilità allo strike equivalente \(K_i=F_i(K/F_A)\). Moltiplicando la moneyness per il prezzo di ciascun future, ottieni uno strike con la stessa moneyness sul suo smile. Questa è la correzione: prima leggevi le IV a \(K_i=F_i\), quindi a moneyness 1 per qualunque opzione. Ora le leggi dove cade lo strike che stai valutando.
# paths_april, paths_may = simulate_paths(...) chiama il simulatore esistente con le due IV appena calcolate. Ottieni due matrici. Ogni riga è un percorso e ogni colonna è una data di fixing. Passando lo stesso seed, riusi gli stessi shock Monte Carlo per tutti gli strike provati.
# return np.where(...).mean(axis=1) sceglie, in ogni colonna, il prezzo simulato del future previsto dal calendario. mean(axis=1) fa la media lungo i fixing di ogni percorso. Con 100.000 percorsi ottieni quindi 100.000 medie asiatiche, una per scenario.
# return find_K(simulate_average, call_strike) passa a find_K la funzione pronta per essere chiamata. Scrivi simulate_average senza parentesi perché sarà find_K a fornirle lo strike. Questa riga è indentata di quattro spazi, quindi appartiene alla funzione esterna calculate_collar_strike.
