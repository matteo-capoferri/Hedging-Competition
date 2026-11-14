from scipy.optimize import brentq
import numpy as np
from pricing.price_collar import price_collar


def find_K(simulate_average, Kc):
    call_average = simulate_average(Kc)
    call_price = np.maximum(call_average - Kc, 0).mean()

    def objective(Kp):
        put_average = simulate_average(Kp)
        put_price = np.maximum(Kp - put_average, 0).mean()
        return put_price - call_price

    return brentq(objective, 0.0, Kc)

# def find_K(simulate_average, Kc): riceve la funzione annidata e lo strike fisso della call, che nel vostro caso è 78.
# call_average = simulate_average(Kc) esegue la simulazione per la call. Il 78 arriva qui e viene usato per leggere le due IV agli strike equivalenti.
# call_price = np.maximum(call_average - Kc, 0).mean() calcola \((A-78)^+\) per ciascun percorso e poi fa la media tra i percorsi. Questa media dà il premio Monte Carlo della call. Il calcolo avviene una volta sola.
# def objective(Kp): definisce ciò che brentq deve portare a zero. Kp è uno strike candidato scelto dal solver.
# put_average = simulate_average(Kp) simula le medie asiatiche usando le IV associate a quello specifico candidato. Questo è il passaggio che mancava al Monte Carlo precedente.
# put_price = np.maximum(Kp - put_average, 0).mean() calcola \((K_p-A)^+\) per ciascun percorso e poi fa la media tra i percorsi.
# return put_price - call_price restituisce la differenza che vuoi annullare. Quando vale zero, i due premi coincidono e il collar è a costo zero.
# return brentq(objective, 0.0, Kc) cerca la radice fra 0 e 78. Ho verificato che sui vostri dati la differenza cambia segno tra questi due estremi. È quindi un intervallo valido per il collar all’emissione.