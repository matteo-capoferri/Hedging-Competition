import matplotlib.pyplot as plt


def plot_smiles(smiles, emission_date):
    """
    INPUT:
        smiles = dictionary containing one smile for each contract.
            K = strikes in increasing order.
            IV = implied volatilities corresponding to those strikes.
        emission_date = market date displayed in the chart title.

    OUTPUT:
        None = no value is returned; a new chart is created.
        Call plt.show() to display the chart.

    Plot implied volatility against strike for each contract.
    """
    plt.figure()
    for contract in smiles:
        smile = smiles[contract]
        plt.scatter(smile["K"], smile["IV"], label=contract)
    plt.xlabel("Strike ($/barrel)")
    plt.ylabel("Implied volatility")
    plt.title(f"Brent volatility smiles - {emission_date:%d/%m/%Y}")
    plt.legend()
    plt.grid(True)
