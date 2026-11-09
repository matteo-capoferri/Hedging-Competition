import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from black76 import black_76
import numpy as np

#estrazione dati da excel e parquet
df = pd.read_parquet("unpivoted_bid_offer_train_clean.parquet")
df_excel = pd.read_excel("COJ21_COZ22_historical_series.xlsx", sheet_name="COJ21")

# estrazione prezzi di COJ21 dal parquet
df_coj21 = df.loc[df["actual_underlying_ticker"] == "COJ21", ["date", "actual_underlying_ticker", "underlying_settlement"]].drop_duplicates()
df_cok21 = df.loc[df["actual_underlying_ticker"] == "COK21", ["date", "actual_underlying_ticker", "underlying_settlement"]].drop_duplicates()

# confronto prezzi di CJO21 tra excel e parquet sulle date che condividono
df_merge = pd.merge(df_coj21, df_excel, on = "date", how = "inner")
#df_merge["difference"] = df_merge["settlement"] - df_merge["underlying_settlement"]
#print(df_merge.describe())

# trovo le date che appartengono solo ad excel->sono tutte le date che non sono comuni
# df_missing = pd.merge(df_excel, df_coj21, on = "date", how = "left", indicator = True)
# missing = df_missing.loc[df_missing["_merge"] == "left_only"]
# print(missing["date"])

#prendiamo i prezzi delle vanilla nel 06-01-2020: il primo giorno disponibile
emission_date = pd.Timestamp("2020-01-06")
options_apr21 = df.loc[(df["date"] == emission_date) & (df["actual_underlying_ticker"] == "COJ21"), ["strike", "option_type", "expiration", "settlement_price", "underlying_settlement", "actual_underlying_ticker", "moneyness_usd", "time_to_expiry_days", "option_bid", "option_ask"]]
options_may21 = df.loc[(df["date"] == emission_date) & (df["actual_underlying_ticker"] == "COK21"), ["strike", "option_type", "expiration", "settlement_price", "underlying_settlement", "actual_underlying_ticker", "moneyness_usd", "time_to_expiry_days", "option_bid", "option_ask"]]

# options_apr21.info()
# options_may21.info()
# missing = options_apr21.loc[options_apr21["settlement_price"].isna()]
# print(missing[["strike", "option_type"]])

#mancano dei prezzi->li escludiamo
options_apr21 = options_apr21.dropna(subset = "settlement_price")
options_may21 = options_may21.dropna(subset = "settlement_price")
#print(options_apr21.shape)


#ora vogliamo estrarre lo smile
IV_apr21 = []
IV_may21 = []
for index, row in options_apr21.iterrows():
    vol = brentq(lambda x: black_76(row["underlying_settlement"], 
                                    row["strike"], row["time_to_expiry_days"]/365,
                                    x,
                                    row["option_type"]) - row["settlement_price"], 1e-5, 3)
    IV_apr21.append(vol)

for index, row in options_may21.iterrows():
    vol = brentq(lambda x: black_76(row["underlying_settlement"], 
                                    row["strike"], row["time_to_expiry_days"]/365,
                                    x,
                                    row["option_type"]) - row["settlement_price"], 1e-5, 3)
    IV_may21.append(vol)

plt.scatter(options_apr21["strike"], IV_apr21, label = "apr21")
plt.scatter(options_may21["strike"], IV_may21, label = "may21")
plt.xlabel("Strike ($/barile)")
plt.ylabel("Implied volatility")
plt.title("Brent volatility smiles - 06/01/2020")
plt.legend()
plt.grid(True)
plt.show()
#abbiamo solo 9 punti sullo smile... dovuto al fatto che put e call coincidono siugli strikes
#print(options_apr21["strike"].nunique())


#simulazione montecarlo per il pricing:

#costruzione calendario
first_fixing = pd.Timestamp("2021-02-01")
last_apr21 = pd.Timestamp("2021-02-24") #al penultimo giorno il contratto smette did essere liquido
maturity = pd.Timestamp("2021-02-26")

fixing_apr21 = pd.bdate_range(start=first_fixing, end=last_apr21)
fixing_may21 = pd.bdate_range(start=last_apr21 + pd.offsets.BDay(1), end=maturity)
fixing = fixing_apr21.append(fixing_may21)
fixing = fixing.insert(0, emission_date)
dt_fixing = fixing.to_series().diff().dropna().dt.days/365
dt = dt_fixing.to_numpy()
print(dt.shape)
print(dt)

n_F1 = len(fixing_apr21)
n_F2 = len(fixing_may21)
#generazione rumore gaussiano
rho = 0.8
M =100000
n_step = len(dt)
F0_apr21 = df_coj21["underlying_settlement"].iloc[0]
F0_may21 = df_cok21["underlying_settlement"].iloc[0]

sigma_apr21 = np.interp(F0_apr21, options_apr21["strike"], IV_apr21)
sigma_may21 = np.interp(F0_may21, options_may21["strike"], IV_may21)
rng = np.random.default_rng(42)
Z = rng.standard_normal((M, n_step, 2))
#creiamo W correlati w1 = z1    w2=z1*rho + z2*sqrt(1-rho**2)
W1 = Z[:,:,0]
W2 = W1*rho + Z[:,:,2] * np.sqrt(1-rho**2)

increments_1 = (-0.5 * sigma_apr21**2 * dt + np.sqrt(dt) * W1)
log_returns_1 = np.cumsum(increments_1, axis=1)
increments_2 = (-0.5 * sigma_may21**2 * dt + np.sqrt(dt) * W2)
log_returns_2 = np.cumsum(increments_2, axis=1)
F1 = F0_apr21 * np.exp(log_returns_1)
F2 = F0_may21 * np.exp(log_returns_2)
print(F1)
print(F2)

