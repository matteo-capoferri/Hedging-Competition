import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from black76 import black_76







#estrazione dati da excel e parquet
df = pd.read_parquet("unpivoted_bid_offer_train_clean.parquet")
df_excel = pd.read_excel("COJ21_COZ22_historical_series.xlsx", sheet_name="COJ21")

# estrazione prezzi di COJ21 dal parquet
df_coj21 = df.loc[df["actual_underlying_ticker"] == "COJ21", ["date", "actual_underlying_ticker", "underlying_settlement"]].drop_duplicates()

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



