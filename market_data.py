"""Daily market observations and the resolved fixing state of an Asian payoff."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FUTURE_KEY = ["date", "future_contract_code"]
OPTION_KEY = ["date", "option_ticker"]
PARQUET_FILE = "unpivoted_bid_offer_train_clean.parquet"
EXCEL_FILE = "COJ21_COZ22_historical_series.xlsx"


def default_data_dir() -> Path:
    project = Path(__file__).resolve().parent
    files = (project / PARQUET_FILE, project / EXCEL_FILE)
    return project if all(path.is_file() for path in files) else project.parent


def load_data(data_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    quotes = pd.read_parquet(data_dir / PARQUET_FILE)
    required = {
        "date", "expiration", "future_contract_code", "option_ticker", "strike",
        "option_type", "option_bid", "option_ask", "underlying_settlement",
        "underlying_bid", "underlying_ask", "sofr_rate",
    }
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")
    for column in ("date", "expiration"):
        quotes[column] = pd.to_datetime(quotes[column], errors="raise")
        if quotes[column].isna().any():
            raise ValueError(f"Date mancanti nella colonna {column}")
    if quotes[["option_ticker", "future_contract_code"]].isna().any().any():
        raise ValueError("Ticker o codici contratto mancanti")
    if quotes.duplicated(OPTION_KEY).any():
        raise ValueError("Quotazioni duplicate per data e ticker: verificare la fonte")
    history = pd.read_excel(
        data_dir / EXCEL_FILE, sheet_name=None,
    )
    for name, frame in history.items():
        if not {"date", "settlement", "bid", "ask"}.issubset(frame.columns):
            raise ValueError(f"Schema inatteso nel foglio {name}")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame.date.isna().any():
            raise ValueError(f"Date mancanti nel foglio {name}")
        if frame.date.duplicated().any():
            raise ValueError(f"Date duplicate nel foglio {name}")
        history[name] = frame.sort_values("date").reset_index(drop=True)
    quotes = quotes.sort_values(OPTION_KEY).reset_index(drop=True)
    return quotes, history


def build_future_quotes(
    options: pd.DataFrame, history: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Reconcile sources without constructing bid/ask pairs from separate rows."""
    columns = {
        "underlying_settlement": "settlement",
        "underlying_bid": "bid",
        "underlying_ask": "ask",
    }
    parquet = options[FUTURE_KEY + list(columns)].rename(columns=columns)
    sheets = [frame.assign(future_contract_code=code) for code, frame in history.items()]
    excel = pd.concat(sheets, ignore_index=True).reindex(columns=parquet.columns) if sheets else parquet.iloc[:0]
    settlements = _merge_settlements(parquet, excel)
    return _merge_bid_ask(settlements, parquet, excel).sort_index()


def _merge_settlements(parquet: pd.DataFrame, excel: pd.DataFrame) -> pd.DataFrame:
    parquet_settlements = parquet.groupby(FUTURE_KEY).settlement
    if parquet_settlements.nunique().gt(1).any():
        raise ValueError("Settlement futures incoerenti nella stessa data/contratto")

    quotes = pd.concat(
        [parquet_settlements.first().rename("parquet"),
         excel.set_index(FUTURE_KEY).settlement.rename("excel")],
        axis=1,
    )
    common = quotes[["parquet", "excel"]].dropna()
    if not np.allclose(common.parquet, common.excel, atol=1e-8, rtol=0):
        raise ValueError("Settlement Excel e Parquet discordanti")
    quotes["settlement"] = quotes.parquet.combine_first(quotes.excel)
    quotes["settlement_source"] = np.select(
        [quotes.parquet.notna() & quotes.excel.notna(), quotes.parquet.notna(),
         quotes.excel.notna()],
        ["parquet+excel", "parquet", "excel"],
        default="missing",
    )
    return quotes[["settlement", "settlement_source"]]


def _merge_bid_ask(settlements, parquet, excel):
    pair_columns = FUTURE_KEY + ["bid", "ask"]
    pairs = parquet.loc[_valid_pair(parquet), pair_columns].drop_duplicates()
    counts = pairs.groupby(FUTURE_KEY).size()
    unique = pairs.loc[~pairs.duplicated(FUTURE_KEY, keep=False)].set_index(FUTURE_KEY)
    valid_excel = excel.loc[_valid_pair(excel), pair_columns]
    confirmed = pairs.merge(valid_excel, on=pair_columns).set_index(FUTURE_KEY)
    excel_indexed = valid_excel.set_index(FUTURE_KEY)
    excel_only = excel_indexed.loc[~excel_indexed.index.isin(counts.index)]
    parquet_only = unique.loc[~unique.index.isin(excel_indexed.index)]
    selected = pd.concat(
        [confirmed.assign(bid_ask_source="parquet+excel"),
         parquet_only.assign(bid_ask_source="parquet"),
         excel_only.assign(bid_ask_source="excel")]
    )
    quotes = settlements.join(selected)
    quotes["parquet_pair_count"] = counts.reindex(quotes.index, fill_value=0)
    quotes["bid_ask_status"] = np.select(
        [quotes.bid_ask_source.notna(), quotes.parquet_pair_count.gt(0)],
        ["usable", "ambiguous"],
        default="missing",
    )
    return quotes


def _valid_pair(quotes: pd.DataFrame) -> pd.Series:
    return np.isfinite(quotes.bid) & np.isfinite(quotes.ask) & quotes.bid.le(quotes.ask)


@dataclass(frozen=True)
class MarketData:
    future_quotes: pd.DataFrame
    option_quotes: pd.DataFrame
    sofr: pd.Series

    @classmethod
    def load(cls, data_dir: Path) -> "MarketData":
        return cls.from_sources(*load_data(data_dir))

    @classmethod
    def from_sources(
        cls, options: pd.DataFrame, history: Mapping[str, pd.DataFrame]
    ) -> "MarketData":
        rates = options.groupby("date").sofr_rate
        if rates.nunique().gt(1).any():
            raise ValueError("Valori SOFR discordanti nella stessa seduta")
        return cls(
            build_future_quotes(options, history),
            options.set_index(OPTION_KEY).sort_index(),
            rates.first(),
        )

    def fixing_table(self, schedule: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """Show known fixings at the close and leave future settlements blank."""
        available = self.future_quotes.index.get_level_values("date") <= as_of
        quotes = self.future_quotes.loc[available].reindex(columns=["settlement", "settlement_source"])
        table = schedule.join(quotes, on=FUTURE_KEY, validate="many_to_one")
        table["status"] = np.select(
            [table.date.gt(as_of), np.isfinite(table.settlement)],
            ["pending", "observed"],
            default="missing",
        )
        table["contribution"] = table.weight * table.settlement
        return table

    def resolve_fixings(self, schedule: pd.DataFrame, as_of: pd.Timestamp) -> "FixingState":
        table = self.fixing_table(schedule, as_of)
        missing = table.loc[table.status.eq("missing"), FUTURE_KEY]
        if not missing.empty:
            keys = list(missing.itertuples(index=False, name=None))
            raise ValueError(f"Settlement mancanti per fixing gia' trascorsi: {keys}")
        return FixingState(
            float(table.contribution.sum()),
            schedule.loc[schedule.date.gt(as_of)].copy(),
        )


@dataclass(frozen=True)
class FixingState:
    observed_contribution: float
    remaining: pd.DataFrame

    def average(self, paths: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compute A on paths whose columns follow the remaining fixing dates."""
        if self.remaining.empty:
            return np.array([self.observed_contribution])

        columns = []
        for i, code in enumerate(self.remaining.future_contract_code):
            matrix = paths[code]
            if matrix.ndim != 2 or matrix.shape[1] != len(self.remaining):
                raise ValueError("Le colonne dei percorsi devono seguire i fixing residui")
            columns.append(matrix[:, i])
        return self.observed_contribution + np.column_stack(columns) @ self.remaining.weight.to_numpy()
