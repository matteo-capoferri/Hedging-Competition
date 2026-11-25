import json
from pathlib import Path

import matplotlib.pyplot as plt


def save_backtest_report(backtest, output_dir):
    """Esporta i risultati gia' calcolati, senza rivalutare il portafoglio."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(output_dir / "daily_backtest.csv")
    worst_dates = backtest["hedged_pnl"].pow(2).nlargest(10).index
    backtest.loc[worst_dates].to_csv(output_dir / "worst_days.csv")
    summary = {
        "sessions": len(backtest),
        "start": str(backtest.index[0].date()),
        "end": str(backtest.index[-1].date()),
        "hedged_mean_squared_pnl_usd2": float(backtest["hedged_pnl"].pow(2).mean()),
        "unhedged_mean_squared_pnl_usd2": float(backtest["collar_pnl"].pow(2).mean()),
        "cumulative_pnl_usd": float(backtest["hedged_pnl"].sum()),
        "final_portfolio_value_usd": float(backtest["portfolio_value"].iloc[-1]),
        "execution_pnl_usd": float(backtest["execution_pnl"].sum()),
        "option_execution_pnl_usd": float(backtest["option_execution_pnl"].sum()),
        "options_pnl_usd": float(backtest["options_pnl"].sum()),
        "interest_pnl_usd": float(backtest["interest_pnl"].sum()),
        # Monitoraggio contabile dei dati reali; le diagnostiche degli ordini sono nei test.
        "max_accounting_error_usd": float(backtest["accounting_error"].abs().max()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    backtest["portfolio_value"].plot(ax=axes[0], title="Valore del portafoglio")
    axes[0].set_ylabel("USD")
    backtest["hedged_pnl"].plot(ax=axes[1], title="P&L giornaliero")
    axes[1].set_ylabel("USD")
    backtest[["collar_pnl", "hedged_pnl"]].pow(2).plot(
        ax=axes[2], title="P&L giornaliero al quadrato"
    )
    axes[2].set_ylabel("USD squared")
    backtest[["position_april", "position_may"]].plot(
        ax=axes[3], title="Posizioni futures", drawstyle="steps-post"
    )
    axes[3].set_ylabel("Contratti")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "backtest.png", dpi=150)
    plt.close(fig)
