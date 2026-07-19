import numpy as np
from garch import get_returns, fit_garch

def forecast_volatility(result, horizon=30):
    """
    Prévision de la variance conditionnelle sur 'horizon' jours.
    Retourne la volatilité annualisée prévue jour par jour.
    """
    forecasts = result.forecast(horizon=horizon, reindex=False)
    var_daily  = forecasts.variance.values[-1]        # variance en %²/jour
    vol_daily  = np.sqrt(var_daily)                   # volatilité en %/jour
    vol_annual = vol_daily * np.sqrt(252)             # annualisée
    return vol_daily, vol_annual

def print_forecast(vol_daily, vol_annual, horizon=30):
    print(f"\n── Prévision de volatilité — {horizon} jours ──")
    print(f"{'Jour':<6} {'Vol/jour (%)':>14} {'Vol/an (%)':>12}")
    print("-" * 35)
    for i in range(horizon):
        print(f"{i+1:<6} {vol_daily[i]:>14.4f} {vol_annual[i]:>12.2f}")

if __name__ == "__main__":
    returns          = get_returns()
    result           = fit_garch(returns)
    vol_daily, vol_annual = forecast_volatility(result, horizon=30)
    print_forecast(vol_daily, vol_annual)