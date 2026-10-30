import numpy as np
from garch import get_returns, fit_garch, print_params, compare_with_arch, long_run_variance
from forecast import forecast_volatility, print_forecast

returns = get_returns()
fit = fit_garch(returns)

print_params(fit)
compare_with_arch(returns, fit)

vol_daily, vol_annual = forecast_volatility(fit, horizon=30)
print_forecast(vol_daily, vol_annual)
v_lt = long_run_variance(fit["params"])
print(f"{'→ ∞':<6}{np.sqrt(v_lt):>14.3f}{np.sqrt(v_lt * 252):>12.2f}")

# VaR 99 % à 1 jour : conditionnelle (GARCH) vs volatilité de long terme
z99 = 2.326
print(f"\n── VaR 99 % à 1 jour ──")
print(f"  Conditionnelle (σ GARCH de demain) : {z99 * vol_daily[0]:.2f} %")
print(f"  Avec la volatilité de long terme   : {z99 * np.sqrt(v_lt):.2f} %")
