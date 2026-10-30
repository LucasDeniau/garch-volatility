"""Prévisions de volatilité GARCH(1,1) : formule fermée + simulation Monte Carlo."""
import numpy as np
from garch import long_run_variance


def forecast_variance(fit, horizon=30):
    """
    Prévision fermée de la variance à h jours :
        σ²_{t+h} = V + (α+β)^(h−1) · (σ²_{t+1} − V)
    """
    p = fit["params"]
    eps_last, s2_last = fit["eps"][-1], fit["sigma2"][-1]
    s2_next = p["omega"] + p["alpha"] * eps_last ** 2 + p["beta"] * s2_last   # σ²_{t+1}
    V = long_run_variance(p)
    h = np.arange(horizon)                                                  # 0 … horizon−1
    return V + (p["alpha"] + p["beta"]) ** h * (s2_next - V)


def forecast_volatility(fit, horizon=30):
    """Volatilité prévue jour par jour, en %/jour et annualisée."""
    vol_daily = np.sqrt(forecast_variance(fit, horizon))
    return vol_daily, vol_daily * np.sqrt(252)


def simulate_volatility(fit, horizon=30, n_paths=10_000, seed=42):
    """
    Simule n_paths trajectoires futures du modèle (chocs z ~ N(0,1)) et renvoie
    la volatilité annualisée σ_{t+h} de chaque trajectoire : sert à construire
    un vrai intervalle (quantiles) autour de la prévision.
    """
    p = fit["params"]
    rng = np.random.default_rng(seed)
    eps_last, s2_last = fit["eps"][-1], fit["sigma2"][-1]
    sigma2 = np.full(n_paths, p["omega"] + p["alpha"] * eps_last ** 2 + p["beta"] * s2_last)
    paths = np.empty((horizon, n_paths))
    for h in range(horizon):
        paths[h] = sigma2
        eps2 = sigma2 * rng.standard_normal(n_paths) ** 2          # choc simulé au carré
        sigma2 = p["omega"] + p["alpha"] * eps2 + p["beta"] * sigma2
    return np.sqrt(paths * 252)


def print_forecast(vol_daily, vol_annual, days=(1, 3, 5, 10, 20, 30)):
    print(f"\n── Prévision de volatilité ──")
    print(f"{'Jour':<6}{'Vol/jour (%)':>14}{'Vol/an (%)':>12}")
    print("-" * 32)
    for d in days:
        if d <= len(vol_daily):
            print(f"{d:<6}{vol_daily[d-1]:>14.3f}{vol_annual[d-1]:>12.2f}")


if __name__ == "__main__":
    from garch import get_returns, fit_garch
    fit = fit_garch(get_returns())
    print_forecast(*forecast_volatility(fit, 30))
