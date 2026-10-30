"""
GARCH(1,1) implémenté from scratch : récurrence de la variance,
log-vraisemblance et estimation par maximum de vraisemblance (scipy).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TICKER = "SPY"
START = "2014-01-01"
END = "2024-01-01"          # données de 2014 à fin 2023
PARAM_NAMES = ["mu", "omega", "alpha", "beta"]


# ── DONNÉES ──────────────────────────────────────────────
def get_returns():
    """Télécharge les prix SPY et calcule les rendements journaliers en %."""
    import yfinance as yf
    raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    prices = raw["Close"].squeeze()            # Série 1D (et non DataFrame)
    returns = prices.pct_change().dropna() * 100  # en % : ω sera en %²
    returns.name = TICKER
    return returns


# ── MODÈLE ───────────────────────────────────────────────
def garch_variance(params, r):
    """
    Récurrence GARCH(1,1) :  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    avec ε_t = r_t − μ. On initialise σ²_0 par la variance empirique.
    """
    mu, omega, alpha, beta = params
    eps = r - mu
    sigma2 = np.empty(len(r))
    sigma2[0] = eps.var()
    for t in range(1, len(r)):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2, eps


def neg_log_likelihood(params, r):
    """
    Opposé de la log-vraisemblance gaussienne :
        ℓ = −½ Σ [ ln(2π) + ln σ²_t + ε²_t / σ²_t ]
    On minimise −ℓ (scipy ne sait que minimiser).
    """
    sigma2, eps = garch_variance(params, r)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)


def fit_garch(returns):
    """Estime (μ, ω, α, β) par maximum de vraisemblance sous contraintes."""
    r = np.asarray(returns, dtype=float)

    # Point de départ : α=0.1, β=0.8 et ω choisi pour que la variance
    # de long terme ω/(1−α−β) soit égale à la variance empirique
    x0 = np.array([r.mean(), 0.1 * r.var(), 0.1, 0.8])

    bounds = [(None, None),   # μ libre
              (1e-8, None),   # ω > 0
              (0.0, 1.0),     # α ≥ 0
              (0.0, 1.0)]     # β ≥ 0
    # Stationnarité : α + β < 1
    constraints = {"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3]}

    res = minimize(neg_log_likelihood, x0, args=(r,), method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"Optimisation échouée : {res.message}")

    params = res.x
    sigma2, eps = garch_variance(params, r)
    loglik = -res.fun
    k = len(params)
    return {
        "params": dict(zip(PARAM_NAMES, params)),
        "loglik": loglik,
        "aic": 2 * k - 2 * loglik,
        "sigma2": sigma2,
        "eps": eps,
        "conditional_volatility": pd.Series(np.sqrt(sigma2), index=getattr(returns, "index", None)),
        "n_obs": len(r),
    }


# ── QUANTITÉS DÉRIVÉES ───────────────────────────────────
def long_run_variance(p):
    """Variance de long terme V = ω / (1 − α − β)."""
    return p["omega"] / (1 - p["alpha"] - p["beta"])


def half_life(p):
    """Nombre de jours pour que l'écart à V soit divisé par deux."""
    return np.log(0.5) / np.log(p["alpha"] + p["beta"])


def print_params(fit):
    p = fit["params"]
    v_lt = long_run_variance(p)
    print(f"\n── GARCH(1,1) from scratch — {TICKER} 2014–2023 ({fit['n_obs']} obs.) ──")
    print(f"  mu    (μ) : {p['mu']:.4f} %/jour")
    print(f"  omega (ω) : {p['omega']:.6f}")
    print(f"  alpha (α) : {p['alpha']:.4f}  ← réactivité aux chocs")
    print(f"  beta  (β) : {p['beta']:.4f}  ← persistance")
    print(f"  α + β     : {p['alpha'] + p['beta']:.4f}  ← doit être < 1")
    print(f"  Demi-vie  : {half_life(p):.1f} jours")
    print(f"  Vol LT    : {np.sqrt(v_lt):.4f} %/jour  |  {np.sqrt(v_lt * 252):.2f} %/an")
    print(f"\n  Log-likelihood : {fit['loglik']:.2f}")
    print(f"  AIC            : {fit['aic']:.2f}")


def compare_with_arch(returns, fit):
    """Vérifie l'implémentation en comparant à la librairie de référence arch."""
    try:
        from arch import arch_model
    except ImportError:
        print("\n(arch non installé : comparaison ignorée — pip install arch)")
        return
    ref = arch_model(returns, mean="Constant", vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
    ref_params = [ref.params["mu"], ref.params["omega"], ref.params["alpha[1]"], ref.params["beta[1]"]]
    print("\n── Vérification : from scratch vs librairie arch ──")
    print(f"  {'':<8}{'Scratch':>12}{'arch':>12}")
    for name, ref_val in zip(PARAM_NAMES, ref_params):
        print(f"  {name:<8}{fit['params'][name]:>12.4f}{ref_val:>12.4f}")
    print(f"  {'loglik':<8}{fit['loglik']:>12.2f}{ref.loglikelihood:>12.2f}")
    print("  (petits écarts attendus : arch initialise σ²_0 différemment)")


if __name__ == "__main__":
    returns = get_returns()
    fit = fit_garch(returns)
    print_params(fit)
    compare_with_arch(returns, fit)
