import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

TICKER = "SPY"
START  = "2014-01-01"
END    = "2024-01-01"

def get_returns():
    """Télécharge les prix SPY et calcule les rendements en %."""
    raw     = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    prices  = raw["Close"]
    returns = prices.pct_change().dropna() * 100  # en % pour GARCH
    return returns

def fit_garch(returns):
    """Estime un modèle GARCH(1,1) par maximum de vraisemblance."""
    model  = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    result = model.fit(disp="off")
    return result

def print_params(result):
    params = result.params
    alpha  = params["alpha[1]"]
    beta   = params["beta[1]"]
    omega  = params["omega"]

    sigma_lt = np.sqrt(omega / (1 - alpha - beta)) 

    print("\n── GARCH(1,1) — SPY 2014–2024 ──")
    print(f"  omega (ω)       : {omega:.6f}")
    print(f"  alpha (α)       : {alpha:.4f}   ← réactivité aux chocs")
    print(f"  beta  (β)       : {beta:.4f}   ← persistance")
    print(f"  α + β           : {alpha+beta:.4f}   ← doit être < 1")
    print(f"  Volatilité LT   : {sigma_lt:.4f}% /jour")
    print(f"  Volatilité LT   : {sigma_lt * np.sqrt(252):.2f}% /an")
    print(f"\n  Log-likelihood  : {result.loglikelihood:.2f}")
    print(f"  AIC             : {result.aic:.2f}")

if __name__ == "__main__":
    returns = get_returns()
    result  = fit_garch(returns)
    print_params(result)