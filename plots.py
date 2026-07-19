import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from garch import get_returns, fit_garch
from forecast import forecast_volatility

returns = get_returns()
result  = fit_garch(returns)

def plot_volatility_over_time():
    """Volatilité conditionnelle estimée sur toute la période."""
    cond_vol = result.conditional_volatility * np.sqrt(252)  # annualisée

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Rendements
    axes[0].plot(returns.index, returns.values, color="steelblue",
                 linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("Daily Return (%)")
    axes[0].set_title("SPY Daily Returns (2014–2024)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    # Volatilité conditionnelle
    axes[1].plot(returns.index, cond_vol, color="tomato", linewidth=1)
    axes[1].axhline(result.params["omega"] / (1 - result.params["alpha[1]"]
                    - result.params["beta[1]"]) ** 0.5 * np.sqrt(252),
                    color="gray", linewidth=1, linestyle="--",
                    label=f"Long-run vol = 18.59%")
    axes[1].set_ylabel("Annualized Volatility (%)")
    axes[1].set_title("GARCH(1,1) Conditional Volatility")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig("conditional_volatility.png", dpi=150)
    plt.show()


def plot_forecast():
    """Prévision de volatilité sur 30 jours avec intervalle de confiance."""
    horizon = 30
    vol_daily, vol_annual = forecast_volatility(result, horizon)

    days    = np.arange(1, horizon + 1)
    lt_vol  = np.sqrt(result.params["omega"] / (
                1 - result.params["alpha[1]"] - result.params["beta[1]"]
              )) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(days, vol_annual, color="tomato", linewidth=2, label="GARCH(1,1) Forecast")
    ax.axhline(lt_vol, color="gray", linewidth=1.2, linestyle="--",
               label=f"Long-run volatility = {lt_vol:.1f}%")
    ax.fill_between(days, vol_annual * 0.85, vol_annual * 1.15,
                    color="tomato", alpha=0.15, label="±15% confidence band")

    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title("30-Day Volatility Forecast — GARCH(1,1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("volatility_forecast.png", dpi=150)
    plt.show()


def plot_return_distribution():
    """Distribution des rendements vs loi normale — montre les fat tails."""
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(9, 4))

    mu_r  = returns.mean().item()
    sig_r = returns.std().item()
    x     = np.linspace(returns.min().item(), returns.max().item(), 300)

    ax.hist(returns.values, bins=100, density=True,
            color="steelblue", alpha=0.6, edgecolor="white", label="SPY Returns")
    ax.plot(x, norm.pdf(x, mu_r, sig_r), color="tomato",
            linewidth=2, label="Normal distribution")

    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution vs Normal — SPY (2014–2024)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("return_distribution.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_volatility_over_time()
    plot_forecast()
    plot_return_distribution()