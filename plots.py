import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from garch import get_returns, fit_garch, long_run_variance
from forecast import forecast_volatility, simulate_volatility

returns = get_returns()
fit = fit_garch(returns)
lt_vol = np.sqrt(long_run_variance(fit["params"]) * 252)


def plot_volatility_over_time():
    """Rendements et volatilité conditionnelle estimée sur toute la période."""
    cond_vol = fit["conditional_volatility"] * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(returns.index, returns.values, color="steelblue", linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("Daily Return (%)")
    axes[0].set_title("SPY Daily Returns (2014–2023)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(returns.index, cond_vol, color="tomato", linewidth=1)
    axes[1].axhline(lt_vol, color="gray", linewidth=1, linestyle="--",
                    label=f"Long-run vol = {lt_vol:.2f}%")
    axes[1].set_ylabel("Annualized Volatility (%)")
    axes[1].set_title("GARCH(1,1) Conditional Volatility (from-scratch MLE)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig("conditional_volatility.png", dpi=150)
    plt.show()


def plot_forecast(horizon=30):
    """Prévision à 30 jours avec intervalle 5 %–95 % issu de simulations Monte Carlo."""
    _, vol_annual = forecast_volatility(fit, horizon)
    sims = simulate_volatility(fit, horizon)
    lower, upper = np.percentile(sims, [5, 95], axis=1)
    days = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(days, vol_annual, color="tomato", linewidth=2, label="GARCH(1,1) forecast")
    ax.fill_between(days, lower, upper, color="tomato", alpha=0.15,
                    label="5%–95% interval (10,000 simulated paths)")
    ax.axhline(lt_vol, color="gray", linewidth=1.2, linestyle="--",
               label=f"Long-run volatility = {lt_vol:.1f}%")
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title("30-Day Volatility Forecast — GARCH(1,1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("volatility_forecast.png", dpi=150)
    plt.show()


def plot_return_distribution():
    """Distribution des rendements vs loi normale : met en évidence les queues épaisses."""
    r = returns.values
    x = np.linspace(r.min(), r.max(), 300)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(r, bins=100, density=True, color="steelblue", alpha=0.6,
            edgecolor="white", label="SPY Returns")
    ax.plot(x, norm.pdf(x, r.mean(), r.std()), color="tomato", linewidth=2,
            label="Normal distribution")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution vs Normal — SPY (2014–2023)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("return_distribution.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_volatility_over_time()
    plot_forecast()
    plot_return_distribution()
