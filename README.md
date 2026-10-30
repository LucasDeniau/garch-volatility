# GARCH(1,1) Volatility Modeling

> Conditional volatility estimation and short-term forecasting on the S&P 500, with a from-scratch maximum likelihood implementation of GARCH(1,1).

---

## Overview

For my third quantitative finance project, I implemented a GARCH(1,1) model from scratch — variance recursion, Gaussian log-likelihood and constrained maximum likelihood estimation with `scipy` — and validated it against the reference `arch` library.

Unlike Black-Scholes, which assumes constant volatility, GARCH captures two well-documented features of financial returns: **volatility clustering** (turbulent periods tend to follow turbulent periods) and **mean reversion** (volatility reverts to its long-run average).

The model is estimated on 10 years of daily SPY returns (2014–2023), then used to forecast volatility over a 30-day horizon and to compute a conditional Value-at-Risk.

---

## Theory

### Why Constant Volatility Fails

Financial returns exhibit **fat tails** and **volatility clustering** — both inconsistent with a constant σ. The March 2020 crash sent SPY conditional volatility to around 120% annualized, a regime impossible to capture with a fixed parameter.

### The GARCH(1,1) Model

Returns are modeled as:

$$r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \sim \mathcal{N}(0,1)$$

The conditional variance evolves according to:

$$\sigma_t^2 = \omega + \alpha \cdot \varepsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

| Parameter     | Role                                                          |
|---------------|---------------------------------------------------------------|
| **ω** (omega) | Constant variance floor                                       |
| **α** (alpha) | Reactivity to recent shocks — weight on $\varepsilon_{t-1}^2$ |
| **β** (beta)  | Persistence of past volatility — weight on $\sigma_{t-1}^2$   |

Stationarity requires **α + β < 1**. Taking expectations gives the unconditional (long-run) variance:

$$V = \sigma_{LT}^2 = \frac{\omega}{1 - \alpha - \beta}$$

### Maximum Likelihood Estimation

Parameters $(\mu, \omega, \alpha, \beta)$ are estimated by maximizing the Gaussian log-likelihood:

$$\ell = -\frac{1}{2} \sum_{t=1}^{T} \left[ \ln(2\pi) + \ln \sigma_t^2 + \frac{\varepsilon_t^2}{\sigma_t^2} \right]$$

where $\sigma_t^2$ is computed recursively (initialized at the sample variance). The term $\varepsilon_t^2 / \sigma_t^2$ penalizes large shocks the model did not anticipate, while $\ln \sigma_t^2$ penalizes unnecessarily high predicted variance. The optimization is solved with SLSQP under the constraints $\omega > 0$, $\alpha, \beta \geq 0$ and $\alpha + \beta < 1$.

### Forecasting & Mean Reversion

The h-step-ahead variance forecast has a closed form:

$$\sigma_{t+h}^2 = V + (\alpha + \beta)^{h-1} \left( \sigma_{t+1}^2 - V \right)$$

The gap to the long-run variance shrinks geometrically at rate $(\alpha + \beta)$. Its **half-life** is $\ln(0.5) / \ln(\alpha + \beta)$.

A 5%–95% forecast interval is obtained by simulating 10,000 future paths of the model.

---

## Results

Estimated on **SPY daily returns, 2014–2023** (2,515 observations).

### Estimated Parameters

| Parameter      | Value           | Interpretation                           |
|----------------|-----------------|------------------------------------------|
| **μ**          | 0.0880 %/day    | Mean daily return                        |
| **ω**          | 0.0375          | Variance floor (in %²)                   |
| **α**          | 0.2023          | Strong reactivity to shocks              |
| **β**          | 0.7704          | High persistence                         |
| **α + β**      | 0.9728          | Near-integrated — slow mean reversion    |
| **Half-life**  | 25.1 days       | Time for a volatility shock to halve     |
| **σ_LT**       | **18.64%/year** | Long-run annualized volatility           |
| Log-likelihood | −3172.34        | Model fit                                |
| AIC            | 6352.68         | Information criterion                    |

The high persistence α + β ≈ 0.97 is typical of equity indices: volatility shocks fade slowly, consistent with prolonged stress periods (2018, 2020, 2022). A volatility shock takes about one month to be halved.

### Validation Against `arch`

| Parameter      | From scratch | `arch` library |
|----------------|--------------|----------------|
| μ              | 0.0880       | 0.0877         |
| ω              | 0.0375       | 0.0378         |
| α              | 0.2023       | 0.2020         |
| β              | 0.7704       | 0.7704         |
| Log-likelihood | −3172.34     | −3171.79       |

The small differences come from the initialization of $\sigma_0^2$ (sample variance here, exponential backcast in `arch`).

### 30-Day Volatility Forecast

| Day | Vol/day (%) | Vol/year (%) |
|-----|-------------|--------------|
| 1   | 0.616       | 9.79         |
| 3   | 0.659       | 10.45        |
| 5   | 0.696       | 11.05        |
| 10  | 0.774       | 12.29        |
| 20  | 0.887       | 14.09        |
| 30  | 0.964       | 15.31        |
| → ∞ | 1.174       | 18.64        |

The forecast starts at 9.79%/year (low-volatility regime at the end of 2023) and converges slowly toward the long-run level of 18.64%.

### Application: Conditional Value-at-Risk

| 1-day VaR 99%                                  | Value  |
|------------------------------------------------|--------|
| Conditional (GARCH volatility forecast)        | 1.43%  |
| Unconditional (long-run volatility)            | 2.73%  |

A GARCH-based VaR adapts to the current market regime: it is low in calm periods and rises immediately when volatility picks up, whereas a VaR based on average volatility reacts slowly.

### Visualizations

![Conditional Volatility](conditional_volatility.png)
*Top: SPY daily returns. Bottom: GARCH(1,1) conditional volatility — the COVID crash (March 2020) illustrates volatility clustering.*

![Volatility Forecast](volatility_forecast.png)
*30-day volatility forecast with a 5%–95% interval from 10,000 simulated paths. Mean reversion toward 18.6% is clearly visible.*

![Return Distribution](return_distribution.png)
*SPY return distribution vs fitted normal — fat tails and excess kurtosis.*

### Limitations & Extensions

- **Symmetric response**: positive and negative shocks have the same impact, whereas equity volatility rises more after falls (leverage effect). Extensions: GJR-GARCH, EGARCH.
- **Gaussian innovations**: standardized residuals often remain fat-tailed. Extension: Student-t innovations, compared through AIC.
- **Fourth moment**: with these parameters, $(\alpha+\beta)^2 + 2\alpha^2 > 1$, so the theoretical kurtosis of the model is infinite — consistent with very heavy tails, but a sign that the Gaussian specification is stretched.
- **Constant parameters** over 10 years mixing very different regimes; **single asset** (a portfolio would require a multivariate model such as DCC-GARCH).
- **No out-of-sample evaluation**: a natural next step is to backtest a rolling GARCH VaR (Kupiec test).

---

## Project Structure

```
garch-volatility/
├── garch.py        # Data, variance recursion, log-likelihood, MLE, validation vs arch
├── forecast.py     # Closed-form forecast and Monte Carlo simulation of future volatility
├── plots.py        # Conditional volatility, forecast interval, return distribution
└── main.py         # Entry point — parameters, validation, forecast and VaR
```

---

## Usage

```bash
# Install dependencies (arch is only used for validation)
pip install numpy pandas scipy matplotlib yfinance arch

# Run estimation, validation and forecast
python main.py

# Generate visualizations
python plots.py
```

---

## References

- Engle, R. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation*. Econometrica.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.
- Hull, J. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
