import numpy as np
from garch import get_returns, fit_garch, print_params
from forecast import forecast_volatility, print_forecast

returns = get_returns()
result  = fit_garch(returns)

print_params(result)
print_forecast(*forecast_volatility(result, horizon=10), horizon=10)