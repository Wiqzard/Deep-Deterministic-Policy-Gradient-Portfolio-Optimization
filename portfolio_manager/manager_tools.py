#@title IMPORTS
import pandas as pd
import numpy as np
from numpy import ndarray as ndarray
from random import randint
import scipy.optimize as optimize
from cvxopt import matrix, solvers
import sys, os


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def opt_weights(
    X,
    metric="return",
    max_leverage=1,
    rf_rate=0.0,
    alpha=0.0,
    freq: float = 252,
    no_cash=False,
    sd_factor=1.0,
    **kwargs,
):
    """Find best constant rebalanced portfolio with regards to some metric.
    :param X: Prices in ratios.
    :param metric: what performance metric to optimize, can be either `return` or `sharpe`
    :max_leverage: maximum leverage
    :rf_rate: risk-free rate for `sharpe`, can be used to make it more aggressive
    :alpha: regularization parameter for volatility in sharpe
    :freq: frequency for sharpe (default 252 for daily data)
    :no_cash: if True, we can't keep cash (that is sum of weights == max_leverage)
    """
    assert metric in ("return", "sharpe", "drawdown", "ulcer")
    assert X.notnull().all().all()

    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    if metric == "return":
        objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    elif metric == "sharpe":
        objective = lambda b: -sharpe(
            np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)),
            rf_rate=rf_rate,
            alpha=alpha,
            freq=freq,
            sd_factor=sd_factor,
        )

    if no_cash:
        cons = ({"type": "eq", "fun": lambda b: max_leverage - sum(b)},)
    else:
        cons = ({"type": "ineq", "fun": lambda b: max_leverage - sum(b)},)

    while True:
        # problem optimization
        res = optimize.minimize(
            objective,
            x_0,
            bounds=[(0.0, max_leverage)] * len(x_0),
            constraints=cons,
            method="slsqp",
            **kwargs,
        )
        # result can be out-of-bounds -> try it again
        EPS = 1e-7
        if (res.x < 0.0 - EPS).any() or (res.x > max_leverage + EPS).any():
            X = X + np.random.randn(1)[0] * 1e-5
            continue
            print("Optimal weights not found, trying again...")
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                continue
                print("Solution does not exist, use zero weights.")
                res.x = np.zeros(X.shape[1])
            else:
                continue
                print("Converged, but not successfully.")
                
            break

    return res.x

def rolling_corr(x, y, **kwargs):
    """Rolling correlation between columns from x and y."""

    def rolling(dataframe, *args, **kwargs):
        ret = dataframe.copy()
        for col in ret:
            ret[col] = ret[col].rolling(*args, **kwargs).mean()
        return ret

    n, k = x.shape

    EX = rolling(x, **kwargs)
    EY = rolling(y, **kwargs)
    EX2 = rolling(x**2, **kwargs)
    EY2 = rolling(y**2, **kwargs)

    RXY = np.zeros((n, k, k))

    for i, col_x in enumerate(x):
        for j, col_y in enumerate(y):
            DX = EX2[col_x] - EX[col_x] ** 2
            DY = EY2[col_y] - EY[col_y] ** 2
            product = x[col_x] * y[col_y]
            RXY[:, i, j] = product.rolling(**kwargs).mean() - EX[col_x] * EY[col_y]
            RXY[:, i, j] = RXY[:, i, j] / np.sqrt(DX * DY)

    return RXY, EX.values

def mc_simplex(d, points):
    """Sample random points from a simplex with dimension d.
    :param d: Number of dimensions.
    :param points: Total number of points.
    """
    a = np.sort(np.random.random((points, d)))
    a = np.hstack([np.zeros((points, 1)), a, np.ones((points, 1))])
    return np.diff(a)

def opt_markowitz(
    mu, sigma, long_only=True, reg=0.0, rf_rate=0.0, q=1.0, max_leverage=1.0
):  
    with HiddenPrints():
      """Get optimal weights from Markowitz framework."""
      # delete assets with NaN values or no volatility
      keep = ~(mu.isnull() | (np.diag(sigma) < 0.00000001))
      mu = mu[keep]
      sigma = sigma.loc[keep, keep]
      m = len(mu)
      # replace NaN values with 0
      sigma = sigma.fillna(0.0)
      # convert to matrices
      sigma = np.matrix(sigma)
      mu = np.matrix(mu).T
      # regularization for sigma matrix
      sigma += np.eye(m) * reg
      # pure approach - problems with singular matrix
      if not long_only:
          sigma_inv = np.linalg.inv(sigma)
          b = q / 2 * (1 + rf_rate) * sigma_inv @ (mu - rf_rate)
          b = np.ravel(b)
      else:
          def maximize(mu, sigma, r, q):
              n = len(mu)
              P = 2 * matrix((sigma - r * mu * mu.T + (n * r) ** 2) / (1 + r))
              q = matrix(-mu) * q
              G = matrix(-np.eye(n))
              h = matrix(np.zeros(n))
              if max_leverage is None or max_leverage == float("inf"):
                  sol = solvers.qp(P, q, G, h)
              else:
                  A = matrix(np.ones(n)).T
                  b = matrix(np.array([float(max_leverage)]))
                  sol = solvers.qp(P, q, G, h, A, b)
              return np.squeeze(sol["x"])

          while True:
              try:
                  b = maximize(mu, sigma, rf_rate, q)
                  break
              except ValueError:
                  raise
                  # deal with singularity
                  sigma = sigma + 0.0001 * np.eye(len(sigma))
      # add back values for NaN assets
      b = pd.Series(b, index=keep.index[keep])
      b = b.reindex(keep.index).fillna(0.0)
      return b

def simplex_proj(y):
    """Projection of y onto simplex."""
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.0

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(y - tmax, 0.0)
