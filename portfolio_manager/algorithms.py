# @title CRP
import numpy as np

from portfolio_manager.portfolio_manager import PortfolioManager
from portfolio_manager.manager_tools import *


class CRP(PortfolioManager):
    """Constant rebalanced portfolio = use fixed weights all the time. Uniform weights
    are commonly used as a benchmark."""

    def __init__(self, args, flag, b=None):
        """
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super().__init__(args, flag=flag)
        self.name = "CRP"
        self.b = np.array(b) if b is not None else None

    def step(self, x, last_b, history):
        # init b to default if necessary
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        return self.b

    def weights(self, X):
        if self.b is None:
            b = X * 0 + 1
            b = b.div(b.sum(axis=1), axis=0)
            return b
        elif self.b.ndim == 1:
            return np.repeat([self.b], X.shape[0], axis=0)
        else:
            return self.b


class UBAH(PortfolioManager):
    PRICE_TYPE = "raw"

    def __init__(self, args, flag, b=None):
        super().__init__(args, flag=flag)
        self.name = "UBAH"
        self.b = np.array(b) if b is not None else None

    def weights(self, X):
        self.b = np.ones(8) / 8
        b = X.mul((self.b), axis=1)
        b = b.div(b.sum(axis=1), axis=0)
        return b


class BCRP(CRP):
    """Best Constant Rebalanced Portfolio = Constant Rebalanced Portfolio constructed
    with hindsight.
    """

    def __init__(self, args, flag, **kwargs):
        super().__init__(args, flag=flag)
        self.opt_weights_kwargs = kwargs
        self.name = "BCRP"

    def weights(self, X):
        """Find weights which maximize return on X in hindsight!"""
        # update frequency
        self.opt_weights_kwargs["freq"] = 94

        self.b = opt_weights(X, **self.opt_weights_kwargs)
        return super().weights(X)


class BestMarkowitz(CRP):
    """Optimal Markowitz portfolio constructed in hindsight."""

    def __init__(self, args, flag, global_sharpe=None, sharpe=None, **kwargs):
        super().__init__(args, flag=flag)
        self.global_sharpe = global_sharpe
        self._sharpe = sharpe
        self.opt_markowitz_kwargs = kwargs
        self.name = "BM"

    def weights(self, X):
        """Find optimal markowitz weights."""
        # update frequency
        freq = 1  # tools.freq(X.index)

        R = X - 1
        # calculate mean and covariance matrix and annualize them
        sigma = R.cov() * freq
        if self._sharpe:
            mu = pd.Series(np.sqrt(np.diag(sigma)), X.columns) * pd.Series(
                self._sharpe
            ).reindex(X.columns)
        elif self.global_sharpe:
            mu = pd.Series(np.sqrt(np.diag(sigma)) * self.global_sharpe, X.columns)
        else:
            mu = R.mean() * freq

        self.b = opt_markowitz(mu, sigma, **self.opt_markowitz_kwargs)

        return super().weights(R)


class UP(PortfolioManager):
    """Universal Portfolio by Thomas Cover enhanced for "leverage" (instead of just
    taking weights from a simplex, leverage allows us to stretch simplex to
    contain negative positions).
    """

    def __init__(self, args, flag, eval_points=1e4, leverage=1.0):
        """
        :param eval_points: Number of evaluated points (approximately). Complexity of the
            algorithm is O(time * eval_points * nr_assets**2) because of matrix multiplication.
        :param leverage: Maximum leverage used. leverage == 1 corresponds to simplex,
            leverage == 1/nr_stocks to uniform CRP. leverage > 1 allows negative weights
            in portfolio.
        """
        super().__init__(args, flag=flag)
        self.name = "UP"
        self.eval_points = int(eval_points)
        self.leverage = leverage

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        """Create a mesh on simplex and keep wealth of all strategies."""
        m = X.shape[1]
        # create set of CRPs
        self.W = np.matrix(mc_simplex(m - 1, self.eval_points))
        self.S = np.matrix(np.ones(self.W.shape[0])).T

        # stretch simplex based on leverage (simple calculation yields this)
        leverage = max(self.leverage, 1.0 / m)
        stretch = (leverage - 1.0 / m) / (1.0 - 1.0 / m)
        self.W = (self.W - 1.0 / m) * stretch + 1.0 / m

    def step(self, x, last_b, history):
        # calculate new wealth of all CRPs
        self.S = np.multiply(self.S, self.W * np.matrix(x).T)
        b = self.W.T * self.S

        return b / sum(b)


class Anticor(PortfolioManager):
    """Anticor (anti-correlation) is a heuristic portfolio selection algorithm.
    It adopts the consistency of positive lagged cross-correlation and negative
    autocorrelation to adjust the portfolio.
    """

    def __init__(self, args, flag, window=30):
        """
        :param window: Window parameter.
        :param c_version: Use c_version, up to 10x speed-up.
        """
        super().__init__(args, flag=flag)
        self.window = window
        self.name = "Anticor"

    def weights(self, X):
        window = self.window
        port = X
        n, m = port.shape
        weights = 1.0 / m * np.ones(port.shape)

        CORR, EX = rolling_corr(port, port.shift(window), window=window)
        for t in range(n - 1):
            M = CORR[t, :, :]
            mu = EX[t, :]
            # claim[i,j] is claim from stock i to j
            claim = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue

                    if mu[i] > mu[j] and M[i, j] > 0:
                        claim[i, j] += M[i, j]
                        # autocorrelation
                        if M[i, i] < 0:
                            claim[i, j] += abs(M[i, i])
                        if M[j, j] < 0:
                            claim[i, j] += abs(M[j, j])
            # calculate transfer
            transfer = claim * 0.0
            for i in range(m):
                total_claim = sum(claim[i, :])
                if total_claim != 0:
                    transfer[i, :] = weights[t, i] * claim[i, :] / total_claim
            # update weights
            weights[t + 1, :] = (
                weights[t, :] + np.sum(transfer, axis=0) - np.sum(transfer, axis=1)
            )
        return weights


class OLMAR(PortfolioManager):
    """On-Line Portfolio Selection with Moving Average Reversion"""

    PRICE_TYPE = "raw"

    def __init__(self, args, flag, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super().__init__(args, flag, min_history=window)
        self.name = "OLMAR"
        # input check
        window = 10
        eps = 5
        if window < 2:
            raise ValueError("window parameter must be >=3")
        if eps < 1:
            raise ValueError("epsilon parameter must be >=1")
        self.window = window
        self.eps = eps

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window :])
        return self.update(last_b, x_pred, self.eps)

    def predict(self, x, history):
        """Predict returns on next day."""
        return (history / x).mean()

    def update(self, b, x_pred, eps):
        """Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights."""
        x_pred_mean = np.mean(x_pred)
        lam = max(
            0.0, (eps - np.dot(b, x_pred)) / np.linalg.norm(x_pred - x_pred_mean) ** 2
        )
        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x_pred - x_pred_mean)

        # project it onto simplex
        return simplex_proj(b)


def norm(x):
    axis = 0 if isinstance(x, pd.Series) else 1
    return np.sqrt((x**2).sum(axis=axis))


class RMR(OLMAR):
    """Robust Median Reversion. Strategy exploiting mean-reversion by robust
    L1-median estimator. Practically the same as OLMAR.
    """

    PRICE_TYPE = "raw"

    def __init__(self, args, flag, window=5, eps=10.0, tau=0.001):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        :param tau: Precision for finding median. Recommended value is around 0.001. Strongly
                    affects algo speed.
        """
        super().__init__(args, window, eps, flag)  # =flag)
        self.tau = tau
        self.name = "RMR"

    def predict(self, x, history):
        """find L1 median to historical prices"""
        y = history.mean()
        y_last = None
        while y_last is None or norm(y - y_last) / norm(y_last) > self.tau:
            y_last = y
            d = norm(history - y)
            y = history.div(d, axis=0).sum() / (1.0 / d).sum()
        return y / x
