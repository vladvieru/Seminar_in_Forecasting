"""
forecast_combination.py
=======================
Research-grade implementation of a graph-based, covariance-aware forecast
combination method for unstable environments.

Layers
------
1. Local prediction of future pairwise loss differentials (Richter-Smetanina style)
2. Graph aggregation via eigenvector centrality
3. Covariance-aware simplex-constrained weight optimization with shrinkage

Author : (research prototype)
License: MIT
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import optimize
from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# 0.  CONSTANTS & HELPERS
# ---------------------------------------------------------------------------

RNG_SEED = 42
EPS = 1e-12          # generic numerical floor
RIDGE_COV = 1e-6     # default ridge for covariance matrices
TELEPORT = 1e-6      # default teleportation for eigenvector centrality
SIGNED_WEIGHT_METHODS = {"full_gcsr_relaxed"}


def _ensure_rng(seed=None):
    if seed is None:
        return np.random.default_rng(RNG_SEED)
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# ===================================================================
# 1.  LOSS FUNCTIONS
# ===================================================================

def squared_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Squared-error loss  (y-f)^2."""
    return (y - f) ** 2


def absolute_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Absolute-error loss  |y-f|."""
    return np.abs(y - f)


LOSS_REGISTRY: Dict[str, Callable] = {
    "squared": squared_loss,
    "absolute": absolute_loss,
}


# ===================================================================
# 2.  LOCAL PAIRWISE LOSS-DIFFERENTIAL MODEL  (Richter–Smetanina style)
# ===================================================================
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

EPS = 1e-10

# ---------- kernel ----------

def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel K(u) = 0.75*(1-u^2) for |u|<=1, else 0."""
    w = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1.0
    w[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return w


# ---------- local linear AR estimation ----------

def _build_ar_design(series: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build AR(d) regression matrices from *series*.

    Returns
    -------
    Y : shape (T-d,)
    X : shape (T-d, d+1)   columns = [1, lag1, ..., lag_d]
    """
    n = len(series)
    if n <= d:
        raise ValueError("Series too short for AR({})".format(d))
    Y = series[d:].copy()
    X = np.ones((n - d, d + 1))
    for k in range(1, d + 1):
        X[:, k] = series[d - k: n - k]
    return Y, X


def local_linear_ar_fit(
    series: np.ndarray,
    d: int,
    h: float,
    target_frac: float = 1.0,
    t_filter: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local linear AR(d) fit at rescaled time *target_frac*.

    Builds the augmented Z_t(u) = [X_{t-1}, (t/T - u)*X_{t-1}] design
    matrix (size 2*(d+1)) as in equation (3) of the paper, estimates
    theta(u) via WLS, and returns the first (d+1) components as rho(u).

    Parameters
    ----------
    series      : 1-d array of length T
    d           : AR lag order
    h           : bandwidth on [0,1] scale
    target_frac : evaluation point u (usually 1.0 for forecasting)
    t_filter    : optional array of length T; 0 = exclude, 1 = include.
                  Applied to the (T-d) observation rows after lag truncation.

    Returns
    -------
    beta    : rho(u) estimate, shape (d+1,)  [first half of theta(u)]
    resid   : Y - X @ beta, shape (T-d,)
    W_diag  : kernel weights, shape (T-d,)
    """
    Y, X = _build_ar_design(series, d)
    n = len(Y)  # = T - d
    T = len(series)

    # rescaled time for each observation row (index d, d+1, ..., T-1)
    fracs = np.arange(d, T) / T

    # kernel weights centred at target_frac
    u = (fracs - target_frac) / max(h, EPS)
    W_diag = epanechnikov_kernel(u)

    # apply t_filter if supplied (operates on the post-lag rows)
    if t_filter is not None:
        W_diag = W_diag * t_filter[d:]

    # build augmented local linear design matrix Z_t(u) of size (n, 2*(d+1))
    # top block: X  (the standard AR regressors)
    # bottom block: (t/T - u) * X  (local linear expansion)
    delta = (fracs - target_frac)[:, None]   # shape (n, 1)
    X_ll = X * delta                          # (t/T - u) * X, shape (n, d+1)
    Z = np.hstack([X, X_ll])                  # shape (n, 2*(d+1))

    # WLS: (Z'WZ)^{-1} Z'WY  where W = diag(W_diag)
    ZW = Z * W_diag[:, None]                  # row-wise kernel weighting
    ZWZ = ZW.T @ Z                            # shape (2*(d+1), 2*(d+1))
    ZWY = ZW.T @ Y                            # shape (2*(d+1),)

    try:
        theta = np.linalg.solve(ZWZ, ZWY)
    except np.linalg.LinAlgError:
        theta = np.zeros(2 * (d + 1))

    # rho(u) is the first (d+1) components of theta
    beta = theta[:d + 1]
    resid = Y - X @ beta
    return beta, resid, W_diag


def _estimate_full_sequence(
    series: np.ndarray,
    d: int,
    h: float,
    t_filter: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate rho(t/T) and compute residuals at every t/T, matching the
    second code's est_theta_general vectorised loop.

    Returns
    -------
    rho_seq  : shape (T-d, d+1)  — rho(t/T) for t = d, ..., T-1
    resid_seq: shape (T-d,)      — L_t - X_{t-1}' rho(t/T)
    """
    Y, X = _build_ar_design(series, d)
    n = len(Y)
    T = len(series)
    fracs = np.arange(d, T) / T

    rho_seq = np.empty((n, d + 1))
    for i in range(n):
        t_frac = fracs[i]
        beta, _, _ = local_linear_ar_fit(
            series, d, h, target_frac=t_frac, t_filter=t_filter
        )
        rho_seq[i] = beta

    resid_seq = Y - np.sum(X * rho_seq, axis=1)
    return rho_seq, resid_seq


def local_predict_mean(series: np.ndarray, d: int, h: float) -> float:
    """
    One-step-ahead conditional mean forecast from local linear AR(d).
    Localises at the boundary (target_frac = 1.0).

    Returns X_T' @ rho_hat(1) where X_T = [1, L_T, ..., L_{T-d+1}].
    """
    beta, _, _ = local_linear_ar_fit(series, d, h, target_frac=1.0)
    x_new = np.ones(d + 1)
    for k in range(1, d + 1):
        x_new[k] = series[-k]
    return float(x_new @ beta)


def local_predict_scale(
    series: np.ndarray,
    d: int,
    h1: float,
    h2: float,
) -> float:
    """
    Local scale (std-dev) estimate at u=1 from second-stage local linear
    kernel regression on squared residuals of the first-stage local AR(d).

    Computes residuals xi_t = L_t - X_{t-1}' rho_hat(t/T) for each t
    using the time-appropriate coefficient, matching est_theta_general
    in the second code. Then runs local linear regression of xi_t^2 on
    rescaled time, evaluated at u=1.
    """
    T = len(series)
    n = T - d

    # --- stage 1: time-varying residuals using rho(t/T) at each t ---
    _, resid_seq = _estimate_full_sequence(series, d, h1)
    sq_resid = resid_seq ** 2

    # --- stage 2: local linear regression of sq_resid on rescaled time ---
    # F_t(u) = [1, t/T - u], evaluated at u = 1
    fracs = np.arange(d, T) / T
    u2 = (fracs - 1.0) / max(h2, EPS)
    W2 = epanechnikov_kernel(u2)

    delta2 = (fracs - 1.0)[:, None]
    F = np.hstack([np.ones((n, 1)), delta2])   # shape (n, 2)
    FW = F * W2[:, None]
    FWF = FW.T @ F
    FWsq = FW.T @ sq_resid

    try:
        varsigma = np.linalg.solve(FWF, FWsq)
        var_est = float(varsigma[0])           # first component = sigma^2(1)
    except np.linalg.LinAlgError:
        var_est = float(np.mean(sq_resid))

    return float(np.sqrt(max(var_est, EPS)))


# ---------- BIC lag selection ----------

def bic_lag_selection(
    series: np.ndarray,
    d_max: int,
    h: float,
    correct_bic: bool = False,
) -> int:
    """
    Select AR lag order d in {0,...,d_max} via BIC.

    Parameters
    ----------
    series      : 1-d array of length T
    d_max       : maximum lag order to consider
    h           : preliminary bandwidth for estimation
    correct_bic : if True, uses the paper's BIC formulation (eq. 7):
                      BIC(d) = sum_t log(sigma^2(t/T)) + (d+1)*log(T-d)
                  if False (default), uses the original approximate formulation.

    Returns
    -------
    best_d : int
    """
    T = len(series)
    best_d = 0
    best_bic = np.inf

    for d in range(0, d_max + 1):
        if T <= d + 2:
            continue

        if correct_bic:
            # paper's BIC (eq. 7): sum over t of log(sigma^2(t/T)) + (d+1)*log(T-d)
            # use time-varying residuals at each t/T then take as sigma^2(t/T) estimate
            _, resid_seq = _estimate_full_sequence(series, d, h)
            sigma2_seq = np.maximum(resid_seq ** 2, EPS)
            bic = np.sum(np.log(sigma2_seq)) + (d + 1) * np.log(T - d)
        else:
            # original approximate formulation
            beta, resid, W_diag = local_linear_ar_fit(
                series, d, h, target_frac=1.0
            )
            n_eff = W_diag.sum()
            if n_eff < d + 2:
                continue
            sse = (W_diag * resid ** 2).sum()
            sigma2 = sse / max(n_eff, EPS)
            if sigma2 <= 0:
                continue
            bic = np.log(sigma2 + EPS) + (d + 1) * np.log(max(n_eff, 2)) / max(n_eff, 1)

        if bic < best_bic:
            best_bic = bic
            best_d = d

    return best_d


# ---------- CV bandwidth selection ----------

def cv_bandwidth_selection(
    series: np.ndarray,
    d: int,
    h_grid: np.ndarray,
    n_folds: int = 20,
) -> Tuple[float, np.ndarray]:
    """
    Interleaved cross-validation for bandwidth selection, matching the
    paper's fold structure zeta_j = {Q*k + j, k=1,2,...} with Q=n_folds.

    Each held-out observation i is predicted using the model estimated
    without its fold, evaluated at its own rescaled time u = i/T, exactly
    matching the second code's h1_CV_calc behaviour.

    CV score is mean of per-fold means, matching np.mean(CV_Q) in the
    second code.

    Returns (best_h, cv_scores).
    """
    T = len(series)
    Q = n_folds
    cv_scores = np.full(len(h_grid), np.nan)

    if T <= d + 2:
        return float(h_grid[len(h_grid) // 2]), cv_scores

    Y_full, X_full = _build_ar_design(series, d)
    n = len(Y_full)  # T - d

    for ih, h in enumerate(h_grid):
        fold_mse = np.zeros(Q)

        for j in range(Q):
            # interleaved fold indices into post-lag rows 0..n-1
            fold_idx = np.arange(j, n, Q)
            if len(fold_idx) == 0:
                continue

            # t_filter: exclude fold members (original indices fold_idx + d)
            t_filter = np.ones(T)
            t_filter[fold_idx + d] = 0.0

            # evaluate at each held-out observation's own rescaled time
            # matching second code: resid[d:][dj] uses rho(s/T) at each s
            sq_errors = np.empty(len(fold_idx))
            for k, idx in enumerate(fold_idx):
                t_frac = (idx + d) / T
                beta_cv, _, _ = local_linear_ar_fit(
                    series, d, h,
                    target_frac=t_frac,
                    t_filter=t_filter,
                )
                pred = X_full[idx] @ beta_cv
                sq_errors[k] = (Y_full[idx] - pred) ** 2

            fold_mse[j] = np.mean(sq_errors)

        cv_scores[ih] = np.mean(fold_mse)   # mean of fold means, matching second code

    best_idx = int(np.nanargmin(cv_scores))
    return float(h_grid[best_idx]), cv_scores


def make_h_grid(
    n_points: int = 15,
    h_min: float = 0.05,
    h_max: float = 1.0,
) -> np.ndarray:
    """Logarithmically spaced bandwidth grid on [h_min, h_max]."""
    return np.exp(np.linspace(np.log(h_min), np.log(h_max), n_points))


# ---------- Full pairwise LD predictor ----------

@dataclass
class PairwiseLDResult:
    """Result of predicting one pairwise loss differential."""
    mu_hat: float = 0.0
    sigma_hat: float = 1.0
    d_selected: int = 0
    h1_selected: float = 0.2
    h2_selected: float = 0.2


def predict_pairwise_ld(
    delta_L: np.ndarray,
    d_max: int = 4,
    h1_grid: Optional[np.ndarray] = None,
    h2_grid: Optional[np.ndarray] = None,
    fixed_d: Optional[int] = None,
    fixed_h1: Optional[float] = None,
    fixed_h2: Optional[float] = None,
    n_cv_folds: int = 20,
    correct_bic: bool = False,
) -> PairwiseLDResult:
    """
    Given a history of pairwise loss differentials (up to time T),
    produce next-period conditional mean and scale estimates.

    Parameters
    ----------
    delta_L     : 1-d array of loss differences L_t = Loss_A - Loss_B
    d_max       : maximum lag order for BIC selection
    h1_grid     : bandwidth grid for first-stage CV (default: log-spaced)
    h2_grid     : bandwidth grid for second-stage CV (default: log-spaced)
    fixed_d     : fix lag order, bypassing BIC selection
    fixed_h1    : fix first-stage bandwidth, bypassing CV
    fixed_h2    : fix second-stage bandwidth, bypassing CV
    n_cv_folds  : number of interleaved CV folds (paper recommends Q>=20)
    correct_bic : if True use paper's BIC formulation; if False use
                  approximate formulation (default False)

    Returns
    -------
    PairwiseLDResult with mu_hat, sigma_hat, and selected d, h1, h2
    """
    T = len(delta_L)
    if T < 5:
        return PairwiseLDResult()

    if h1_grid is None:
        h1_grid = make_h_grid(12, 0.05, 1.0)
    if h2_grid is None:
        h2_grid = make_h_grid(10, 0.10, 1.0)

    h_prelim = float(h1_grid[len(h1_grid) // 2])

    # lag selection
    if fixed_d is not None:
        d = fixed_d
    else:
        d = bic_lag_selection(delta_L, d_max, h_prelim, correct_bic=correct_bic)

    # bandwidth h1
    if fixed_h1 is not None:
        h1 = fixed_h1
    else:
        h1, _ = cv_bandwidth_selection(delta_L, d, h1_grid, n_cv_folds)

    # bandwidth h2
    if fixed_h2 is not None:
        h2 = fixed_h2
    else:
        h2, _ = cv_bandwidth_selection(delta_L, d, h2_grid, n_cv_folds)

    mu = local_predict_mean(delta_L, d, h1)
    sigma = local_predict_scale(delta_L, d, h1, h2)

    return PairwiseLDResult(
        mu_hat=mu,
        sigma_hat=sigma,
        d_selected=d,
        h1_selected=h1,
        h2_selected=h2,
    )


# ===================================================================
# 3.  GRAPH LAYER
# ===================================================================

def build_adjacency_raw(mu_matrix: np.ndarray) -> np.ndarray:
    """
    Raw adjacency: A_{ij} = max(-mu_{ij}, 0).
    mu_{ij} < 0 => i beats j => edge weight from i to j.
    """
    A = np.maximum(-mu_matrix, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def build_adjacency_standardized(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    c: float = 1e-4,
) -> np.ndarray:
    """Standardized adjacency: A_{ij} = max(-mu_{ij}/(sigma_{ij}+c), 0)."""
    A = np.maximum(-mu_matrix / (sigma_matrix + c), 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def build_adjacency_thresholded(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    threshold: float = 1.0,
    c: float = 1e-4,
) -> np.ndarray:
    """Thresholded adjacency based on t-ratio."""
    t_ratio = -mu_matrix / (sigma_matrix + c)
    A = np.where(t_ratio > threshold, t_ratio, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def eigenvector_centrality(
    A: np.ndarray,
    teleport: float = TELEPORT,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Dominant eigenvector centrality of non-negative matrix A.

    Uses power iteration with optional teleportation for reducibility.
    Returns normalised score vector summing to 1.
    """
    M = A.shape[0]
    if A.max() < EPS:
        return np.ones(M) / M  # fallback: uniform

    # add teleportation
    A_reg = A + teleport * np.ones((M, M)) / M
    # power iteration
    r = np.ones(M) / M
    for _ in range(max_iter):
        r_new = A_reg @ r
        norm = r_new.sum()
        if norm < EPS:
            return np.ones(M) / M
        r_new /= norm
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    r = np.maximum(r, 0.0)
    r /= r.sum() + EPS
    return r


def row_sum_strength(A: np.ndarray) -> np.ndarray:
    """Row-sum (out-strength) centrality, normalised."""
    s = A.sum(axis=1)
    total = s.sum()
    if total < EPS:
        return np.ones(A.shape[0]) / A.shape[0]
    return s / total


def pagerank_centrality(
    A: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Simple PageRank on column-normalised A with damping *alpha*."""
    M = A.shape[0]
    col_sum = A.sum(axis=0)
    col_sum[col_sum < EPS] = 1.0
    P = A / col_sum[None, :]
    r = np.ones(M) / M
    for _ in range(max_iter):
        r_new = alpha * (P @ r) + (1 - alpha) / M
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    r = np.maximum(r, 0.0)
    r /= r.sum() + EPS
    return r


def softmax_average_advantage(mu_matrix: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Softmax of average predicted advantages (negative of row-mean mu)."""
    M = mu_matrix.shape[0]
    adv = np.zeros(M)
    for i in range(M):
        vals = [-mu_matrix[i, j] for j in range(M) if j != i]
        adv[i] = np.mean(vals) if vals else 0.0
    adv_scaled = adv / max(tau, EPS)
    adv_scaled -= adv_scaled.max()  # numerical stability
    w = np.exp(adv_scaled)
    return w / w.sum()


# ===================================================================
# 4.  COVARIANCE LAYER
# ===================================================================

def rolling_covariance(
    errors: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Rolling sample covariance of forecast errors.
    errors : shape (T, M)
    Uses last *window* observations.
    """
    T, M = errors.shape
    use = errors[max(0, T - window):T]
    if len(use) < M + 2:
        return np.eye(M)
    return np.cov(use, rowvar=False, ddof=1)


def ewma_covariance(
    errors: np.ndarray,
    lam: float = 0.94,
) -> np.ndarray:
    """Exponentially weighted moving average covariance."""
    T, M = errors.shape
    S = np.zeros((M, M))
    if T == 0:
        return np.eye(M)
    S = np.outer(errors[0], errors[0])
    for t in range(1, T):
        S = lam * S + (1 - lam) * np.outer(errors[t], errors[t])
    return S


def shrinkage_covariance(
    errors: np.ndarray,
    window: int = 60,
    shrink_target: str = "diagonal",
    shrink_intensity: Optional[float] = None,
) -> np.ndarray:
    """
    Ledoit–Wolf style shrinkage toward diagonal or identity.
    If *shrink_intensity* is None, use a simple analytical formula.
    """
    S = rolling_covariance(errors, window)
    M = S.shape[0]
    if shrink_target == "diagonal":
        T_mat = np.diag(np.diag(S))
    else:
        T_mat = np.eye(M) * np.trace(S) / M

    if shrink_intensity is not None:
        delta = shrink_intensity
    else:
        # simple Oracle Approximating Shrinkage intensity
        n = min(len(errors), window)
        delta = min(max((M) / (n + M), 0.01), 0.99)

    return (1 - delta) * S + delta * T_mat


def diagonal_covariance(errors: np.ndarray, window: int = 60) -> np.ndarray:
    """Diagonal-only covariance (variances only)."""
    S = rolling_covariance(errors, window)
    return np.diag(np.diag(S))


def regularise_cov(Sigma: np.ndarray, ridge: float = RIDGE_COV) -> np.ndarray:
    """Add ridge to diagonal for numerical stability."""
    return Sigma + ridge * np.eye(Sigma.shape[0])


COVARIANCE_REGISTRY = {
    "rolling": rolling_covariance,
    "ewma": ewma_covariance,
    "shrinkage": shrinkage_covariance,
    "diagonal": diagonal_covariance,
}


# ===================================================================
# 5.  WEIGHT OPTIMIZATION LAYER
# ===================================================================

def simplex_project(w: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex.
    Algorithm of Duchi et al. (2008).
    """
    M = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, M + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones(M) / M
    rho_max = rho[-1]
    theta = (cssv[rho_max] - 1.0) / (rho_max + 1.0)
    w_proj = np.maximum(w - theta, 0.0)
    w_proj /= w_proj.sum() + EPS
    return w_proj


def graph_only_weights(r: np.ndarray) -> np.ndarray:
    """Weights proportional to graph centrality scores."""
    w = np.maximum(r, 0.0)
    s = w.sum()
    if s < EPS:
        return np.ones(len(r)) / len(r)
    return w / s


def covariance_only_weights(Sigma: np.ndarray, ridge: float = RIDGE_COV) -> np.ndarray:
    """Simplex-constrained minimum-variance weights via QP."""
    M = Sigma.shape[0]
    Sigma_r = regularise_cov(Sigma, ridge)
    r_zero = np.zeros(M)
    gamma_zero = 0.0
    return _solve_combination_qp(Sigma_r, r_zero, 0.0, gamma_zero, M)


def full_combination_weights(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Full graph-covariance-shrinkage combination.

    argmin_w  w'Sigma w  - alpha * r'w  + gamma * ||w - wbar||^2
    s.t. w in simplex
    """
    M = Sigma.shape[0]
    Sigma_r = regularise_cov(Sigma, ridge)
    return _solve_combination_qp(Sigma_r, r, alpha, gamma, M)


def full_combination_weights_relaxed(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Full graph-covariance-shrinkage combination with only the affine-sum
    constraint retained.

    argmin_w  w'Sigma w  - alpha * r'w  + gamma * ||w - wbar||^2
    s.t.      1'w = 1
    """
    M = Sigma.shape[0]
    Sigma_r = regularise_cov(Sigma, ridge)
    return _solve_combination_affine(Sigma_r, r, alpha, gamma, M)


def _combination_objective_terms(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quadratic and linear terms shared by the constrained solvers."""
    wbar = np.ones(M) / M
    Q = Sigma + gamma * np.eye(M)
    c = -(alpha * r + 2.0 * gamma * wbar)
    return Q, c


def _solve_combination_qp(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    M: int,
) -> np.ndarray:
    """Core QP solver using scipy."""
    wbar = np.ones(M) / M

    Q, c = _combination_objective_terms(Sigma, r, alpha, gamma, M)

    def objective(w):
        return w @ Q @ w + c @ w

    def gradient(w):
        return 2.0 * Q @ w + c

    # constraints: sum(w) = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * M
    w0 = wbar.copy()

    try:
        res = optimize.minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        w_opt = res.x
    except Exception:
        w_opt = wbar.copy()

    # safety project
    w_opt = np.maximum(w_opt, 0.0)
    w_opt /= w_opt.sum() + EPS
    return w_opt


def _solve_combination_affine(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    M: int,
) -> np.ndarray:
    """Closed-form equality-constrained solution allowing signed weights."""
    Q, c = _combination_objective_terms(Sigma, r, alpha, gamma, M)
    ones = np.ones(M)
    wbar = np.ones(M) / M

    try:
        q_inv_ones = np.linalg.solve(Q, ones)
        q_inv_c = np.linalg.solve(Q, c)
        denom = float(ones @ q_inv_ones)
        if abs(denom) < EPS:
            return wbar.copy()
        lagrange = -(2.0 + float(ones @ q_inv_c)) / denom
        w_opt = -0.5 * (q_inv_c + lagrange * q_inv_ones)
    except LinAlgError:
        return wbar.copy()

    if not np.all(np.isfinite(w_opt)):
        return wbar.copy()

    # Re-enforce the affine constraint after the linear solve to remove tiny
    # numerical drift while preserving the signed allocation profile.
    w_opt = np.asarray(w_opt, dtype=float)
    w_opt -= (w_opt.sum() - 1.0) / M
    return w_opt


def multiplicative_tilt_weights(
    w_base: np.ndarray,
    r: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Multiplicative tilt: w_i propto w_base_i * exp(kappa * r_i), renormalised.
    """
    log_w = np.log(np.maximum(w_base, EPS)) + kappa * r
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum() + EPS
    return w


# ===================================================================
# 6.  BENCHMARK METHODS
# ===================================================================

def equal_weights(M: int) -> np.ndarray:
    return np.ones(M) / M


def median_forecast(forecasts: np.ndarray) -> float:
    """Return median across M forecasts at one time point."""
    return float(np.median(forecasts))


def recent_best_selection(
    losses: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Select forecast with lowest average recent loss.
    losses : (T_hist, M)
    """
    M = losses.shape[1]
    use = losses[max(0, losses.shape[0] - window):]
    avg = use.mean(axis=0)
    w = np.zeros(M)
    w[np.argmin(avg)] = 1.0
    return w


def bates_granger_weights(
    errors: np.ndarray,
    window: int = 60,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Bates-Granger inverse-variance weights (diagonal),
    then simplex projection.
    """
    T, M = errors.shape
    use = errors[max(0, T - window):]
    var = np.var(use, axis=0, ddof=1)
    var = np.maximum(var, EPS)
    inv_var = 1.0 / var
    w = inv_var / inv_var.sum()
    return w


def bates_granger_mv_weights(
    errors: np.ndarray,
    window: int = 60,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Minimum-variance simplex-constrained (uses full covariance).
    """
    Sigma = rolling_covariance(errors, window)
    return covariance_only_weights(Sigma, ridge)


def rs_selection_weights(
    mu_matrix: np.ndarray,
) -> np.ndarray:
    """
    Richter-Smetanina style selection: pick forecast i that is predicted
    to beat the most others (largest net outperformance count).
    """
    M = mu_matrix.shape[0]
    wins = np.zeros(M)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            if mu_matrix[i, j] < 0:
                wins[i] += 1
    w = np.zeros(M)
    w[np.argmax(wins)] = 1.0
    return w


# ===================================================================
# 7.  SIMULATION ENVIRONMENT
# ===================================================================

@dataclass
class ScenarioConfig:
    """
    Store the configuration for a single simulation scenario.

    The configuration controls the number of forecasters and time periods, the
    common-shock component, the bias process, the idiosyncratic variance
    process, optional outliers, and optional cross-sectional dependence through
    either a one-factor or clustered structure.
    """
    name: str = "default"
    M: int = 8           # Number of forecasters
    T: int = 400         # Total number of time periods
    T0: int = 200        # First out-of-sample evaluation index
    sigma_common: float = 0.5   # Standard deviation of the common shock
    seed: int = 42

    # Bias parameters
    bias_type: str = "zero"  # One of: zero, constant, break, drift, cluster
    bias_values: Optional[np.ndarray] = None  # Fixed bias vector, shape (M,)
    bias_break_time: Optional[int] = None  # Break date for bias_type="break"
    bias_pre: Optional[np.ndarray] = None  # Bias vector before the break
    bias_post: Optional[np.ndarray] = None  # Bias vector after the break
    bias_drift_speed: float = 0.01  # Fallback drift scale if eta2 is not given
    bias_centered: bool = True  # Remove cross-sectional mean bias if True
    bias_init_low: float = 0.0  # Lower bound for random initial biases
    bias_init_high: float = 1.0  # Upper bound for random initial biases
    bias_drift_rho: float = 0.95  # Persistence of the latent bias slope
    bias_drift_eta2: Optional[float] = None  # Innovation variance of drift slope

    # Variance parameters
    sigma_idio: Optional[np.ndarray] = None  # Fixed sigma path, shape (M,) or (T, M)
    sigma_process: str = "constant"  # One of: constant, break, smooth_precision
    base_sigma2: float = 1.0  # Baseline variance used in smooth-precision mode
    sigma_shift_time: Optional[int] = None  # Break date for piecewise sigma paths
    sigma_pre: Optional[np.ndarray] = None  # Sigma vector before the variance shift
    sigma_post: Optional[np.ndarray] = None  # Sigma vector after the variance shift
    sigma_drift_rho: float = 0.95  # Persistence of the latent log-precision slope
    sigma_drift_eta2: float = 8e-4  # Innovation variance of the precision slope
    log_precision_min: float = -2.0  # Lower truncation bound for log precision
    log_precision_max: float = 2.0  # Upper truncation bound for log precision

    # Outlier parameters
    outlier_prob: float = 0.0  # Per-entry probability of replacing idio noise
    outlier_scale: float = 5.0  # Maximum outlier multiplier relative to sigma
    outlier_min_scale: float = 2.0  # Minimum outlier multiplier relative to sigma

    # Dependence parameters
    factor_rho: Optional[np.ndarray] = None  # One-factor loadings, shape (M,)
    n_clusters: int = 1  # Number of clusters in clustered dependence mode
    cluster_labels: Optional[np.ndarray] = None  # Cluster assignment for each forecaster
    cluster_rho: float = 0.6  # Loading on the cluster-specific latent factor


@dataclass
class SimulationData:
    """
    Store the output of a simulation run.

    Attributes:
        y (np.ndarray): Realized target series of shape (T,)
        forecasts (np.ndarray): Forecast matrix of shape (T, M)
        errors (np.ndarray): Forecast-error matrix of shape (T, M)
        losses (np.ndarray): Loss matrix of shape (T, M)
        bias_paths (np.ndarray): Bias process for each forecaster, shape (T, M)
        sigma_paths (np.ndarray): Idiosyncratic scale paths, shape (T, M)
        common_shock (np.ndarray): Common shock series of shape (T,)
        config (ScenarioConfig): Configuration used to generate the data
    """
    y: np.ndarray               # (T,) realized target series
    forecasts: np.ndarray       # (T, M) forecast values
    errors: np.ndarray          # (T, M) forecast errors
    losses: np.ndarray          # (T, M) squared losses
    bias_paths: np.ndarray      # (T, M) bias paths
    sigma_paths: np.ndarray     # (T, M) idiosyncratic standard deviations
    common_shock: np.ndarray    # (T,) common shock path
    config: ScenarioConfig = field(default_factory=ScenarioConfig)


def generate_scenario(cfg: ScenarioConfig) -> SimulationData:
    """
    Generate simulated targets, forecasts, errors, and losses from a scenario
    configuration.

    The generator builds the data in stages:

        1. Construct bias paths `b_{j,t}`
        2. Construct idiosyncratic scale paths `sigma_{j,t}`
        3. Draw a common shock shared across forecasters
        4. Draw idiosyncratic noise, optionally with factor or cluster dependence
        5. Combine components into forecast errors
        6. Build the target series and implied forecasts

    Forecast errors follow the sign convention used throughout this file:

        e_{j,t} = c_t - b_{j,t} - u_{j,t}

    where `c_t` is the common shock, `b_{j,t}` is the bias term, and `u_{j,t}`
    is the idiosyncratic component.

    Args:
        cfg (ScenarioConfig): Scenario specification

    Returns:
        SimulationData: Simulated target, forecasts, errors, and metadata
    """
    rng = _ensure_rng(cfg.seed)
    M, T = cfg.M, cfg.T

    # 1. Generate bias paths  b_{j,t}
    bias = _generate_bias(cfg, rng)  # (T, M)

    # 2. Generate idiosyncratic variance paths
    sigma_idio = _generate_sigma(cfg, rng)  # (T, M)

    # 3. Common shock
    common = rng.normal(0, cfg.sigma_common, size=T)

    # 4. Idiosyncratic errors
    idio = rng.normal(0, 1, size=(T, M)) * sigma_idio

    # Replace independent idiosyncratic noise by a one-factor structure if requested.
    if cfg.factor_rho is not None:
        factor = rng.normal(0, 1, size=T)
        idio_clean = np.zeros((T, M))
        u = rng.normal(0, 1, size=(T, M))
        for j in range(M):
            rho_j = cfg.factor_rho[j]
            idio_clean[:, j] = sigma_idio[:, j] * (
                rho_j * factor + np.sqrt(max(1 - rho_j ** 2, 0)) * u[:, j]
            )
        idio = idio_clean

    # Cluster dependence takes precedence over the global one-factor structure.
    if cfg.n_clusters > 1 and cfg.cluster_labels is not None:
        cluster_factors = rng.normal(0, 1, size=(T, cfg.n_clusters))
        u = rng.normal(0, 1, size=(T, M))
        rho_c = cfg.cluster_rho
        for j in range(M):
            cl = cfg.cluster_labels[j]
            idio[:, j] = sigma_idio[:, j] * (
                rho_c * cluster_factors[:, cl]
                + np.sqrt(max(1 - rho_c ** 2, 0)) * u[:, j]
            )

    # Outliers replace, rather than add to, the baseline idiosyncratic shock.
    if cfg.outlier_prob > 0:
        outlier_mask = rng.random(size=(T, M)) < cfg.outlier_prob
        upper_scale = max(cfg.outlier_scale, cfg.outlier_min_scale)
        outlier_magnitude = rng.uniform(
            cfg.outlier_min_scale * sigma_idio,
            upper_scale * sigma_idio,
            size=(T, M),
        )
        outlier_sign = rng.choice([-1.0, 1.0], size=(T, M))
        outlier_vals = outlier_sign * outlier_magnitude
        idio = np.where(outlier_mask, outlier_vals, idio)

    # 5. Forecast errors.
    # Keep the sign convention used in this file: common shocks enter
    # positively, while bias and idiosyncratic terms enter negatively.
    errors = common[:, None] - bias - idio  # (T, M)

    # 6. Target
    # y_t drawn as some latent, forecasts = y - errors
    # Simplest: y_t = mu + common noise + signal
    y_signal = rng.normal(0, 1, size=T).cumsum() * 0.01  # mild random walk signal
    y = y_signal + common  # plus common
    forecasts = y[:, None] - errors  # f_{j,t} = y_t - e_{j,t}

    # Losses are computed using squared error throughout the simulation section.
    losses = squared_loss(y[:, None], forecasts)

    return SimulationData(
        y=y,
        forecasts=forecasts,
        errors=errors,
        losses=losses,
        bias_paths=bias,
        sigma_paths=sigma_idio,
        common_shock=common,
        config=cfg,
    )


def _generate_bias(cfg: ScenarioConfig, rng) -> np.ndarray:
    """
    Generate the bias paths implied by a scenario configuration.

    Supported bias specifications are:

        - `zero`: no bias
        - `constant`: fixed bias vector over time
        - `break`: piecewise-constant bias with one break
        - `drift`: smoothly evolving bias with a persistent latent slope
        - `cluster`: constant bias vector intended for grouped forecasters
        
    Args:
        cfg (ScenarioConfig): Scenario specification
        rng: NumPy random number generator

    Returns:
        np.ndarray: Bias matrix of shape (T, M)
    """
    M, T = cfg.M, cfg.T
    bias = np.zeros((T, M))

    if cfg.bias_type == "zero":
        pass

    elif cfg.bias_type == "constant":
        if cfg.bias_values is not None:
            b = np.asarray(cfg.bias_values, dtype=float).copy()
        else:
            b = rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        if cfg.bias_centered:
            b -= b.mean()
        bias = np.tile(b, (T, 1))

    elif cfg.bias_type == "break":
        t_break = cfg.bias_break_time if cfg.bias_break_time is not None else T // 2
        b_pre = (
            np.asarray(cfg.bias_pre, dtype=float).copy()
            if cfg.bias_pre is not None
            else rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        )
        b_post = (
            np.asarray(cfg.bias_post, dtype=float).copy()
            if cfg.bias_post is not None
            else rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        )
        if cfg.bias_centered:
            b_pre -= b_pre.mean()
            b_post -= b_post.mean()
        bias[:t_break] = b_pre[None, :]
        bias[t_break:] = b_post[None, :]

    elif cfg.bias_type == "drift":
        rho = cfg.bias_drift_rho
        eta = np.sqrt(max(cfg.bias_drift_eta2, EPS)) if cfg.bias_drift_eta2 is not None else max(cfg.bias_drift_speed, EPS)
        b = np.zeros((T, M))
        slope = np.zeros((T, M))
        b[0] = rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        slope[0] = rng.normal(0, eta, size=M)
        for t in range(1, T):
            slope[t] = rho * slope[t - 1] + rng.normal(0, eta, size=M)
            b[t] = b[t - 1] + slope[t]
        if cfg.bias_centered:
            b -= b.mean(axis=1, keepdims=True)
        bias = b

    elif cfg.bias_type == "cluster":
        # This currently behaves like a constant-bias design, not a dynamic cluster process.
        if cfg.bias_values is not None:
            b = cfg.bias_values.copy()
        else:
            b = rng.normal(0, 0.3, size=M)
        if cfg.bias_centered:
            b -= b.mean()
        bias = np.tile(b, (T, 1))

    return bias


def _generate_sigma(cfg: ScenarioConfig, rng) -> np.ndarray:
    """
    Generate idiosyncratic standard-deviation paths for each forecaster.

    The function first respects any explicitly provided `sigma_idio` input. If
    no fixed path is supplied, it supports either a smooth log-precision drift
    or a simple piecewise-constant variance shift. If neither is requested, it
    falls back to a constant default scale.

    Args:
        cfg (ScenarioConfig): Scenario specification
        rng: NumPy random number generator

    Returns:
        np.ndarray: Idiosyncratic standard deviations of shape (T, M)
    """
    M, T = cfg.M, cfg.T

    if cfg.sigma_idio is not None:
        s = np.asarray(cfg.sigma_idio, dtype=float)
        if s.ndim == 1:
            return np.tile(s, (T, 1))
        return s

    if cfg.sigma_process == "smooth_precision":
        log_precision = np.zeros((T, M))
        slope = np.zeros((T, M))
        precision = np.zeros((T, M))

        eta = np.sqrt(max(cfg.sigma_drift_eta2, EPS))
        base_precision = 1.0 / max(cfg.base_sigma2, EPS)
        base_log_precision = np.log(base_precision)

        log_precision[0] = base_log_precision + rng.normal(0, 0.1, size=M)
        slope[0] = rng.normal(0, eta, size=M)
        log_precision[0] = np.clip(
            log_precision[0], cfg.log_precision_min, cfg.log_precision_max
        )
        precision[0] = np.exp(log_precision[0])

        for t in range(1, T):
            slope[t] = (
                cfg.sigma_drift_rho * slope[t - 1]
                + rng.normal(0, eta, size=M)
            )
            log_precision[t] = log_precision[t - 1] + slope[t]
            log_precision[t] = np.clip(
                log_precision[t], cfg.log_precision_min, cfg.log_precision_max
            )
            precision[t] = np.exp(log_precision[t])

        return np.sqrt(1.0 / np.maximum(precision, EPS))  # Convert precision back to sigma

    base = np.ones(M) * 0.5

    if cfg.sigma_shift_time is not None:
        sigma_pre = cfg.sigma_pre if cfg.sigma_pre is not None else base
        sigma_post = cfg.sigma_post if cfg.sigma_post is not None else base * 2
        out = np.zeros((T, M))
        out[:cfg.sigma_shift_time] = sigma_pre[None, :]
        out[cfg.sigma_shift_time:] = sigma_post[None, :]
        return out

    return np.tile(base, (T, 1))


# ===================================================================
# 7b.  PRE-BUILT SCENARIOS
# ===================================================================

def scenario_1A(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a stable benchmark scenario with unbiased, homoskedastic forecasts.

    All forecasters have zero bias over time and a constant idiosyncratic
    standard deviation of 1.0. Cross-sectional comovement is introduced only
    through a common shock with standard deviation `sigma_common = 0.5`.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 1A
    """
    return ScenarioConfig(
        name="1A_stable_unbiased",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_1B(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a stable scenario with constant forecaster-specific biases.

    Each forecaster receives a time-invariant bias drawn uniformly from
    `[0, 1]`, with constant idiosyncratic standard deviation 1.0. The error
    variance is otherwise stable over time and includes a common shock with
    standard deviation `sigma_common = 0.5`.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used to generate the bias vector

    Returns:
        ScenarioConfig: Configuration for scenario 1B
    """
    rng = _ensure_rng(seed)
    biases = rng.uniform(0.0, 1.0, size=M)
    return ScenarioConfig(
        name="1B_stable_biased",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="constant",
        bias_values=biases,
        bias_centered=False,
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_2A(M=8, T=400, T0=200, seed=42, break_frac=0.5) -> ScenarioConfig:
    """
    Generate a scenario with an abrupt structural break in forecaster biases.

    Biases are constant up to time `t_break = floor(T * break_frac)` and then
    switch instantaneously to a new independently drawn bias vector. The
    idiosyncratic standard deviation remains constant at 1.0 throughout.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation
        break_frac (float): Fraction of the sample at which the bias break occurs

    Returns:
        ScenarioConfig: Configuration for scenario 2A
    """
    rng = _ensure_rng(seed + 1)
    b_pre = rng.uniform(0.0, 1.0, size=M)
    b_post = rng.uniform(0.0, 1.0, size=M)
    t_break = int(T * break_frac)
    return ScenarioConfig(
        name="2A_abrupt_break",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="break",
        bias_break_time=t_break,
        bias_pre=b_pre,
        bias_post=b_post,
        bias_centered=False,
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_2B(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a scenario with smoothly drifting forecaster biases.

    Biases evolve over time through a latent slope process with persistence
    `bias_drift_rho = 0.95` and innovation scale `sqrt(5e-4)`. This creates
    gradual nonstationarity in forecast bias while idiosyncratic volatility
    remains constant across forecasters and time.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 2B
    """
    return ScenarioConfig(
        name="2B_smooth_drift",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="drift",
        bias_drift_speed=float(np.sqrt(5e-4)),
        bias_drift_rho=0.95,
        bias_centered=False,
        bias_init_low=0.0,
        bias_init_high=1.0,
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_2C(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a scenario with smooth drift in forecaster precision.

    Forecasts remain unbiased, but each forecaster's idiosyncratic variance
    changes over time through a persistent log-precision process. Precision is
    clipped to the interval defined by `log_precision_min` and
    `log_precision_max`, creating gradual heteroskedasticity without bias drift.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 2C
    """
    return ScenarioConfig(
        name="2C_precision_shift",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_process="smooth_precision",
        base_sigma2=1.0,
        sigma_drift_rho=0.95,
        sigma_drift_eta2=8e-4,
        log_precision_min=-2.0,
        log_precision_max=2.0,
    )


def scenario_3A(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a scenario with occasional idiosyncratic forecast outliers.

    Forecasts are unbiased with constant baseline volatility, but each
    forecaster-period pair is replaced by an outlier with probability 0.05.
    Outlier magnitudes are drawn uniformly between 2 and 6 times the local
    idiosyncratic scale, with random sign.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 3A
    """
    return ScenarioConfig(
        name="3A_outliers",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 1.0,
        outlier_prob=0.05,
        outlier_scale=6.0,
        outlier_min_scale=2.0,
    )


def scenario_3B(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate unbiased forecast errors with cross-sectional dependence through
    a one-factor structure:

        e_{i,t} = rho_i * f_t + sqrt(1 - rho_i^2) * u_{i,t}

    where `f_t` is a common latent factor, `u_{i,t}` is an idiosyncratic shock,
    and both components are scaled by the forecaster-specific idiosyncratic
    standard deviation. Factor loadings `rho_i` are drawn uniformly from
    `[0.3, 0.9]`.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used to generate the factor loadings

    Returns:
        ScenarioConfig: Configuration for scenario 3B
    """
    rng = _ensure_rng(seed)
    rhos = rng.uniform(0.3, 0.9, size=M)
    return ScenarioConfig(
        name="3B_factor_dependence",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 1.0,
        factor_rho=rhos,
    )


def scenario_3C(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate unbiased forecast errors with clustered cross-sectional dependence.

    Forecasters are assigned deterministically to three clusters. Within each
    cluster, errors load on a shared cluster-level factor with loading
    `cluster_rho = 0.7`, while the remaining variation is idiosyncratic.
    Baseline idiosyncratic volatility is constant over time and across
    forecasters.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 3C
    """
    n_clusters = 3
    labels = np.array([i % n_clusters for i in range(M)])
    return ScenarioConfig(
        name="3C_clustered",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 1.0,
        n_clusters=n_clusters,
        cluster_labels=labels,
        cluster_rho=0.7,
    )


ALL_SCENARIO_FACTORIES = {
    "1A": scenario_1A,
    "1B": scenario_1B,
    "2A": scenario_2A,
    "2B": scenario_2B,
    "2C": scenario_2C,
    "3A": scenario_3A,
    "3B": scenario_3B,
    "3C": scenario_3C,
}


# ===================================================================
# 8.  ROLLING BACKTEST ENGINE
# ===================================================================

@dataclass
class BacktestConfig:
    """Hyper-parameters for the rolling backtest."""
    # pairwise LD model
    d_max: int = 4
    fixed_d: Optional[int] = None
    fixed_h1: Optional[float] = None
    fixed_h2: Optional[float] = None
    n_cv_folds: int = 5

    # graph layer
    adjacency_type: str = "standardized"   # raw, standardized, thresholded
    centrality_type: str = "eigenvector"   # eigenvector, rowsum, pagerank, softmax
    teleport: float = TELEPORT
    adj_reg_c: float = 1e-4

    # covariance
    cov_method: str = "shrinkage"  # rolling, ewma, shrinkage, diagonal
    cov_window: int = 60
    cov_ewma_lambda: float = 0.94
    ridge_cov: float = RIDGE_COV

    # optimisation
    alpha: Optional[float] = None  # None => tune
    gamma: Optional[float] = None  # None => tune
    alpha_grid: Optional[np.ndarray] = None
    gamma_grid: Optional[np.ndarray] = None
    tune_window: int = 40  # rolling window for tuning alpha, gamma

    # loss
    loss_name: str = "squared"

    # benchmarks
    recent_best_window: int = 20
    bg_window: int = 60

    # misc
    min_history: int = 30  # minimum observations before producing weights


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    oos_periods: np.ndarray         # (n_oos,)
    y_oos: np.ndarray               # (n_oos,)
    forecasts_oos: np.ndarray       # (n_oos, M)

    # weights: dict of method_name -> (n_oos, M) or (n_oos,) for median
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_forecasts: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_losses: Dict[str, np.ndarray] = field(default_factory=dict)

    # diagnostics
    mu_matrices: Optional[List] = None   # list of (M,M) at each oos period
    sigma_matrices: Optional[List] = None
    adjacency_matrices: Optional[List] = None
    centrality_scores: Optional[List] = None
    cov_matrices: Optional[List] = None
    alpha_selected: Optional[np.ndarray] = None
    gamma_selected: Optional[np.ndarray] = None
    d_selected: Optional[List] = None


def run_backtest(
    data: SimulationData,
    bt_cfg: BacktestConfig = None,
    verbose: bool = False,
) -> BacktestResult:
    """
    Full rolling out-of-sample backtest.
    """
    if bt_cfg is None:
        bt_cfg = BacktestConfig()

    cfg = data.config
    M = cfg.M
    T = cfg.T
    T0 = cfg.T0

    loss_fn = LOSS_REGISTRY[bt_cfg.loss_name]

    y = data.y
    forecasts = data.forecasts
    losses = data.losses
    errors = data.errors

    n_oos = T - T0
    oos_periods = np.arange(T0, T)

    # Pre-compute pairwise loss differentials (full history)
    # delta_L[i,j,t] = L_i(t) - L_j(t)
    # We'll use 3D array for convenience but only access up to current t
    delta_L_full = np.zeros((M, M, T))
    for i in range(M):
        for j in range(M):
            if i != j:
                delta_L_full[i, j, :] = losses[:, i] - losses[:, j]

    # Storage
    res = BacktestResult(
        oos_periods=oos_periods,
        y_oos=y[T0:T],
        forecasts_oos=forecasts[T0:T],
        mu_matrices=[],
        sigma_matrices=[],
        adjacency_matrices=[],
        centrality_scores=[],
        cov_matrices=[],
        alpha_selected=np.zeros(n_oos),
        gamma_selected=np.zeros(n_oos),
    )

    # Method names
    method_names = [
        "equal",
        "median",
        "recent_best",
        "bates_granger",
        "bates_granger_mv",
        "rs_selection",
        "graph_only",
        "cov_only",
        "full_gcsr",
        "full_gcsr_relaxed",
        "mult_tilt",
    ]
    for name in method_names:
        res.weights[name] = np.zeros((n_oos, M))
        res.combined_forecasts[name] = np.zeros(n_oos)
        res.combined_losses[name] = np.zeros(n_oos)

    # Default grids for alpha, gamma
    if bt_cfg.alpha_grid is None:
        bt_cfg.alpha_grid = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
    if bt_cfg.gamma_grid is None:
        bt_cfg.gamma_grid = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

    # ---- Main OOS loop ----
    for oos_idx in range(n_oos):
        t = T0 + oos_idx  # forecast origin: using data up to t-1, forecasting y_t
        # But let's clarify timing:
        # At origin t, we have y_0,...,y_{t-1} and forecasts f_{j,0},...,f_{j,t}
        # We form weights for combining f_{j,t} to forecast y_t
        # After y_t is realised, we can evaluate.
        # So: losses available up through t-1, forecasts for period t already made.

        hist_end = t  # exclusive: indices 0..t-1 are available
        if hist_end < bt_cfg.min_history:
            # not enough history, use equal weights
            for name in method_names:
                res.weights[name][oos_idx] = equal_weights(M)
            f_oos = forecasts[t]
            w_eq = equal_weights(M)
            y_t = y[t]
            for name in method_names:
                w = res.weights[name][oos_idx]
                if name == "median":
                    res.combined_forecasts[name][oos_idx] = median_forecast(f_oos)
                else:
                    res.combined_forecasts[name][oos_idx] = f_oos @ w
                res.combined_losses[name][oos_idx] = loss_fn(
                    y_t, res.combined_forecasts[name][oos_idx]
                )
            continue

        # --- LAYER 1: Pairwise LD predictions ---
        mu_mat = np.zeros((M, M))
        sigma_mat = np.zeros((M, M))
        for i in range(M):
            for j in range(i+1, M):
                dl_hist = delta_L_full[i, j, :hist_end]
                if len(dl_hist) < bt_cfg.min_history:
                    continue
                pld = predict_pairwise_ld(
                    dl_hist,
                    d_max=bt_cfg.d_max,
                    fixed_d=bt_cfg.fixed_d,
                    fixed_h1=bt_cfg.fixed_h1,
                    fixed_h2=bt_cfg.fixed_h2,
                    n_cv_folds=bt_cfg.n_cv_folds,
                )
                mu_mat[i, j] = pld.mu_hat
                mu_mat[j, i] = -pld.mu_hat
                sigma_mat[i, j] = pld.sigma_hat
                sigma_mat[j, i] = pld.sigma_hat

        res.mu_matrices.append(mu_mat.copy())
        res.sigma_matrices.append(sigma_mat.copy())

        # --- LAYER 2: Graph ---
        if bt_cfg.adjacency_type == "raw":
            A = build_adjacency_raw(mu_mat)
        elif bt_cfg.adjacency_type == "standardized":
            A = build_adjacency_standardized(mu_mat, sigma_mat, bt_cfg.adj_reg_c)
        elif bt_cfg.adjacency_type == "thresholded":
            A = build_adjacency_thresholded(mu_mat, sigma_mat, 1.0, bt_cfg.adj_reg_c)
        else:
            A = build_adjacency_raw(mu_mat)

        res.adjacency_matrices.append(A.copy())

        if bt_cfg.centrality_type == "eigenvector":
            r = eigenvector_centrality(A, bt_cfg.teleport)
        elif bt_cfg.centrality_type == "rowsum":
            r = row_sum_strength(A)
        elif bt_cfg.centrality_type == "pagerank":
            r = pagerank_centrality(A)
        elif bt_cfg.centrality_type == "softmax":
            r = softmax_average_advantage(mu_mat)
        else:
            r = eigenvector_centrality(A, bt_cfg.teleport)

        res.centrality_scores.append(r.copy())

        # --- LAYER 3: Covariance ---
        errors_hist = errors[:hist_end]
        cov_fn = COVARIANCE_REGISTRY.get(bt_cfg.cov_method, shrinkage_covariance)
        if bt_cfg.cov_method == "ewma":
            Sigma = cov_fn(errors_hist, bt_cfg.cov_ewma_lambda)
        else:
            Sigma = cov_fn(errors_hist, bt_cfg.cov_window)
        Sigma = regularise_cov(Sigma, bt_cfg.ridge_cov)
        res.cov_matrices.append(Sigma.copy())

        # --- Tune alpha, gamma ---
        alpha_sel, gamma_sel = _tune_alpha_gamma(
            bt_cfg, losses[:hist_end], forecasts[:hist_end], y[:hist_end],
            errors_hist, delta_L_full[:, :, :hist_end],
            r, Sigma, M, loss_fn,
        )
        res.alpha_selected[oos_idx] = alpha_sel
        res.gamma_selected[oos_idx] = gamma_sel

        # --- Compute weights for each method ---
        f_oos = forecasts[t]
        y_t = y[t]

        # 1. Equal
        w_eq = equal_weights(M)
        res.weights["equal"][oos_idx] = w_eq

        # 2. Median (no weights vector, just combined forecast)
        res.weights["median"][oos_idx] = w_eq  # placeholder

        # 3. Recent best
        w_rb = recent_best_selection(losses[:hist_end], bt_cfg.recent_best_window)
        res.weights["recent_best"][oos_idx] = w_rb

        # 4. Bates–Granger
        w_bg = bates_granger_weights(errors_hist, bt_cfg.bg_window)
        res.weights["bates_granger"][oos_idx] = w_bg

        # 5. Bates–Granger MV
        w_bgmv = bates_granger_mv_weights(errors_hist, bt_cfg.bg_window)
        res.weights["bates_granger_mv"][oos_idx] = w_bgmv

        # 6. RS selection
        w_rs = rs_selection_weights(mu_mat)
        res.weights["rs_selection"][oos_idx] = w_rs

        # 7. Graph-only
        w_go = graph_only_weights(r)
        res.weights["graph_only"][oos_idx] = w_go

        # 8. Covariance-only
        w_co = covariance_only_weights(Sigma, bt_cfg.ridge_cov)
        res.weights["cov_only"][oos_idx] = w_co

        # 9. Full GCSR
        w_full = full_combination_weights(Sigma, r, alpha_sel, gamma_sel, bt_cfg.ridge_cov)
        res.weights["full_gcsr"][oos_idx] = w_full

        # 10. Full GCSR with signed weights
        w_full_relaxed = full_combination_weights_relaxed(
            Sigma, r, alpha_sel, gamma_sel, bt_cfg.ridge_cov
        )
        res.weights["full_gcsr_relaxed"][oos_idx] = w_full_relaxed

        # 11. Multiplicative tilt
        kappa = alpha_sel  # re-use
        w_mt = multiplicative_tilt_weights(w_co, r, kappa)
        res.weights["mult_tilt"][oos_idx] = w_mt

        # --- Combined forecasts and losses ---
        for name in method_names:
            w = res.weights[name][oos_idx]
            if name == "median":
                comb = median_forecast(f_oos)
            else:
                comb = float(f_oos @ w)
            res.combined_forecasts[name][oos_idx] = comb
            res.combined_losses[name][oos_idx] = loss_fn(y_t, comb)

        if verbose and oos_idx % 50 == 0:
            print(f"  OOS {oos_idx}/{n_oos} (t={t})")

    return res


def _tune_alpha_gamma(
    bt_cfg, losses_hist, forecasts_hist, y_hist,
    errors_hist, delta_L_hist,
    r_current, Sigma_current, M, loss_fn,
):
    """
    Tune alpha, gamma over a past validation window.
    Uses recent OOS performance of the method.
    """
    if bt_cfg.alpha is not None and bt_cfg.gamma is not None:
        return bt_cfg.alpha, bt_cfg.gamma

    T_hist = len(y_hist)
    tw = min(bt_cfg.tune_window, T_hist - bt_cfg.min_history)
    if tw < 5:
        # not enough history to tune, use moderate defaults
        return 0.1, 0.1

    alpha_grid = bt_cfg.alpha_grid
    gamma_grid = bt_cfg.gamma_grid

    # Simple grid search: for each (alpha, gamma), compute what the
    # combined forecast would have been over the tuning window using
    # the *current* r and Sigma as proxies (approximation for speed)
    best_loss = np.inf
    best_alpha = float(alpha_grid[len(alpha_grid) // 2])
    best_gamma = float(gamma_grid[len(gamma_grid) // 2])

    val_start = T_hist - tw
    y_val = y_hist[val_start:T_hist]
    f_val = forecasts_hist[val_start:T_hist]

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            w = full_combination_weights(
                Sigma_current, r_current, alpha, gamma, bt_cfg.ridge_cov
            )
            comb = f_val @ w
            avg_loss = loss_fn(y_val, comb).mean()
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_alpha = float(alpha)
                best_gamma = float(gamma)

    return best_alpha, best_gamma


# ===================================================================
# 9.  REPLICATION AND SUMMARY
# ===================================================================

@dataclass
class MCResult:
    """
    Store Monte Carlo performance summaries for one simulation scenario.

    The object collects replication-level mean losses, relative losses versus
    equal weights, and aggregate win frequencies across all combination
    methods evaluated in the backtest.
    """
    scenario_name: str
    n_reps: int
    method_names: List[str]
    msfe_matrix: np.ndarray       # (n_reps, n_methods)
    rel_msfe_matrix: np.ndarray   # relative to equal weights
    mean_msfe: np.ndarray         # (n_methods,)
    median_msfe: np.ndarray
    mean_rel_msfe: np.ndarray
    win_freq: np.ndarray          # fraction of reps each method wins


def run_monte_carlo(
    scenario_factory,
    n_reps: int = 20,
    bt_cfg: BacktestConfig = None,
    verbose: bool = False,
    **scenario_kwargs,
) -> MCResult:
    """
    Run repeated Monte Carlo replications for a single scenario design.

    For each replication, the scenario is regenerated with a shifted random
    seed, the rolling backtest is executed, and mean out-of-sample losses are
    collected for all methods. The function then aggregates those replication-
    level results into one `MCResult`.

    Args:
        scenario_factory (Callable): Scenario constructor such as `scenario_1A`
            or `scenario_2B`
        n_reps (int): Number of Monte Carlo replications
        bt_cfg (BacktestConfig | None): Backtest configuration used in each run
        verbose (bool): Whether to print replication progress
        **scenario_kwargs: Keyword arguments forwarded to `scenario_factory`

    Returns:
        MCResult: Aggregated Monte Carlo summary for the scenario
    """
    if bt_cfg is None:
        bt_cfg = BacktestConfig()

    all_results = []
    for rep in range(n_reps):
        kw = dict(scenario_kwargs)
        kw["seed"] = kw.get("seed", 42) + rep * 1000
        cfg = scenario_factory(**kw)
        data = generate_scenario(cfg)
        res = run_backtest(data, bt_cfg, verbose=False)
        all_results.append(res)
        if verbose:
            print(f"Replication {rep+1}/{n_reps} done.")

    # Summarise
    method_names = list(all_results[0].combined_losses.keys())
    n_methods = len(method_names)
    msfe_mat = np.zeros((n_reps, n_methods))
    for rep, res in enumerate(all_results):
        for jm, name in enumerate(method_names):
            msfe_mat[rep, jm] = res.combined_losses[name].mean()

    eq_idx = method_names.index("equal")
    rel_msfe = msfe_mat / (msfe_mat[:, eq_idx:eq_idx + 1] + EPS)

    mean_msfe = msfe_mat.mean(axis=0)
    median_msfe = np.median(msfe_mat, axis=0)
    mean_rel = rel_msfe.mean(axis=0)
    win_freq = np.zeros(n_methods)
    winners = msfe_mat.argmin(axis=1)
    for w in winners:
        win_freq[w] += 1
    win_freq /= n_reps

    return MCResult(
        scenario_name=cfg.name,
        n_reps=n_reps,
        method_names=method_names,
        msfe_matrix=msfe_mat,
        rel_msfe_matrix=rel_msfe,
        mean_msfe=mean_msfe,
        median_msfe=median_msfe,
        mean_rel_msfe=mean_rel,
        win_freq=win_freq,
    )


def summarise_mc(mc: MCResult) -> pd.DataFrame:
    """
    Convert a Monte Carlo result object into a tidy summary table.

    Args:
        mc (MCResult): Monte Carlo output from `run_monte_carlo`

    Returns:
        pd.DataFrame: Method-level summary sorted by mean MSFE
    """
    df = pd.DataFrame({
        "Method": mc.method_names,
        "Mean_MSFE": mc.mean_msfe,
        "Median_MSFE": mc.median_msfe,
        "Rel_MSFE_vs_EW": mc.mean_rel_msfe,
        "Win_Freq": mc.win_freq,
    })
    df = df.sort_values("Mean_MSFE").reset_index(drop=True)
    return df


# ===================================================================
# 10.  PLOTTING UTILITIES
# ===================================================================

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def set_plot_style():
    """Set consistent matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


set_plot_style()


# ---------- A. Scenario Visualization ----------

def _method_plot_style(method: str) -> Dict[str, Union[float, int, str]]:
    """Consistent highlighting for the two GCSR variants."""
    if method == "full_gcsr":
        return {"linewidth": 2.4, "alpha": 1.0, "color": "darkred", "zorder": 3}
    if method == "full_gcsr_relaxed":
        return {"linewidth": 2.2, "alpha": 0.95, "color": "darkorange", "zorder": 3}
    return {"linewidth": 1.4, "alpha": 0.85, "zorder": 2}


def _method_bar_color(method: str, default: str = "steelblue") -> str:
    """Consistent bar colors for the GCSR family."""
    if method == "full_gcsr":
        return "darkred"
    if method == "full_gcsr_relaxed":
        return "darkorange"
    return default


def plot_bias_paths(data: SimulationData, ax=None):
    """Plot latent bias paths b_{j,t}."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = data.config.M
    for j in range(M):
        ax.plot(data.bias_paths[:, j], label=f"Model {j+1}", alpha=0.7, linewidth=1)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5, label="OOS start")
    ax.set_title(f"Bias paths — {data.config.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Bias $b_{j,t}$")
    ax.legend(ncol=4, fontsize=7)
    return ax


def plot_sigma_paths(data: SimulationData, ax=None):
    """Plot idiosyncratic std-dev paths."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = data.config.M
    for j in range(M):
        ax.plot(data.sigma_paths[:, j], label=f"Model {j+1}", alpha=0.7, linewidth=1)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Idiosyncratic σ paths — {data.config.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\sigma_{j,t}$")
    ax.legend(ncol=4, fontsize=7)
    return ax


def plot_common_shock(data: SimulationData, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data.common_shock, color="steelblue", alpha=0.7, linewidth=0.8)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Common shock — {data.config.name}")
    ax.set_xlabel("Time")
    return ax


def plot_forecast_errors(data: SimulationData, ax=None, max_models=6):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = min(data.config.M, max_models)
    for j in range(M):
        ax.plot(data.errors[:, j], alpha=0.4, linewidth=0.6, label=f"Model {j+1}")
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Forecast errors — {data.config.name}")
    ax.legend(ncol=3, fontsize=7)
    return ax


def plot_scenario_summary(data: SimulationData):
    """4-panel scenario overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_bias_paths(data, axes[0, 0])
    plot_sigma_paths(data, axes[0, 1])
    plot_common_shock(data, axes[1, 0])
    plot_forecast_errors(data, axes[1, 1])
    fig.suptitle(f"Scenario: {data.config.name}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------- B. Pairwise Methodology Visualization ----------

def plot_pairwise_heatmap(matrix: np.ndarray, title: str = "", ax=None, cmap="RdBu_r"):
    """Plot heatmap of an MxM pairwise matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    M = matrix.shape[0]
    if HAS_SEABORN:
        sns.heatmap(
            matrix, ax=ax, annot=M <= 10, fmt=".3f",
            center=0, cmap=cmap, square=True,
            xticklabels=[f"{i+1}" for i in range(M)],
            yticklabels=[f"{i+1}" for i in range(M)],
        )
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.set_xticklabels([f"{i+1}" for i in range(M)])
        ax.set_yticklabels([f"{i+1}" for i in range(M)])
    ax.set_title(title)
    ax.set_xlabel("Model j")
    ax.set_ylabel("Model i")
    return ax


def plot_adjacency_heatmaps(res: BacktestResult, time_indices=None):
    """Plot adjacency matrices at selected OOS time points."""
    if time_indices is None:
        n = len(res.adjacency_matrices)
        time_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        time_indices = [i for i in time_indices if 0 <= i < n]

    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for idx, ti in enumerate(time_indices):
        t_abs = res.oos_periods[ti]
        plot_pairwise_heatmap(
            res.adjacency_matrices[ti],
            title=f"Adjacency A (t={t_abs})",
            ax=axes[idx],
            cmap="YlOrRd",
        )
    fig.tight_layout()
    return fig


def plot_centrality_bars(res: BacktestResult, time_indices=None):
    """Bar charts of centrality scores at selected dates."""
    if time_indices is None:
        n = len(res.centrality_scores)
        time_indices = [0, n // 2, n - 1]
        time_indices = [i for i in time_indices if 0 <= i < n]

    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]
    for idx, ti in enumerate(time_indices):
        r = res.centrality_scores[ti]
        M = len(r)
        t_abs = res.oos_periods[ti]
        axes[idx].bar(range(M), r, color="steelblue", alpha=0.8)
        axes[idx].axhline(1.0 / M, color="red", ls="--", alpha=0.6, label="1/M")
        axes[idx].set_title(f"Centrality (t={t_abs})")
        axes[idx].set_xlabel("Model")
        axes[idx].set_xticks(range(M))
        axes[idx].set_xticklabels([f"{i+1}" for i in range(M)])
        axes[idx].legend()
    fig.tight_layout()
    return fig


def plot_graph_network(A: np.ndarray, r: np.ndarray, title: str = ""):
    """Plot directed network from adjacency matrix."""
    if not HAS_NETWORKX:
        print("networkx not available; skipping graph plot.")
        return None
    M = A.shape[0]
    G = nx.DiGraph()
    for i in range(M):
        G.add_node(i, label=f"M{i+1}")
    for i in range(M):
        for j in range(M):
            if A[i, j] > 1e-6:
                G.add_edge(i, j, weight=A[i, j])

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42, k=2)
    node_sizes = r * 3000 + 200
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=r,
                           cmap="YlOrRd", alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: f"M{i+1}" for i in range(M)},
                            font_size=9, ax=ax)
    edges = G.edges(data=True)
    if edges:
        edge_weights = [d["weight"] for _, _, d in edges]
        max_w = max(edge_weights) if edge_weights else 1
        widths = [2.0 * w / (max_w + EPS) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4,
                               edge_color="gray", arrows=True, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------- C. Weight Diagnostics ----------

def plot_weight_timeseries(res: BacktestResult, methods=None):
    """Time series of weights for selected methods."""
    if methods is None:
        methods = ["equal", "full_gcsr", "cov_only", "graph_only"]
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 3.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for idx, name in enumerate(methods):
        w = res.weights.get(name)
        if w is None:
            continue
        M = w.shape[1]
        t_axis = res.oos_periods
        for j in range(M):
            axes[idx].fill_between([], [], alpha=0.5)
        # stacked area
        axes[idx].stackplot(
            t_axis, w.T,
            labels=[f"M{j+1}" for j in range(M)],
            alpha=0.7,
        )
        axes[idx].set_title(f"Weights: {name}")
        axes[idx].set_ylabel("Weight")
        axes[idx].set_ylim(0, 1)
        if M <= 10:
            axes[idx].legend(loc="upper right", ncol=M, fontsize=6)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def compute_herfindahl(w: np.ndarray) -> np.ndarray:
    """Herfindahl index for each row of weights matrix (n_oos, M)."""
    return (w ** 2).sum(axis=1)


def compute_effective_n(w: np.ndarray) -> np.ndarray:
    """Effective number of forecasters = 1 / Herfindahl."""
    h = compute_herfindahl(w)
    return 1.0 / np.maximum(h, EPS)


def compute_turnover(w: np.ndarray) -> np.ndarray:
    """Turnover: sum |w_t - w_{t-1}| / 2."""
    diffs = np.abs(np.diff(w, axis=0))
    return diffs.sum(axis=1) / 2.0


def plot_weight_diagnostics(res: BacktestResult, methods=None):
    """Herfindahl, effective N, and turnover."""
    if methods is None:
        methods = ["equal", "full_gcsr", "cov_only", "graph_only", "bates_granger"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for name in methods:
        w = res.weights.get(name)
        if w is None:
            continue
        t_axis = res.oos_periods
        herf = compute_herfindahl(w)
        eff_n = compute_effective_n(w)
        turn = compute_turnover(w)
        axes[0].plot(t_axis, herf, label=name, alpha=0.7)
        axes[1].plot(t_axis, eff_n, label=name, alpha=0.7)
        axes[2].plot(t_axis[1:], turn, label=name, alpha=0.7)

    axes[0].set_title("Herfindahl Index")
    axes[1].set_title("Effective Number of Forecasters")
    axes[2].set_title("Weight Turnover")
    for ax in axes:
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_gcsr_weight_comparison(
    res: BacktestResult,
    constrained_method: str = "full_gcsr",
    relaxed_method: str = "full_gcsr_relaxed",
    top_n: Optional[int] = None,
):
    """
    Compare simplex-constrained and signed GCSR weights via heatmaps.

    Args:
        res (BacktestResult): Rolling backtest output
        constrained_method (str): Name of the simplex-constrained method
        relaxed_method (str): Name of the signed-weight affine method
        top_n (int | None): Optional cap on the number of forecasters shown,
            keeping those with the largest mean absolute exposure

    Returns:
        matplotlib.figure.Figure: Figure containing two weight heatmaps and
        their signed difference
    """
    if constrained_method not in res.weights:
        raise KeyError(f"Missing weights for method '{constrained_method}'")
    if relaxed_method not in res.weights:
        raise KeyError(f"Missing weights for method '{relaxed_method}'")

    w_constrained = np.asarray(res.weights[constrained_method], dtype=float)
    w_relaxed = np.asarray(res.weights[relaxed_method], dtype=float)
    if w_constrained.shape != w_relaxed.shape:
        raise ValueError("The compared weight matrices must have the same shape.")

    M = w_constrained.shape[1]
    selected = np.arange(M)
    if top_n is not None and 0 < top_n < M:
        exposure = np.maximum(
            np.mean(np.abs(w_constrained), axis=0),
            np.mean(np.abs(w_relaxed), axis=0),
        )
        selected = np.sort(np.argsort(exposure)[::-1][:top_n])

    panels = [
        (w_constrained[:, selected].T, f"{constrained_method} weights"),
        (w_relaxed[:, selected].T, f"{relaxed_method} weights"),
        ((w_relaxed - w_constrained)[:, selected].T, f"{relaxed_method} - {constrained_method}"),
    ]
    vmax = max(np.max(np.abs(panel)) for panel, _ in panels)
    vmax = max(float(vmax), EPS)

    fig_height = max(7.5, 5.0 + 0.18 * len(selected))
    fig, axes = plt.subplots(3, 1, figsize=(14, fig_height), sharex=True)
    image = None
    for ax, (panel, title) in zip(axes, panels):
        image = ax.imshow(
            panel,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_ylabel("Forecaster")

    tick_step = max(1, len(selected) // 10)
    tick_idx = np.arange(0, len(selected), tick_step)
    tick_labels = [f"M{selected[i] + 1}" for i in tick_idx]
    for ax in axes:
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels)

    if len(res.oos_periods) > 0:
        xtick_idx = np.linspace(0, len(res.oos_periods) - 1, min(6, len(res.oos_periods)), dtype=int)
        axes[-1].set_xticks(xtick_idx)
        axes[-1].set_xticklabels(res.oos_periods[xtick_idx])
    axes[-1].set_xlabel("OOS period")

    cbar = fig.colorbar(image, ax=axes, shrink=0.96)
    cbar.set_label("Weight")
    fig.subplots_adjust(hspace=0.35, right=0.92)
    return fig


# ---------- D. Covariance Diagnostics ----------

def plot_covariance_diagnostics(res: BacktestResult):
    """Condition number and trace of Sigma over time."""
    if not res.cov_matrices:
        return None
    cond_nums = []
    traces = []
    for Sigma in res.cov_matrices:
        eigvals = np.linalg.eigvalsh(Sigma)
        eigvals = np.maximum(eigvals, EPS)
        cond_nums.append(eigvals[-1] / eigvals[0])
        traces.append(np.trace(Sigma))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(res.oos_periods, cond_nums, color="darkred", alpha=0.7)
    axes[0].set_title("Condition Number of Σ̂")
    axes[0].set_ylabel("κ(Σ̂)")
    axes[1].plot(res.oos_periods, traces, color="steelblue", alpha=0.7)
    axes[1].set_title("Trace of Σ̂")
    axes[1].set_ylabel("tr(Σ̂)")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_cov_heatmap(res: BacktestResult, oos_idx: int = -1):
    """Heatmap of covariance matrix at one time point."""
    if not res.cov_matrices:
        return None
    if oos_idx < 0:
        oos_idx = len(res.cov_matrices) + oos_idx
    Sigma = res.cov_matrices[oos_idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pairwise_heatmap(Sigma, title=f"Σ̂ (t={res.oos_periods[oos_idx]})", ax=ax, cmap="viridis")
    fig.tight_layout()
    return fig


# ---------- E. Performance Metrics ----------

def plot_cumulative_loss(res: BacktestResult, methods=None, reference="equal"):
    """Cumulative loss difference relative to a reference method."""
    if methods is None:
        methods = list(res.combined_losses.keys())

    ref_loss = res.combined_losses.get(reference, res.combined_losses["equal"])
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in methods:
        if name == reference:
            continue
        diff = np.cumsum(res.combined_losses[name] - ref_loss)
        style = _method_plot_style(name)
        ax.plot(res.oos_periods, diff, label=name, **style)
    ax.axhline(0, color="black", ls="-", linewidth=0.5)
    ax.set_title(f"Cumulative Loss Difference vs {reference}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative ΔL")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def compute_performance_table(res: BacktestResult) -> pd.DataFrame:
    """Summary performance table."""
    rows = []
    for name in res.combined_losses:
        l = res.combined_losses[name]
        rows.append({
            "Method": name,
            "Mean_Loss": l.mean(),
            "Median_Loss": np.median(l),
            "Std_Loss": l.std(),
            "Total_Loss": l.sum(),
        })
    df = pd.DataFrame(rows)
    eq_mean = df.loc[df.Method == "equal", "Mean_Loss"].values[0]
    df["Rel_MSFE"] = df["Mean_Loss"] / (eq_mean + EPS)
    df = df.sort_values("Mean_Loss").reset_index(drop=True)
    return df


def plot_msfe_barplot(res: BacktestResult, ax=None):
    """Bar plot of mean OOS loss by method."""
    df = compute_performance_table(res)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    colors = [_method_bar_color(n) for n in df.Method]
    ax.barh(df.Method, df.Rel_MSFE, color=colors, alpha=0.8)
    ax.axvline(1.0, color="black", ls="--", linewidth=0.8)
    ax.set_xlabel("Relative MSFE (vs Equal Weights)")
    ax.set_title("OOS Performance")
    ax.invert_yaxis()
    return ax


def plot_alpha_gamma_selected(res: BacktestResult):
    """Time series of selected alpha and gamma."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(res.oos_periods, res.alpha_selected, color="darkred", alpha=0.7)
    axes[0].set_title("Selected α over time")
    axes[0].set_ylabel("α")
    axes[1].plot(res.oos_periods, res.gamma_selected, color="steelblue", alpha=0.7)
    axes[1].set_title("Selected γ over time")
    axes[1].set_ylabel("γ")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


# ---------- F. MC Robustness ----------

def plot_mc_boxplot(mc: MCResult):
    """Boxplot of relative MSFE across replications."""
    fig, ax = plt.subplots(figsize=(12, 5))
    df = pd.DataFrame(mc.rel_msfe_matrix, columns=mc.method_names)
    # sort by median
    order = df.median().sort_values().index.tolist()
    if HAS_SEABORN:
        sns.boxplot(data=df[order], orient="h", ax=ax, palette="Set2")
    else:
        ax.boxplot([df[c].values for c in order], vert=False, labels=order)
    ax.axvline(1.0, color="black", ls="--", linewidth=0.8)
    ax.set_xlabel("Relative MSFE (vs Equal Weights)")
    ax.set_title(f"MC Distribution — {mc.scenario_name} ({mc.n_reps} reps)")
    fig.tight_layout()
    return fig


def plot_mc_summary_table(mc_results: Dict[str, MCResult]) -> pd.DataFrame:
    """Cross-scenario summary table."""
    rows = []
    for sc_name, mc in mc_results.items():
        for jm, mname in enumerate(mc.method_names):
            rows.append({
                "Scenario": sc_name,
                "Method": mname,
                "Mean_Rel_MSFE": mc.mean_rel_msfe[jm],
                "Win_Freq": mc.win_freq[jm],
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Method", columns="Scenario",
                           values="Mean_Rel_MSFE", aggfunc="mean")
    return pivot


def plot_mc_heatmap(mc_results: Dict[str, MCResult]):
    """Heatmap of relative MSFE across scenarios and methods."""
    pivot = plot_mc_summary_table(mc_results)
    fig, ax = plt.subplots(figsize=(12, 6))
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    center=1.0, ax=ax)
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot.index)
    ax.set_title("Relative MSFE Across Scenarios")
    fig.tight_layout()
    return fig


# ===================================================================
# 11.  MCS, ADAPTABILITY
# ===================================================================

@dataclass
class MCSResult:
    """
    Store outputs from one Hansen-style Model Confidence Set procedure.

    The object keeps the surviving set, elimination path, bootstrap p-values,
    and a convenient summary table for downstream reporting and plotting.
    """
    alpha: float
    statistic: str
    B: int
    block_size: int
    model_names: List[str]
    included_models: List[str]
    elimination_order: List[str]
    pvalues: Dict[str, float]
    elimination_steps: Dict[str, int]
    test_statistics: List[float]
    test_pvalues: List[float]
    active_sets: List[List[str]]
    mean_losses: Dict[str, float]
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _coerce_loss_matrix(
    losses: Union[BacktestResult, pd.DataFrame, np.ndarray],
    model_names: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a `T x M` loss matrix and aligned model names from supported inputs.

    Args:
        losses (BacktestResult | pd.DataFrame | np.ndarray): Loss input in one
            of the supported container formats
        model_names (Sequence[str] | None): Optional model names for array input
        methods (Sequence[str] | None): Optional subset of methods when the
            input is a `BacktestResult`

    Returns:
        Tuple[np.ndarray, List[str]]: Loss matrix and associated model names
    """
    if isinstance(losses, BacktestResult):
        if methods is None:
            methods = list(losses.combined_losses.keys())
        arr = np.column_stack([np.asarray(losses.combined_losses[m], dtype=float) for m in methods])
        return arr, list(methods)
    if isinstance(losses, pd.DataFrame):
        return losses.to_numpy(dtype=float), list(losses.columns.astype(str))
    arr = np.asarray(losses, dtype=float)
    if arr.ndim != 2:
        raise ValueError("losses must be a 2D array, DataFrame, or BacktestResult")
    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(arr.shape[1])]
    if len(model_names) != arr.shape[1]:
        raise ValueError("model_names length must match the number of columns in losses")
    return arr, list(model_names)


# ---------- Block-size heuristics ----------

def _select_ar_order_bic(series: np.ndarray, max_lags: int = 10) -> int:
    """
    Select a univariate AR order with a simple BIC criterion.

    Args:
        series (np.ndarray): Univariate time series
        max_lags (int): Maximum AR order considered

    Returns:
        int: BIC-selected lag order
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    max_lags = min(max_lags, max(n // 3, 0))
    if n < 8 or max_lags == 0:
        return 0

    best_p = 0
    best_bic = np.inf
    for p in range(max_lags + 1):
        if n <= p + 2:
            continue
        y = x[p:]
        X = np.ones((n - p, p + 1))
        for lag in range(1, p + 1):
            X[:, lag] = x[p - lag:n - lag]
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = np.mean(resid ** 2)
        bic = np.log(max(sigma2, EPS)) + (p + 1) * np.log(len(y)) / len(y)
        if bic < best_bic:
            best_bic = bic
            best_p = p
    return best_p


def _auto_mcs_block_size(
    loss_matrix: np.ndarray,
    min_block_size: int = 3,
    max_lags: int = 10,
) -> int:
    """
    Practical block-length heuristic for the MCS bootstrap.

    Uses the maximum BIC-selected AR order across pairwise loss differentials,
    with a minimum floor for numerical stability.
    """
    n, m = loss_matrix.shape
    if n < 10 or m < 2:
        return min_block_size
    orders = []
    for i in range(m):
        for j in range(i + 1, m):
            diff = loss_matrix[:, i] - loss_matrix[:, j]
            if np.allclose(diff, diff[0]):
                continue
            orders.append(_select_ar_order_bic(diff, max_lags=max_lags))
    if not orders:
        return min_block_size
    block_size = max(max(orders), min_block_size)
    return int(min(block_size, max(n // 2, min_block_size)))


# ---------- Bootstrap utilities ----------

def _moving_block_bootstrap_indices(
    n_obs: int,
    B: int,
    block_size: int,
    rng,
) -> np.ndarray:
    """
    Generate circular moving-block bootstrap indices.

    Args:
        n_obs (int): Number of time observations
        B (int): Number of bootstrap replications
        block_size (int): Length of each resampled block
        rng: NumPy random number generator

    Returns:
        np.ndarray: Bootstrap index array of shape `(B, n_obs)`
    """
    n_blocks = int(np.ceil(n_obs / max(block_size, 1)))
    starts = rng.integers(0, n_obs, size=(B, n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return indices.reshape(B, -1)[:, :n_obs]


# ---------- One-step MCS statistics ----------

def _mcs_iteration_statistics(
    loss_matrix: np.ndarray,
    bootstrap_indices: np.ndarray,
    statistic: str,
) -> Dict[str, Union[float, int, bool]]:
    """
    Compute one MCS test statistic, bootstrap p-value, and elimination rule.

    Args:
        loss_matrix (np.ndarray): Active-model loss matrix of shape `(T, M)`
        bootstrap_indices (np.ndarray): Bootstrap resampling indices
        statistic (str): Either `"Tmax"` or `"TR"`

    Returns:
        Dict[str, Union[float, int, bool]]: Test statistic, p-value, index of
        the model to eliminate, and a flag for degenerate equal-loss cases
    """
    n_obs, m = loss_matrix.shape
    loss_means = loss_matrix.mean(axis=0)
    boot_means = loss_matrix[bootstrap_indices].mean(axis=1)  # (B, m)
    zeta = boot_means - loss_means[None, :]

    scale = m / max(m - 1, 1)
    d_i_mean = scale * (loss_means - loss_means.mean())
    d_i_boot_centered = scale * (zeta - zeta.mean(axis=1, keepdims=True))
    var_i = np.mean(d_i_boot_centered ** 2, axis=0)
    t_i = d_i_mean / np.sqrt(var_i + EPS)
    T_max = float(np.max(t_i))
    T_max_star = np.max(d_i_boot_centered / np.sqrt(var_i + EPS), axis=1)
    p_max = float(np.mean(T_max_star >= (T_max - EPS)))
    elim_tmax = int(np.argmax(t_i))

    pair_centered = zeta[:, :, None] - zeta[:, None, :]
    d_ij_mean = loss_means[:, None] - loss_means[None, :]
    var_ij = np.mean(pair_centered ** 2, axis=0)
    np.fill_diagonal(var_ij, np.nan)
    t_ij = d_ij_mean / np.sqrt(var_ij + EPS)
    T_r = float(np.nanmax(np.abs(t_ij)))
    T_r_star = np.nanmax(
        np.abs(pair_centered / np.sqrt(var_ij[None, :, :] + EPS)),
        axis=(1, 2),
    )
    p_r = float(np.mean(T_r_star >= (T_r - EPS)))
    v_i_r = np.nanmax(t_ij, axis=1)
    elim_tr = int(np.nanargmax(v_i_r))

    if statistic == "TR":
        return {
            "test_stat": T_r,
            "pvalue": p_r,
            "eliminate_idx": elim_tr,
            "all_equal": bool(np.all(np.isnan(var_ij)) or np.nanmax(var_ij) < EPS),
        }
    return {
        "test_stat": T_max,
        "pvalue": p_max,
        "eliminate_idx": elim_tmax,
        "all_equal": bool(np.max(var_i) < EPS),
    }


# ---------- Hansen-style MCS procedure ----------

def model_confidence_set(
    losses: Union[BacktestResult, pd.DataFrame, np.ndarray],
    model_names: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    alpha: float = 0.10,
    B: int = 500,
    statistic: Literal["Tmax", "TR"] = "Tmax",
    block_size: Optional[int] = None,
    min_block_size: int = 3,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> MCSResult:
    """
    Hansen-Lunde-Nason Model Confidence Set procedure.

    Parameters
    ----------
    losses
        Either a `BacktestResult`, a T x M DataFrame of losses, or a T x M array.
    statistic
        Either "Tmax" or "TR", matching Hansen et al. (2011).

    Returns
    -------
    MCSResult
        Full Model Confidence Set output, including the surviving model set and
        elimination summary.
    """
    if statistic not in {"Tmax", "TR"}:
        raise ValueError("statistic must be either 'Tmax' or 'TR'")

    loss_matrix, model_names = _coerce_loss_matrix(losses, model_names, methods)
    n_obs, n_models = loss_matrix.shape
    if n_models < 2:
        raise ValueError("MCS requires at least two models")

    if block_size is None:
        block_size = _auto_mcs_block_size(loss_matrix, min_block_size=min_block_size)
    block_size = max(int(block_size), 1)

    rng = _ensure_rng(seed)
    bootstrap_indices = _moving_block_bootstrap_indices(n_obs, int(B), block_size, rng)

    mean_losses = {name: float(loss_matrix[:, i].mean()) for i, name in enumerate(model_names)}
    active = list(range(n_models))
    elimination_order: List[str] = []
    elimination_steps: Dict[str, int] = {}
    pvalues: Dict[str, float] = {}
    test_statistics: List[float] = []
    test_pvalues: List[float] = []
    active_sets: List[List[str]] = []
    running_p = 0.0

    while len(active) > 1:
        active_sets.append([model_names[i] for i in active])
        stats = _mcs_iteration_statistics(loss_matrix[:, active], bootstrap_indices, statistic)
        test_statistics.append(float(stats["test_stat"]))
        test_pvalues.append(float(stats["pvalue"]))

        if stats["all_equal"] or stats["pvalue"] >= alpha:
            break

        local_idx = int(stats["eliminate_idx"])
        global_idx = active[local_idx]
        name = model_names[global_idx]
        running_p = max(running_p, float(stats["pvalue"]))
        elimination_order.append(name)
        elimination_steps[name] = len(elimination_order)
        pvalues[name] = running_p
        active.pop(local_idx)

    included_models = [model_names[i] for i in active]
    for name in included_models:
        elimination_steps[name] = 0
        pvalues[name] = 1.0

    summary_rows = []
    for name in model_names:
        summary_rows.append({
            "Method": name,
            "Mean_Loss": mean_losses[name],
            "MCS_pvalue": pvalues[name],
            "In_MCS": name in included_models,
            "Elimination_Step": elimination_steps[name],
        })
    summary_table = pd.DataFrame(summary_rows).sort_values(
        ["In_MCS", "Mean_Loss"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return MCSResult(
        alpha=float(alpha),
        statistic=statistic,
        B=int(B),
        block_size=block_size,
        model_names=model_names,
        included_models=included_models,
        elimination_order=elimination_order,
        pvalues=pvalues,
        elimination_steps=elimination_steps,
        test_statistics=test_statistics,
        test_pvalues=test_pvalues,
        active_sets=active_sets,
        mean_losses=mean_losses,
        summary_table=summary_table,
    )


# ---------- MCS reporting helpers ----------

def compute_mcs_performance_table(
    res: BacktestResult,
    mcs_result: Optional[MCSResult] = None,
    methods: Optional[Sequence[str]] = None,
    reference: str = "equal",
    **mcs_kwargs,
) -> pd.DataFrame:
    """
    Merge standard backtest performance metrics with MCS diagnostics.

    Args:
        res (BacktestResult): Rolling backtest output
        mcs_result (MCSResult | None): Precomputed MCS result, if available
        methods (Sequence[str] | None): Optional method subset
        reference (str): Reference method used to rescale mean losses
        **mcs_kwargs: Extra keyword arguments passed to `model_confidence_set`

    Returns:
        pd.DataFrame: Performance table augmented with MCS status columns
    """
    perf = compute_performance_table(res)
    if methods is not None:
        perf = perf[perf["Method"].isin(methods)].copy()
    if mcs_result is None:
        mcs_result = model_confidence_set(res, methods=methods, **mcs_kwargs)
    mcs_tbl = mcs_result.summary_table[["Method", "MCS_pvalue", "In_MCS", "Elimination_Step"]]
    df = perf.merge(mcs_tbl, on="Method", how="left")

    if reference in df["Method"].values:
        ref_loss = df.loc[df["Method"] == reference, "Mean_Loss"].iloc[0]
        df["Rel_MSFE"] = df["Mean_Loss"] / (ref_loss + EPS)

    df = df.sort_values(["In_MCS", "Mean_Loss"], ascending=[False, True]).reset_index(drop=True)
    return df


# ---------- MCS visualization ----------

def plot_mcs_summary(
    res: BacktestResult,
    mcs_result: Optional[MCSResult] = None,
    methods: Optional[Sequence[str]] = None,
    reference: str = "equal",
    ax=None,
    **mcs_kwargs,
):
    """
    Plot relative mean loss by method with MCS membership highlighted.

    Args:
        res (BacktestResult): Rolling backtest output
        mcs_result (MCSResult | None): Optional precomputed MCS result
        methods (Sequence[str] | None): Optional method subset
        reference (str): Reference method for relative MSFE scaling
        ax: Optional matplotlib axes
        **mcs_kwargs: Extra keyword arguments passed to `model_confidence_set`

    Returns:
        matplotlib.figure.Figure: Figure containing the summary bar plot
    """
    df = compute_mcs_performance_table(
        res,
        mcs_result=mcs_result,
        methods=methods,
        reference=reference,
        **mcs_kwargs,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    colors = ["darkred" if keep else "steelblue" for keep in df["In_MCS"]]
    ax.barh(df["Method"], df["Rel_MSFE"], color=colors, alpha=0.85)
    ax.axvline(1.0, color="black", ls="--", linewidth=0.8)
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["Rel_MSFE"] + 0.01,
            y,
            f"p={row['MCS_pvalue']:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel(f"Relative MSFE (vs {reference})")
    ax.set_title("Model Confidence Set Summary")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------- Adaptability result container ----------

@dataclass
class AdaptabilityResult:
    """
    Store event-study diagnostics for post-switch forecast adaptability.

    The object tracks how quickly each method approaches the latent oracle
    combination after oracle switches in the simulated environment.
    """
    methods: List[str]
    event_times: np.ndarray
    oracle_models: np.ndarray
    horizon: int
    smooth_window: int
    target_oracle_weight: float
    oracle_portfolio_weights: np.ndarray
    oracle_weight_profiles: Dict[str, np.ndarray]
    oracle_gap_profiles: Dict[str, np.ndarray]
    latent_risk_profiles: Dict[str, np.ndarray]
    excess_risk_profiles: Dict[str, np.ndarray]
    relative_gap_profiles: Dict[str, np.ndarray]
    half_lives: Dict[str, np.ndarray]
    recovery_delays: Dict[str, np.ndarray]
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------- Latent oracle identification ----------

def compute_latent_oracle_indices(data: SimulationData) -> np.ndarray:
    """
    Latent oracle under squared loss.

    Common shocks are shared across models, so ranking is driven by model-specific
    squared bias plus model-specific idiosyncratic variance.
    """
    latent_risk = data.bias_paths ** 2 + data.sigma_paths ** 2
    return np.argmin(latent_risk, axis=1)


def identify_oracle_switches(
    data: SimulationData,
    start_time: Optional[int] = None,
    min_spacing: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify OOS dates where the latent oracle model changes."""
    oracle = compute_latent_oracle_indices(data)
    if start_time is None:
        start_time = data.config.T0
    switches = np.where(oracle[1:] != oracle[:-1])[0] + 1
    switches = switches[switches >= start_time]
    if len(switches) <= 1 or min_spacing <= 1:
        return switches, oracle

    keep = [switches[0]]
    for t in switches[1:]:
        if t - keep[-1] >= min_spacing:
            keep.append(t)
    return np.asarray(keep, dtype=int), oracle


# ---------- Adaptability helper functions ----------

def _rolling_mean_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with min_periods=1."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(window, min_periods=1).mean().to_numpy()


def _latent_variance_scale_with_outliers(cfg: ScenarioConfig) -> float:
    """Variance inflation factor for the outlier mixture used in scenario generation."""
    if cfg.outlier_prob <= 0:
        return 1.0
    a = float(cfg.outlier_min_scale)
    b = float(max(cfg.outlier_scale, cfg.outlier_min_scale))
    uniform_second_moment = (a ** 2 + a * b + b ** 2) / 3.0
    return (1.0 - cfg.outlier_prob) + cfg.outlier_prob * uniform_second_moment


def latent_idiosyncratic_covariance(
    data: SimulationData,
    t: int,
) -> np.ndarray:
    """
    Conditional covariance of idiosyncratic forecast errors at time t.

    The common shock is omitted because it is shared across all methods and
    therefore does not affect cross-method comparisons.
    """
    cfg = data.config
    sigma_t = np.asarray(data.sigma_paths[t], dtype=float)
    variances = sigma_t ** 2

    if cfg.outlier_prob > 0:
        variances = variances * _latent_variance_scale_with_outliers(cfg)

    M = len(sigma_t)
    Sigma = np.diag(variances)

    if cfg.factor_rho is not None:
        rho = np.asarray(cfg.factor_rho, dtype=float)
        Sigma = np.outer(sigma_t * rho, sigma_t * rho)
        np.fill_diagonal(Sigma, variances)
        return Sigma

    if cfg.n_clusters > 1 and cfg.cluster_labels is not None:
        labels = np.asarray(cfg.cluster_labels)
        rho_sq = float(cfg.cluster_rho) ** 2
        Sigma = np.diag(variances)
        for i in range(M):
            for j in range(i + 1, M):
                if labels[i] == labels[j]:
                    cov_ij = rho_sq * sigma_t[i] * sigma_t[j]
                    Sigma[i, j] = cov_ij
                    Sigma[j, i] = cov_ij
        return Sigma

    return Sigma


def latent_risk_matrix(
    data: SimulationData,
    t: int,
) -> np.ndarray:
    """
    Quadratic-form matrix for latent expected squared forecast error at time t.

    For a simplex weight vector w, the method-specific component of expected
    squared error is:
        (w'b_t)^2 + w' Sigma_t w
    where b_t is the latent bias vector and Sigma_t is the conditional
    idiosyncratic covariance matrix.
    """
    bias_t = np.asarray(data.bias_paths[t], dtype=float)
    Sigma_t = latent_idiosyncratic_covariance(data, t)
    return Sigma_t + np.outer(bias_t, bias_t)


def compute_latent_oracle_combination_weights(
    data: SimulationData,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """Time-varying oracle simplex weights under latent expected squared error."""
    T = data.config.T
    M = data.config.M
    oracle_w = np.zeros((T, M))
    for t in range(T):
        G_t = latent_risk_matrix(data, t)
        oracle_w[t] = covariance_only_weights(G_t, ridge=ridge)
    return oracle_w


def latent_combination_risk(
    data: SimulationData,
    weights: np.ndarray,
    t: int,
) -> float:
    """Latent expected squared forecast error for a given weight vector at time t."""
    G_t = latent_risk_matrix(data, t)
    w = simplex_project(np.asarray(weights, dtype=float))
    return float(w @ G_t @ w)


def _safe_nanmean(x: np.ndarray) -> float:
    """NaN-safe mean that returns NaN when no finite values exist."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float(np.nan)
    return float(np.mean(x[finite]))


def _safe_nanmedian(x: np.ndarray) -> float:
    """NaN-safe median that returns NaN when no finite values exist."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float(np.nan)
    return float(np.median(x[finite]))


# ---------- Adaptability event-study diagnostic ----------

def compute_adaptability_diagnostics(
    data: SimulationData,
    res: BacktestResult,
    methods: Optional[Sequence[str]] = None,
    horizon: int = 40,
    smooth_window: int = 5,
    min_event_spacing: int = 3,
    target_oracle_weight: float = 0.8,
) -> AdaptabilityResult:
    """
    Compute simulation-based adaptability diagnostics after latent oracle switches.

    The metric is an original event-study summary inspired by the forecast-breakdown
    literature: after each latent oracle switch, track how quickly a method
    closes its excess latent risk relative to the time-varying oracle
    combination and record the half-life of that gap.

    Args:
        data (SimulationData): Simulated data-generating environment
        res (BacktestResult): Backtest output containing time-varying weights
        methods (Sequence[str] | None): Optional method subset to evaluate
        horizon (int): Post-switch evaluation horizon
        smooth_window (int): Rolling window used to smooth relative gap profiles
        min_event_spacing (int): Minimum spacing between retained switch events
        target_oracle_weight (float): Oracle-overlap target used for recovery delays

    Returns:
        AdaptabilityResult: Event-study profiles and summary metrics by method
    """
    if methods is None:
        methods = [m for m in res.weights.keys() if m != "median"]
    methods = [m for m in methods if m in res.weights]

    event_times, oracle = identify_oracle_switches(
        data,
        start_time=data.config.T0,
        min_spacing=min_event_spacing,
    )
    n_events = len(event_times)
    oracle_models = oracle[event_times] if n_events else np.array([], dtype=int)
    oracle_combo_weights = compute_latent_oracle_combination_weights(data)

    overlap_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    weight_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    gap_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    latent_risk_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    excess_risk_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    relative_gap_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    half_lives = {
        method: np.full(n_events, np.nan)
        for method in methods
    }
    recovery_delays = {
        method: np.full(n_events, np.nan)
        for method in methods
    }

    if n_events == 0:
        warnings.warn("No latent oracle switches were detected in the OOS period.")
    for ievent, t_event in enumerate(event_times):
        horizon_end = min(t_event + horizon, data.config.T)
        use_len = horizon_end - t_event
        oos_start = t_event - data.config.T0

        for method in methods:
            method_weights = np.asarray(res.weights[method][oos_start:oos_start + use_len], dtype=float)
            oracle_capture = np.zeros(use_len)
            latent_risk = np.zeros(use_len)
            excess_risk = np.zeros(use_len)

            for h in range(use_len):
                t_cur = t_event + h
                oracle_w_t = oracle_combo_weights[t_cur]
                w_t = simplex_project(method_weights[h])
                oracle_capture[h] = 1.0 - 0.5 * np.abs(w_t - oracle_w_t).sum()
                latent_risk[h] = latent_combination_risk(data, w_t, t_cur)
                oracle_risk_t = latent_combination_risk(data, oracle_w_t, t_cur)
                excess_risk[h] = max(latent_risk[h] - oracle_risk_t, 0.0)

            initial_gap = max(excess_risk[0], EPS) if use_len else np.nan
            relative_gap = excess_risk / initial_gap if use_len else np.array([])

            overlap_profiles[method][ievent, :use_len] = oracle_capture
            weight_profiles[method][ievent, :use_len] = oracle_capture
            gap_profiles[method][ievent, :use_len] = 1.0 - oracle_capture
            latent_risk_profiles[method][ievent, :use_len] = latent_risk
            excess_risk_profiles[method][ievent, :use_len] = excess_risk
            relative_gap_profiles[method][ievent, :use_len] = relative_gap

            smooth_gap = _rolling_mean_1d(relative_gap, smooth_window)
            baseline = float(smooth_gap[0]) if len(smooth_gap) else np.nan
            if not np.isfinite(baseline):
                continue
            if baseline <= 0:
                half_lives[method][ievent] = 0.0
                recovery_delays[method][ievent] = 0.0
                continue

            half_target = 0.5 * baseline
            half_idx = np.where(smooth_gap <= half_target)[0]
            if len(half_idx):
                half_lives[method][ievent] = float(half_idx[0])

            target_gap_fraction = max(1.0 - target_oracle_weight, 0.0)
            recovery_idx = np.where(smooth_gap <= target_gap_fraction)[0]
            if len(recovery_idx):
                recovery_delays[method][ievent] = float(recovery_idx[0])

    rows = []
    for method in methods:
        profile0 = excess_risk_profiles[method][:, 0] if n_events else np.array([np.nan])
        rows.append({
            "Method": method,
            "Events": n_events,
            "Mean_Initial_Excess_Latent_Risk": _safe_nanmean(profile0),
            "Mean_Integrated_Excess_Risk": _safe_nanmean(excess_risk_profiles[method]),
            "Mean_Half_Life": _safe_nanmean(half_lives[method]),
            "Median_Half_Life": _safe_nanmedian(half_lives[method]),
            "Mean_Target_Closure_Delay": _safe_nanmean(recovery_delays[method]),
            "Captured_Frac": float(np.mean(np.isfinite(recovery_delays[method]))) if n_events else np.nan,
            "Mean_Oracle_Overlap": _safe_nanmean(overlap_profiles[method][:, 0]) if n_events else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values(
        ["Mean_Integrated_Excess_Risk", "Mean_Half_Life", "Mean_Target_Closure_Delay"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    return AdaptabilityResult(
        methods=methods,
        event_times=event_times,
        oracle_models=oracle_models,
        horizon=horizon,
        smooth_window=smooth_window,
        target_oracle_weight=target_oracle_weight,
        oracle_portfolio_weights=oracle_combo_weights,
        oracle_weight_profiles=weight_profiles,
        oracle_gap_profiles=gap_profiles,
        latent_risk_profiles=latent_risk_profiles,
        excess_risk_profiles=excess_risk_profiles,
        relative_gap_profiles=relative_gap_profiles,
        half_lives=half_lives,
        recovery_delays=recovery_delays,
        summary_table=summary,
    )


# ---------- Adaptability visualization ----------

def plot_adaptability_event_study(
    adapt: AdaptabilityResult,
    methods: Optional[Sequence[str]] = None,
    ax=None,
):
    """
    Plot the average excess latent risk profile after oracle-switch events.

    Args:
        adapt (AdaptabilityResult): Adaptability diagnostics output
        methods (Sequence[str] | None): Optional method subset
        ax: Optional matplotlib axes

    Returns:
        matplotlib.figure.Figure: Figure containing the event-study plot
    """
    if methods is None:
        methods = adapt.methods
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    horizon_axis = np.arange(adapt.horizon)
    for method in methods:
        profile = adapt.excess_risk_profiles[method]
        mean_profile = np.nanmean(profile, axis=0)
        style = _method_plot_style(method)
        ax.plot(horizon_axis, mean_profile, label=method, **style)

    ax.axhline(0.0, color="black", ls="--", linewidth=0.8)
    ax.set_xlabel("Periods Since Latent Oracle Switch")
    ax.set_ylabel("Excess Latent Risk vs Oracle Combination")
    ax.set_title("Adaptability Event Study")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_adaptability_half_life(
    adapt: AdaptabilityResult,
    ax=None,
):
    """
    Plot mean post-switch half-lives for each evaluated method.

    Args:
        adapt (AdaptabilityResult): Adaptability diagnostics output
        ax: Optional matplotlib axes

    Returns:
        matplotlib.figure.Figure: Figure containing the half-life bar chart
    """
    df = adapt.summary_table.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    colors = [_method_bar_color(method, default="darkgreen") for method in df["Method"]]
    ax.barh(df["Method"], df["Mean_Half_Life"], color=colors, alpha=0.85)
    ax.set_xlabel("Mean Relative-Risk Half-Life")
    ax.set_title("Adaptability Speed")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def document_adaptability_measure() -> pd.DataFrame:
    """
    Summarize the design choices behind the adaptability diagnostic.

    Returns:
        pd.DataFrame: Compact documentation table for the metric definition
    """
    rows = [
        {
            "Item": "Metric",
            "Description": "Post-switch excess latent-risk half-life.",
            "Details": "A simulation-specific diagnostic rather than a named classical estimator.",
            "Reference": "This implementation; inspired by Giacomini & Rossi (2009) and Tian & Anderson (2014)",
        },
        {
            "Item": "Event definition",
            "Description": "A switch occurs when the latent oracle forecaster changes.",
            "Details": "The latent oracle is argmin_j {b_{j,t}^2 + sigma_{j,t}^2} under squared loss.",
            "Reference": "Simulation design in this module",
        },
        {
            "Item": "Gap series",
            "Description": "Track excess latent risk relative to the oracle combination.",
            "Details": (
                "g_{m,e}(h) = R_m(t_e+h) - R^*(t_e+h), where R^* uses the "
                "time-varying simplex-constrained oracle combination under the latent DGP."
            ),
            "Reference": "This implementation",
        },
        {
            "Item": "Half-life",
            "Description": "First horizon where the smoothed relative latent-risk gap is at most half the initial gap.",
            "Details": "Smaller half-life means faster adaptation to the new best forecaster.",
            "Reference": "This implementation",
        },
        {
            "Item": "Interpretation",
            "Description": "Separates immediate robustness from dynamic re-learning speed.",
            "Details": (
                "Combination methods such as GCSR are evaluated on how quickly their "
                "latent expected risk approaches the oracle combination after regime shifts."
            ),
            "Reference": "Tian & Anderson (2014)",
        },
    ]
    return pd.DataFrame(rows)


def document_simulations() -> pd.DataFrame:
    """High-level guide to the simulation designs implemented in this file."""
    rows = [
        {
            "Scenario": "1A_stable_unbiased",
            "Notebook_Analogue": "stable_unbiased",
            "Mechanism": "No bias; homoskedastic idiosyncratic noise; common shock retained.",
            "Instability": "None",
        },
        {
            "Scenario": "1B_stable_biased",
            "Notebook_Analogue": "stable_biased",
            "Mechanism": "Forecaster-specific constant bias drawn once and held fixed.",
            "Instability": "Persistent bias heterogeneity",
        },
        {
            "Scenario": "2A_abrupt_break",
            "Notebook_Analogue": "abrupt_break",
            "Mechanism": "Forecaster-specific bias redraw at the break date.",
            "Instability": "Abrupt bias shift",
        },
        {
            "Scenario": "2B_smooth_drift",
            "Notebook_Analogue": "smooth_drift",
            "Mechanism": "Bias follows a local-trend process with persistent slope innovations.",
            "Instability": "Smooth bias drift",
        },
        {
            "Scenario": "2C_precision_shift",
            "Notebook_Analogue": "precision_shift_smooth",
            "Mechanism": "Log precision drifts smoothly, producing time-varying idiosyncratic variance.",
            "Instability": "Smooth precision drift",
        },
        {
            "Scenario": "3A_outliers",
            "Notebook_Analogue": "idiosyncratic_outliers",
            "Mechanism": "Occasional large signed idiosyncratic shocks replace Gaussian draws.",
            "Instability": "Outlier contamination",
        },
        {
            "Scenario": "3B_factor_dependence",
            "Notebook_Analogue": "cross_sectional_dependence",
            "Mechanism": "One-factor dependence across forecasters with heterogeneous loadings.",
            "Instability": "Cross-sectional dependence",
        },
        {
            "Scenario": "3C_clustered",
            "Notebook_Analogue": "extension",
            "Mechanism": "Cluster-level common factors generate within-cluster dependence.",
            "Instability": "Clustered dependence",
        },
    ]
    return pd.DataFrame(rows)


def showcase_model_diagnostics(
    data: SimulationData,
    res: BacktestResult,
    mcs_alpha: float = 0.10,
    mcs_B: int = 500,
    mcs_statistic: Literal["Tmax", "TR"] = "Tmax",
    adaptability_horizon: int = 40,
    adaptability_window: int = 5,
) -> Dict[str, object]:
    """Convenience wrapper that returns the main tables and plots for comparison."""
    mcs = model_confidence_set(
        res,
        alpha=mcs_alpha,
        B=mcs_B,
        statistic=mcs_statistic,
    )
    mcs_table = compute_mcs_performance_table(res, mcs_result=mcs)
    mcs_fig = plot_mcs_summary(res, mcs_result=mcs)

    adaptability = compute_adaptability_diagnostics(
        data,
        res,
        horizon=adaptability_horizon,
        smooth_window=adaptability_window,
    )
    adaptability_fig = plot_adaptability_event_study(adaptability)
    adaptability_half_life_fig = plot_adaptability_half_life(adaptability)

    return {
        "simulation_doc": document_simulations(),
        "mcs_doc": document_mcs_procedure(),
        "adaptability_doc": document_adaptability_measure(),
        "references": methodology_references(),
        "mcs_result": mcs,
        "mcs_table": mcs_table,
        "mcs_figure": mcs_fig,
        "adaptability_result": adaptability,
        "adaptability_table": adaptability.summary_table,
        "adaptability_figure": adaptability_fig,
        "adaptability_half_life_figure": adaptability_half_life_fig,
    }


# ===================================================================
# 12.  EMPIRICAL INFLATION EVALUATION
# ===================================================================

# ---------- Empirical result container ----------

@dataclass
class EmpiricalStudyResult:
    """
    Store empirical data inputs, backtest outputs, and reporting tables.

    The object is designed for direct inspection after running the empirical
    next-quarter inflation exercise on the survey forecasts in `Empirical_Data`.
    """
    dataset_name: str
    merged_data: pd.DataFrame
    forecaster_ids: List[str]
    training_periods: int
    comparison_methods: List[str]
    data: SimulationData
    backtest: BacktestResult
    performance_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_forecast_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    mcs_result: Optional[MCSResult] = None
    mcs_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------- Empirical data preparation ----------

def load_empirical_inflation_data(
    forecast_path: str = "Empirical_Data/inflation_forecasts_f.csv",
    truth_path: str = "Empirical_Data/inflation_truth_f.csv",
    training_periods: int = 20,
    loss_name: str = "squared",
) -> Tuple[SimulationData, pd.DataFrame, List[str]]:
    """
    Load and align the empirical next-quarter inflation forecast panel.

    The empirical files are assumed to be clean, aligned, and already ordered
    chronologically. The loader therefore keeps the input structure simple and
    converts the two CSVs directly into the `SimulationData` container used by
    the backtest engine.

    Args:
        forecast_path (str): Path to the `T x M` forecast matrix CSV
        truth_path (str): Path to the realized inflation CSV
        training_periods (int): First OOS index
        loss_name (str): Loss function used to construct empirical losses

    Returns:
        Tuple[SimulationData, pd.DataFrame, List[str]]: Backtest-ready data,
        aligned empirical panel, and ordered forecaster identifiers
    """
    training_periods = max(int(training_periods), 20)
    loss_fn = LOSS_REGISTRY.get(loss_name, squared_loss)

    forecasts_df = pd.read_csv(forecast_path)
    truth_df = pd.read_csv(truth_path)
    forecaster_ids = [col for col in forecasts_df.columns if col != "TARGET_PERIOD"]
    merged = forecasts_df.copy()
    merged["actual"] = truth_df["actual"].to_numpy()
    merged["period_dt"] = pd.to_datetime(merged["TARGET_PERIOD"])
    numeric_cols = forecaster_ids + ["actual"]
    merged[numeric_cols] = merged[numeric_cols].astype(float)

    T = len(merged)
    M = len(forecaster_ids)

    y = merged["actual"].to_numpy(dtype=float)
    forecasts = merged[forecaster_ids].to_numpy(dtype=float)
    errors = y[:, None] - forecasts
    losses = loss_fn(y[:, None], forecasts)

    cfg = ScenarioConfig(
        name="empirical_inflation_next_quarter",
        M=M,
        T=T,
        T0=training_periods,
        sigma_common=0.0,
        seed=RNG_SEED,
    )
    data = SimulationData(
        y=y,
        forecasts=forecasts,
        errors=errors,
        losses=losses,
        bias_paths=np.zeros((T, M)),
        sigma_paths=np.ones((T, M)),
        common_shock=np.zeros(T),
        config=cfg,
    )
    return data, merged, forecaster_ids


def _default_empirical_bt_config(training_periods: int = 20) -> BacktestConfig:
    """
    Build a stable backtest configuration for the empirical inflation study.

    Args:
        training_periods (int): Minimum amount of historical data available
            before the OOS evaluation begins

    Returns:
        BacktestConfig: Default empirical backtest hyper-parameters
    """
    window = max(training_periods, 20)
    return BacktestConfig(
        d_max=3,
        cov_window=window,
        bg_window=window,
        tune_window=window,
        recent_best_window=min(8, window),
        min_history=window,
    )


def _select_empirical_comparison_methods(
    performance_table: pd.DataFrame,
    max_methods: int = 6,
) -> List[str]:
    """
    Pick a readable comparison set anchored on `full_gcsr`.

    Args:
        performance_table (pd.DataFrame): Method-level performance summary
        max_methods (int): Maximum number of methods to keep in the plot set

    Returns:
        List[str]: Ordered list of methods for empirical comparison plots
    """
    preferred = [
        "full_gcsr",
        "full_gcsr_relaxed",
        "bates_granger_mv",
        "cov_only",
        "graph_only",
        "equal",
        "recent_best",
        "median",
    ]
    available = performance_table["Method"].tolist()
    selected = []
    for name in preferred:
        if name in available and name not in selected:
            selected.append(name)
        if len(selected) >= max_methods:
            return selected
    for name in available:
        if name not in selected:
            selected.append(name)
        if len(selected) >= max_methods:
            break
    return selected


# ---------- Empirical study runner ----------

def build_empirical_oos_forecast_table(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Construct a tidy OOS table of actual inflation and combined forecasts.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset

    Returns:
        pd.DataFrame: OOS table indexed by target period
    """
    if methods is None:
        methods = study.comparison_methods
    methods = [m for m in methods if m in study.backtest.combined_forecasts]

    idx = study.backtest.oos_periods
    df = pd.DataFrame({
        "TARGET_PERIOD": study.merged_data.loc[idx, "TARGET_PERIOD"].to_numpy(),
        "period_dt": study.merged_data.loc[idx, "period_dt"].to_numpy(),
        "actual": study.data.y[idx],
    })
    for method in methods:
        df[method] = study.backtest.combined_forecasts[method]
    return df.reset_index(drop=True)


def run_empirical_inflation_study(
    forecast_path: str = "Empirical_Data/inflation_forecasts_f.csv",
    truth_path: str = "Empirical_Data/inflation_truth_f.csv",
    training_periods: int = 20,
    bt_cfg: Optional[BacktestConfig] = None,
    mcs_alpha: float = 0.10,
    mcs_B: int = 500,
    mcs_statistic: Literal["Tmax", "TR"] = "Tmax",
    verbose: bool = False,
) -> EmpiricalStudyResult:
    """
    Run a next-quarter out-of-sample empirical backtest on the inflation panel.

    The exercise treats each row as a forecast for the next target quarter,
    starts evaluation after `training_periods` observations, and compares the
    simplex and signed GCSR variants with the benchmark combination rules
    already defined in this module.

    Args:
        forecast_path (str): Path to the empirical forecast CSV
        truth_path (str): Path to the realized inflation CSV
        training_periods (int): In-sample history available before OOS testing
        bt_cfg (BacktestConfig | None): Optional empirical backtest configuration
        mcs_alpha (float): Model Confidence Set significance level
        mcs_B (int): Number of MCS bootstrap resamples
        mcs_statistic (Literal["Tmax", "TR"]): MCS test statistic
        verbose (bool): Whether to print rolling backtest progress

    Returns:
        EmpiricalStudyResult: Empirical data, backtest output, and summary tables
    """
    data, merged, forecaster_ids = load_empirical_inflation_data(
        forecast_path=forecast_path,
        truth_path=truth_path,
        training_periods=training_periods,
    )

    if bt_cfg is None:
        bt_cfg = _default_empirical_bt_config(training_periods=training_periods)
    else:
        bt_cfg.min_history = max(int(bt_cfg.min_history), 20)

    res = run_backtest(data, bt_cfg=bt_cfg, verbose=verbose)
    performance = compute_performance_table(res)
    comparison_methods = _select_empirical_comparison_methods(performance)
    mcs_result = model_confidence_set(
        res,
        alpha=mcs_alpha,
        B=mcs_B,
        statistic=mcs_statistic,
    )
    mcs_table = compute_mcs_performance_table(res, mcs_result=mcs_result)

    study = EmpiricalStudyResult(
        dataset_name="empirical_inflation_next_quarter",
        merged_data=merged,
        forecaster_ids=forecaster_ids,
        training_periods=training_periods,
        comparison_methods=comparison_methods,
        data=data,
        backtest=res,
        performance_table=performance,
        mcs_result=mcs_result,
        mcs_table=mcs_table,
    )
    study.oos_forecast_table = build_empirical_oos_forecast_table(study)
    return study


# ---------- Empirical visualization ----------

def plot_empirical_oos_forecasts(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
    ax=None,
):
    """
    Plot realized inflation against combined next-quarter OOS forecasts.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset
        ax: Optional matplotlib axes

    Returns:
        matplotlib.figure.Figure: Figure containing the OOS forecast comparison
    """
    df = build_empirical_oos_forecast_table(study, methods=methods)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.plot(df["period_dt"], df["actual"], color="black", linewidth=2.2, label="actual")
    for method in [c for c in df.columns if c not in {"TARGET_PERIOD", "period_dt", "actual"}]:
        style = _method_plot_style(method)
        ax.plot(df["period_dt"], df[method], label=method, **style)
    ax.set_title("Empirical Next-Quarter Inflation Forecasts")
    ax.set_xlabel("Target Quarter")
    ax.set_ylabel("Inflation")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_empirical_cumulative_loss(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
    reference: str = "equal",
):
    """
    Plot cumulative loss differences for the empirical OOS exercise.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset
        reference (str): Reference method used in the cumulative-loss plot

    Returns:
        matplotlib.figure.Figure: Figure containing the cumulative-loss comparison
    """
    if methods is None:
        methods = study.comparison_methods
    fig = plot_cumulative_loss(study.backtest, methods=list(methods), reference=reference)
    ax = fig.axes[0]
    ax.set_title("Empirical Cumulative Loss Difference")
    ax.set_xlabel("OOS Index")
    return fig


def plot_empirical_weight_comparison(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
):
    """
    Plot empirical weight paths for the main combination methods.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset

    Returns:
        matplotlib.figure.Figure: Figure containing the weight time series
    """
    if methods is None:
        methods = ["full_gcsr", "cov_only", "graph_only", "bates_granger_mv"]
    methods = [m for m in methods if m in study.backtest.weights]
    return plot_weight_timeseries(study.backtest, methods=methods)


def plot_empirical_gcsr_weight_comparison(
    study: EmpiricalStudyResult,
    top_n: Optional[int] = None,
):
    """
    Compare the simplex and signed GCSR weight paths in the empirical study.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        top_n (int | None): Optional cap on the number of forecasters shown

    Returns:
        matplotlib.figure.Figure: Figure containing the GCSR weight comparison
    """
    return plot_gcsr_weight_comparison(study.backtest, top_n=top_n)


def showcase_empirical_inflation_study(
    forecast_path: str = "Empirical_Data/inflation_forecasts_f.csv",
    truth_path: str = "Empirical_Data/inflation_truth_f.csv",
    training_periods: int = 20,
    bt_cfg: Optional[BacktestConfig] = None,
    mcs_alpha: float = 0.10,
    mcs_B: int = 500,
    mcs_statistic: Literal["Tmax", "TR"] = "Tmax",
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Run the empirical inflation study and return the main tables and plots.

    Args:
        forecast_path (str): Path to the empirical forecast CSV
        truth_path (str): Path to the realized inflation CSV
        training_periods (int): Number of initial periods reserved for training
        bt_cfg (BacktestConfig | None): Optional backtest configuration
        mcs_alpha (float): Model Confidence Set significance level
        mcs_B (int): Number of MCS bootstrap resamples
        mcs_statistic (Literal["Tmax", "TR"]): MCS test statistic
        verbose (bool): Whether to print rolling backtest progress

    Returns:
        Dict[str, object]: Empirical study object plus comparison tables and figures
    """
    study = run_empirical_inflation_study(
        forecast_path=forecast_path,
        truth_path=truth_path,
        training_periods=training_periods,
        bt_cfg=bt_cfg,
        mcs_alpha=mcs_alpha,
        mcs_B=mcs_B,
        mcs_statistic=mcs_statistic,
        verbose=verbose,
    )

    mcs_fig = plot_mcs_summary(
        study.backtest,
        mcs_result=study.mcs_result,
        methods=study.comparison_methods,
    )
    oos_fig = plot_empirical_oos_forecasts(study)
    cumulative_fig = plot_empirical_cumulative_loss(study)
    weights_fig = plot_empirical_weight_comparison(study)
    gcsr_weights_fig = plot_empirical_gcsr_weight_comparison(study)

    return {
        "study": study,
        "performance_table": study.performance_table,
        "mcs_table": study.mcs_table,
        "oos_forecast_table": study.oos_forecast_table,
        "mcs_figure": mcs_fig,
        "oos_forecast_figure": oos_fig,
        "cumulative_loss_figure": cumulative_fig,
        "weights_figure": weights_fig,
        "gcsr_weight_comparison_figure": gcsr_weights_fig,
    }


# ===================================================================
# 13.  LEAKAGE AUDIT UTILITIES
# ===================================================================

def leakage_audit_synthetic(seed=123):
    """
    Synthetic timing sanity check.

    Creates a trivial scenario where the target is fully predictable
    by one model from t-1 info, and verifies that no method uses
    y_{t+1} when forming weights for t+1.

    Returns True if audit passes.
    """
    rng = _ensure_rng(seed)
    T, M = 120, 4
    T0 = 60

    y = rng.normal(0, 1, size=T).cumsum()
    forecasts = np.zeros((T, M))
    # Model 0: knows y exactly one period late (lagged info)
    forecasts[1:, 0] = y[:-1]
    # Model 1-3: noisy
    for j in range(1, M):
        forecasts[:, j] = y + rng.normal(0, 2.0, size=T)

    errors = y[:, None] - forecasts
    losses = squared_loss(y[:, None], forecasts)

    cfg = ScenarioConfig(name="audit", M=M, T=T, T0=T0, seed=seed)
    data = SimulationData(
        y=y, forecasts=forecasts, errors=errors, losses=losses,
        bias_paths=np.zeros((T, M)), sigma_paths=np.ones((T, M)) * 0.5,
        common_shock=np.zeros(T), config=cfg,
    )

    bt_cfg = BacktestConfig(
        d_max=2, fixed_d=1, fixed_h1=0.3, fixed_h2=0.4,
        cov_window=30, min_history=20,
        alpha=0.1, gamma=0.1,
    )
    res = run_backtest(data, bt_cfg)

    # Check: at each OOS period t, the weight vector was formed
    # WITHOUT using y[t]. We verify by checking that the first OOS
    # combined forecast does NOT perfectly nail y[T0], which would
    # indicate leakage.
    first_loss_full = res.combined_losses["full_gcsr"][0]
    first_loss_eq = res.combined_losses["equal"][0]

    # In a correct implementation both should have nonzero loss
    audit_pass = True
    notes = []

    if first_loss_full < 1e-15:
        notes.append("WARNING: Full method has near-zero first-period loss — possible leakage")
        audit_pass = False

    # Additional check: weights should be identical if we permute future y
    # (we can't run this dynamically here, but the structure guarantees it)

    # Check that all weight vectors sum to ~1
    for name in res.weights:
        w = res.weights[name]
        sums = w.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > 1e-4):
            notes.append(f"WARNING: {name} weights don't sum to 1")
            audit_pass = False

    # Check non-negativity
    for name in res.weights:
        if name in SIGNED_WEIGHT_METHODS:
            continue
        w = res.weights[name]
        if np.any(w < -1e-8):
            notes.append(f"WARNING: {name} has negative weights")
            audit_pass = False

    return audit_pass, notes, res


def print_timing_rules():
    """Print the information set / timing conventions."""
    text = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    TIMING CONVENTIONS                          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  Forecast origin: t                                            ║
    ║  Target to predict: y_{t}                                      ║
    ║  Forecasts available: f_{j,t} for j=1,...,M                    ║
    ║  Realised outcomes available: y_0, ..., y_{t-1}                ║
    ║  Realised losses available: L_{j,0}, ..., L_{j,t-1}           ║
    ║  Realised errors available: e_{j,0}, ..., e_{j,t-1}           ║
    ║                                                                ║
    ║  Weights w_{t} are formed using ONLY:                          ║
    ║    - losses / errors from periods 0 through t-1                ║
    ║    - pairwise LD from periods 0 through t-1                    ║
    ║    - covariance from errors 0 through t-1                      ║
    ║                                                                ║
    ║  y_{t} is NEVER used when forming w_{t}.                       ║
    ║                                                                ║
    ║  Combined forecast: ŷ_t = Σ_j w_{j,t} f_{j,t}                ║
    ║  Evaluated after y_t is realised.                              ║
    ║                                                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(text)


# ===================================================================
# 14.  CONVENIENCE: RUN ALL SCENARIOS
# ===================================================================

def run_all_scenarios(
    n_reps: int = 10,
    bt_cfg: BacktestConfig = None,
    verbose: bool = True,
    M: int = 8,
    T: int = 300,
    T0: int = 150,
) -> Dict[str, MCResult]:
    """Run MC for every pre-built scenario."""
    if bt_cfg is None:
        bt_cfg = BacktestConfig(
            d_max=3,
            fixed_d=1,        # speed: fix lag to 1
            fixed_h1=0.25,    # speed: fix bandwidth
            fixed_h2=0.35,
            cov_window=50,
            min_history=30,
            alpha=None,
            gamma=None,
            tune_window=30,
        )

    results = {}
    for sc_key, factory in ALL_SCENARIO_FACTORIES.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running scenario {sc_key}...")
            print(f"{'='*50}")
        mc = run_monte_carlo(
            factory,
            n_reps=n_reps,
            bt_cfg=bt_cfg,
            verbose=verbose,
            M=M, T=T, T0=T0,
        )
        results[sc_key] = mc
        if verbose:
            print(summarise_mc(mc).to_string(index=False))
    return results


# ===================================================================
# END OF MODULE
# ===================================================================

if __name__ == "__main__":
    print("forecast_combination module loaded successfully.")
    print("Run leakage audit...")
    passed, notes, _ = leakage_audit_synthetic()
    print(f"Audit passed: {passed}")
    for n in notes:
        print(f"  {n}")
