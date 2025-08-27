#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit.py — Joint constrained curve fitting per i, multi-group parameters per g.

Key features
- SymPy-parse a user function f(i, x, θ) with free parameter symbols θ (no g in the formula).
- Per fixed i: jointly estimate θ_g for each group g by minimizing
      L = L_fit (Huber) + lambda_s * L_smooth + lambda_sim * L_sim
  subject to hard constraints on:
    (1) Monotonicity in x for each g: s_i * f'(x) >= eps_mono
    (2) Non-crossing/tangency across g: f_{g+1}(x) - f_g(x) >= eps_ord
    (3) Denominator safety: den(x)^2 - eps_den^2 >= 0  (if function has a denominator)
- Robust derivative handling:
    * Lambdify each ∂/∂p separately (list of callables) to avoid inhomogeneous array shapes.
    * Broadcast scalar derivatives (e.g., parameter not appearing) to vectors of length N.
- SLSQP fallback uses DENSE Jacobian to avoid shape mismatch inside SciPy.
- Numerical evals wrapped in numpy.errstate to silence transient divide/invalid warnings.

CLI example
    python fit.py --data data.csv --func function.txt --outdir out \
      --grid-points 200 --auto-direction true --lambda-s 1e-4 --lambda-sim 1e-3

Exit code: 0 on success; 2 on hard-constraint violation or solver failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sympy as sp
from sympy import symbols
from sympy.utilities.lambdify import lambdify

from scipy.optimize import minimize, Bounds, NonlinearConstraint, least_squares
from scipy import sparse

# ----------------------------- Utilities -----------------------------

def trapz_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal integration weights for 1D grid x (possibly non-uniform)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 1:
        return np.array([1.0])
    w = np.zeros(n, dtype=float)
    dx = np.diff(x)
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    if n > 2:
        w[1:-1] = (x[2:] - x[:-2]) / 2.0
    return w


def huber_value_and_grad(r: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Huber loss ρ and derivative dρ/dr for residual r."""
    r = np.asarray(r, dtype=float)
    absr = np.abs(r)
    small = absr <= delta
    val = np.empty_like(r)
    grad = np.empty_like(r)
    val[small] = 0.5 * r[small] * r[small]
    grad[small] = r[small]
    val[~small] = delta * (absr[~small] - 0.5 * delta)
    grad[~small] = delta * np.sign(r[~small])
    return val, grad


def unique_sorted(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.unique(np.sort(arr))


# ----------------------------- SymPy Model -----------------------------

@dataclass
class SympyModel:
    expr_str: str
    expr: sp.Expr
    x_sym: sp.Symbol
    i_sym: sp.Symbol
    param_syms: List[sp.Symbol]
    param_names: List[str]
    dfdx_expr: sp.Expr
    d2fdx2_expr: sp.Expr
    denom_expr: sp.Expr
    f_lam: callable
    dfdx_lam: callable
    d2fdx2_lam: callable
    df_dp_lams: List[callable]
    ddfdx_dp_lams: List[callable]
    d2fdx2_dp_lams: List[callable]
    dden_dp_lams: Optional[List[callable]]
    denom_lam: Optional[callable]

    @staticmethod
    def parse(expr_str: str) -> "SympyModel":
        i_sym, x_sym = symbols("i x", real=True)
        expr = sp.sympify(expr_str, locals={"i": i_sym, "x": x_sym})

        # Parameters: free symbols excluding i,x
        param_syms = sorted([s for s in expr.free_symbols if s not in {i_sym, x_sym}], key=lambda s: s.name)
        param_names = [s.name for s in param_syms]

        # Derivatives in x
        dfdx_expr = sp.diff(expr, x_sym)
        d2fdx2_expr = sp.diff(expr, x_sym, 2)

        # Denominator (if any)
        num, den = sp.fraction(sp.together(expr))
        denom_expr = sp.simplify(den)

        # Per-parameter derivative expressions (avoid Matrix lambdify)
        df_dp_exprs = [sp.diff(expr, p) for p in param_syms] if param_syms else []
        ddfdx_dp_exprs = [sp.diff(dfdx_expr, p) for p in param_syms] if param_syms else []
        d2fdx2_dp_exprs = [sp.diff(d2fdx2_expr, p) for p in param_syms] if param_syms else []
        if denom_expr == 1:
            dden_dp_exprs = None
        else:
            dden_dp_exprs = [sp.diff(denom_expr, p) for p in param_syms] if param_syms else []

        args = (x_sym, i_sym, *param_syms)

        f_lam = lambdify(args, expr, "numpy")
        dfdx_lam = lambdify(args, dfdx_expr, "numpy")
        d2fdx2_lam = lambdify(args, d2fdx2_expr, "numpy")
        df_dp_lams = [lambdify(args, e, "numpy") for e in df_dp_exprs]
        ddfdx_dp_lams = [lambdify(args, e, "numpy") for e in ddfdx_dp_exprs]
        d2fdx2_dp_lams = [lambdify(args, e, "numpy") for e in d2fdx2_dp_exprs]
        dden_dp_lams = [lambdify(args, e, "numpy") for e in dden_dp_exprs] if dden_dp_exprs is not None else None
        denom_lam = lambdify(args, denom_expr, "numpy") if dden_dp_exprs is not None else None

        return SympyModel(
            expr_str=expr_str,
            expr=expr,
            x_sym=x_sym,
            i_sym=i_sym,
            param_syms=param_syms,
            param_names=param_names,
            dfdx_expr=dfdx_expr,
            d2fdx2_expr=d2fdx2_expr,
            denom_expr=denom_expr,
            f_lam=f_lam,
            dfdx_lam=dfdx_lam,
            d2fdx2_lam=d2fdx2_lam,
            df_dp_lams=df_dp_lams,
            ddfdx_dp_lams=ddfdx_dp_lams,
            d2fdx2_dp_lams=d2fdx2_dp_lams,
            dden_dp_lams=dden_dp_lams,
            denom_lam=denom_lam
        )

    def _eval_param_deriv_list(self, lams: Optional[List[callable]], x: np.ndarray, i_val: float, theta: np.ndarray) -> Optional[np.ndarray]:
        if lams is None:
            return None
        P = len(self.param_syms)
        x = np.asarray(x)
        x_len = int(np.size(x))
        if P == 0:
            return np.zeros((0, x_len), dtype=float)
        rows = []
        for lam in lams:
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                r = lam(x, i_val, *theta)
            r_arr = np.asarray(r)
            if r_arr.ndim == 0:
                r_arr = np.full((x_len,), float(r_arr), dtype=float)
            else:
                r_arr = r_arr.astype(float, copy=False).reshape(-1)
                if r_arr.size == 1:
                    r_arr = np.full((x_len,), float(r_arr[0]), dtype=float)
                elif r_arr.size != x_len:
                    r_arr = np.broadcast_to(r_arr, (x_len,)).astype(float, copy=False)
            rows.append(r_arr)
        return np.stack(rows, axis=0).astype(float, copy=False)

    # --- vectorized wrappers with np.errstate ---
    def f(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            return np.asarray(self.f_lam(x, i_val, *theta), dtype=float)

    def dfdx(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            return np.asarray(self.dfdx_lam(x, i_val, *theta), dtype=float)

    def d2fdx2(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            return np.asarray(self.d2fdx2_lam(x, i_val, *theta), dtype=float)

    def df_dp(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return self._eval_param_deriv_list(self.df_dp_lams, x, i_val, theta)

    def ddfdx_dp(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return self._eval_param_deriv_list(self.ddfdx_dp_lams, x, i_val, theta)

    def d2fdx2_dp(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return self._eval_param_deriv_list(self.d2fdx2_dp_lams, x, i_val, theta)

    def denom(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> Optional[np.ndarray]:
        if self.denom_lam is None:
            return None
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            return np.asarray(self.denom_lam(x, i_val, *theta), dtype=float)

    def dden_dp(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> Optional[np.ndarray]:
        return self._eval_param_deriv_list(self.dden_dp_lams, x, i_val, theta)


# ----------------------------- Fitting core per i -----------------------------

@dataclass
class SolverConfig:
    eps_mono: float = 1e-6
    eps_ord: float = 1e-6
    eps_den: float = 1e-8
    lambda_s: float = 1e-4
    lambda_sim: float = 1e-3
    huber_delta: float = 1.0
    lower_bound: float = -1e6
    upper_bound: float = 1e6
    maxiter: int = 2000
    verbose: bool = False
    method_primary: str = "trust-constr"
    method_fallback: Optional[str] = "SLSQP"
    order_mode: str = "auto"  # 'auto' | 'g-asc' | 'g-desc'
    auto_direction: bool = True
    s_i_override: Optional[int] = None  # +1 or -1
    seed: Optional[int] = 42


class FitSingleI:
    def __init__(
        self,
        df_i: pd.DataFrame,
        i_value: float,
        model: SympyModel,
        grid_points: int,
        config: SolverConfig,
        outdir: str,
    ):
        self.df_i = df_i.copy()
        self.i_value = float(i_value)
        self.model = model
        self.grid_points = int(grid_points)
        self.config = config
        self.outdir = outdir

        for col in ["g", "x"]:
            if col not in self.df_i.columns:
                raise ValueError(f"Missing required column '{col}' in data for i={i_value}.")
        if "y" not in self.df_i.columns:
            raise ValueError("Missing required column 'y'.")
        self.df_i["w"] = 1.0 if "w" not in self.df_i.columns else self.df_i["w"].fillna(1.0)

        self.groups = unique_sorted(self.df_i["g"].to_numpy())
        self.n_g = len(self.groups)
        self.P = len(self.model.param_syms)
        if self.P == 0:
            raise ValueError("No free parameters found in the function; nothing to estimate.")

        x_all = unique_sorted(self.df_i["x"].to_numpy())
        if x_all.size == 0:
            raise ValueError(f"No x values found for i={i_value}.")
        self.x_min, self.x_max = float(np.min(x_all)), float(np.max(x_all))
        self.x_dense = self._build_dense_grid(x_all, self.grid_points)
        self.x_const = unique_sorted(np.concatenate([x_all, self.x_dense]))

        self.obs_by_g: Dict[float, Dict[str, np.ndarray]] = {}
        for g in self.groups:
            sub = self.df_i[(self.df_i["g"] == g) & (~self.df_i["y"].isna())]
            self.obs_by_g[float(g)] = {
                "x": sub["x"].to_numpy(dtype=float),
                "y": sub["y"].to_numpy(dtype=float),
                "w": sub["w"].to_numpy(dtype=float),
            }

        self.theta0 = self._build_initial_thetas()
        self.s_i = self._infer_monotonic_direction()
        self.g_order = self._infer_group_order()

        lb = np.full(self.P * self.n_g, self.config.lower_bound, dtype=float)
        ub = np.full(self.P * self.n_g, self.config.upper_bound, dtype=float)
        self.bounds = Bounds(lb, ub)

        self.w_smooth = trapz_weights(self.x_dense)

    @staticmethod
    def _make_linspace_inclusive(a: float, b: float, step: float) -> np.ndarray:
        if a == b:
            return np.array([a], dtype=float)
        n = max(2, int(math.floor((b - a) / step)) + 1)
        xs = a + np.arange(n) * step
        if xs[-1] < b - 1e-12:
            xs = np.append(xs, b)
        else:
            xs[-1] = b
        return xs

    def _build_dense_grid(self, x_all: np.ndarray, grid_points: int) -> np.ndarray:
        x_all = unique_sorted(x_all)
        a, b = float(np.min(x_all)), float(np.max(x_all))
        if a == b:
            return np.array([a], dtype=float)

        step_eq = (b - a) / max(grid_points - 1, 1)
        if x_all.size >= 2:
            min_gap = np.min(np.diff(x_all))
            step_obs = min_gap / 5.0 if min_gap > 0 else step_eq
        else:
            step_obs = step_eq
        step = min(step_eq, step_obs)
        return self._make_linspace_inclusive(a, b, step)

    def _build_initial_thetas(self) -> Dict[float, np.ndarray]:
        rng = np.random.default_rng(self.config.seed if self.config.seed is not None else 123)
        thetas: Dict[float, np.ndarray] = {}
        for g in self.groups:
            xg = self.obs_by_g[float(g)]["x"]
            yg = self.obs_by_g[float(g)]["y"]
            if xg.size >= max(2, self.P):
                def resid(theta):
                    return self.model.f(xg, self.i_value, theta) - yg
                def jac(theta):
                    return self.model.df_dp(xg, self.i_value, theta).T  # (N,P)
                try:
                    res = least_squares(
                        resid,
                        x0=np.zeros(self.P, dtype=float),
                        jac=jac,
                        bounds=(
                            np.full(self.P, self.config.lower_bound),
                            np.full(self.P, self.config.upper_bound),
                        ),
                        loss="huber",
                        f_scale=1.0,
                        max_nfev=800,
                        verbose=0,
                    )
                    theta_g = np.clip(res.x, self.config.lower_bound, self.config.upper_bound)
                except Exception:
                    theta_g = rng.normal(scale=1e-2, size=self.P)
            else:
                theta_g = rng.normal(scale=1e-2, size=self.P)
            thetas[float(g)] = theta_g.astype(float)
        return thetas

    def _infer_monotonic_direction(self) -> int:
        if not self.config.auto_direction and self.config.s_i_override in (+1, -1):
            return int(self.config.s_i_override)
        obs = self.df_i[~self.df_i["y"].isna()]
        if obs.shape[0] >= 3:
            try:
                def rankdata(a):
                    a = np.asarray(a, dtype=float)
                    order = np.argsort(a, kind="mergesort")
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(1, a.size + 1, dtype=float)
                    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
                    cum = np.cumsum(counts)
                    start = cum - counts + 1
                    avg = (start + cum) / 2.0
                    return avg[inv]
                x_r = rankdata(obs["x"].to_numpy())
                y_r = rankdata(obs["y"].to_numpy())
                x_c = x_r - x_r.mean()
                y_c = y_r - y_r.mean()
                denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))
                rho = float(np.sum(x_c * y_c) / denom) if denom > 0 else 0.0
                return +1 if rho >= 0 else -1
            except Exception:
                pass
        signs = []
        for g in self.groups:
            d1 = self.model.dfdx(self.x_dense, self.i_value, self.theta0[float(g)])
            med = np.nanmedian(d1)
            if np.isfinite(med) and med != 0.0:
                signs.append(np.sign(med))
        if len(signs) == 0:
            return +1
        return +1 if int(np.sign(np.sum(signs))) >= 0 else -1

    def _infer_group_order(self) -> List[float]:
        mode = self.config.order_mode
        if mode in ("g-asc", "g-desc"):
            ordered = sorted(self.groups)
            return ordered if mode == "g-asc" else ordered[::-1]
        med_vals = []
        for g in self.groups:
            fx = self.model.f(self.x_dense, self.i_value, self.theta0[float(g)])
            med_vals.append((float(g), float(np.nanmedian(fx))))
        return [g for g, _ in sorted(med_vals, key=lambda t: t[1])]

    def _theta_flat_to_dict(self, theta_flat: np.ndarray) -> Dict[float, np.ndarray]:
        out: Dict[float, np.ndarray] = {}
        for idx, g in enumerate(self.groups):
            out[float(g)] = theta_flat[idx*self.P:(idx+1)*self.P]
        return out

    def _theta_dict_to_flat(self, thetas: Dict[float, np.ndarray]) -> np.ndarray:
        return np.concatenate([thetas[float(g)] for g in self.groups], axis=0)

    def _objective_and_grad(self, theta_flat: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, float]]:
        cfg = self.config
        thetas = self._theta_flat_to_dict(theta_flat)

        L_fit = 0.0
        g_fit = np.zeros_like(theta_flat)

        # Huber fit
        for idx_g, g in enumerate(self.groups):
            th = thetas[float(g)]
            obs = self.obs_by_g[float(g)]
            xg, yg, wg = obs["x"], obs["y"], obs["w"]
            if xg.size > 0:
                f = self.model.f(xg, self.i_value, th)
                r = f - yg
                v, dr = huber_value_and_grad(r, cfg.huber_delta)
                L_fit += float(np.sum(wg * v))
                J = self.model.df_dp(xg, self.i_value, th)  # (P,N)
                g_fit[idx_g*self.P:(idx_g+1)*self.P] = J @ (wg * dr)

        # Smoothness
        L_smooth = 0.0
        g_smooth = np.zeros_like(theta_flat)
        w = self.w_smooth
        for idx_g, g in enumerate(self.groups):
            th = thetas[float(g)]
            d2 = self.model.d2fdx2(self.x_dense, self.i_value, th)  # (M,)
            L_smooth += float(np.sum(w * (d2**2)))
            d2_dp = self.model.d2fdx2_dp(self.x_dense, self.i_value, th)  # (P,M)
            g_smooth[idx_g*self.P:(idx_g+1)*self.P] = d2_dp @ (w * (2.0*d2))

        # Similarity
        Theta = np.stack([thetas[float(g)] for g in self.groups], axis=0)  # (G,P)
        theta_bar = np.mean(Theta, axis=0)
        diffs = Theta - theta_bar
        L_sim = float(np.sum(diffs**2))
        g_sim = np.zeros_like(theta_flat)
        for idx_g in range(self.n_g):
            g_sim[idx_g*self.P:(idx_g+1)*self.P] = 2.0 * diffs[idx_g]

        L = L_fit + cfg.lambda_s * L_smooth + cfg.lambda_sim * L_sim
        grad = g_fit + cfg.lambda_s * g_smooth + cfg.lambda_sim * g_sim

        diag = {"L_fit": L_fit, "L_smooth": L_smooth, "L_sim": L_sim}
        return L, grad, diag

    # --- Constraints ---
    def _constraints_mono(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        s = float(self.s_i)
        eps = float(self.config.eps_mono)
        xs = self.x_const
        G, P, M = self.n_g, self.P, xs.size

        vals = []
        rows = []
        cols = []
        data = []
        for idx_g, g in enumerate(self.groups):
            th = theta_flat[idx_g*P:(idx_g+1)*P]
            d1 = self.model.dfdx(xs, self.i_value, th)  # (M,)
            cvals = s * d1 - eps
            vals.append(cvals)
            d1_dp = self.model.ddfdx_dp(xs, self.i_value, th)  # (P,M)
            for k in range(M):
                row = idx_g*M + k
                rows.extend([row]*P)
                cols.extend(range(idx_g*P, (idx_g+1)*P))
                data.extend(list(s * d1_dp[:, k]))
        vals = np.concatenate(vals, axis=0) if len(vals) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(G*M, G*P))
        return vals, J

    def _constraints_order(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        eps = float(self.config.eps_ord)
        xs = self.x_const
        P, M = self.P, xs.size
        pairs = list(zip(self.g_order[:-1], self.g_order[1:]))
        n_pairs = len(pairs)

        vals = []
        rows = []
        cols = []
        data = []
        g_to_idx = {float(g): idx for idx, g in enumerate(self.groups)}

        for pidx, (g_c, g_n) in enumerate(pairs):
            ic = g_to_idx[float(g_c)]
            inn = g_to_idx[float(g_n)]
            th_c = theta_flat[ic*P:(ic+1)*P]
            th_n = theta_flat[inn*P:(inn+1)*P]
            f_c = self.model.f(xs, self.i_value, th_c)
            f_n = self.model.f(xs, self.i_value, th_n)
            cvals = (f_n - f_c) - eps
            vals.append(cvals)
            df_dp_c = self.model.df_dp(xs, self.i_value, th_c)  # (P,M)
            df_dp_n = self.model.df_dp(xs, self.i_value, th_n)  # (P,M)
            for k in range(M):
                row = pidx*M + k
                rows.extend([row]*P); cols.extend(range(inn*P, (inn+1)*P)); data.extend(list(df_dp_n[:, k]))
                rows.extend([row]*P); cols.extend(range(ic*P, (ic+1)*P)); data.extend(list(-df_dp_c[:, k]))
        vals = np.concatenate(vals, axis=0) if len(vals) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(n_pairs*M, self.n_g*self.P))
        return vals, J

    def _constraints_den(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        if self.model.denom_lam is None or self.model.dden_dp_lams is None:
            return np.array([], dtype=float), sparse.csr_matrix((0, self.n_g*self.P))
        xs = self.x_const
        P, M = self.P, xs.size
        eps2 = float(self.config.eps_den)**2

        vals = []
        rows = []
        cols = []
        data = []
        for idx_g, g in enumerate(self.groups):
            th = theta_flat[idx_g*P:(idx_g+1)*P]
            den = self.model.denom(xs, self.i_value, th)  # (M,)
            cvals = (den**2) - eps2
            vals.append(cvals)
            dden_dp = self.model.dden_dp(xs, self.i_value, th)  # (P,M)
            for k in range(M):
                row = idx_g*M + k
                rows.extend([row]*P)
                cols.extend(range(idx_g*P, (idx_g+1)*P))
                data.extend(list(2.0 * den[k] * dden_dp[:, k]))
        vals = np.concatenate(vals, axis=0) if len(vals) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_g*M, self.n_g*self.P))
        return vals, J

    # SciPy hooks
    def objective(self, theta_flat: np.ndarray) -> float:
        L, _, _ = self._objective_and_grad(theta_flat)
        return float(L)

    def objective_grad(self, theta_flat: np.ndarray) -> np.ndarray:
        _, g, _ = self._objective_and_grad(theta_flat)
        return g

    def constraints_all(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        v1, J1 = self._constraints_mono(theta_flat)
        v2, J2 = self._constraints_order(theta_flat)
        v3, J3 = self._constraints_den(theta_flat)
        vals = np.concatenate([v1, v2, v3], axis=0)
        J = sparse.vstack([J1, J2, J3], format="csr")
        return vals, J

    def cons_fun(self, theta_flat: np.ndarray) -> np.ndarray:
        vals, _ = self.constraints_all(theta_flat)
        return vals

    def cons_jac(self, theta_flat: np.ndarray):
        _, J = self.constraints_all(theta_flat)
        return J

    def solve(self) -> Tuple[np.ndarray, Dict]:
        theta0_flat = self._theta_dict_to_flat(self.theta0)
        cons = NonlinearConstraint(self.cons_fun, lb=0.0, ub=np.inf, jac=self.cons_jac)

        # primary
        res = minimize(
            fun=self.objective,
            x0=theta0_flat,
            jac=self.objective_grad,
            bounds=self.bounds,
            constraints=[cons],
            method=self.config.method_primary,
            options={"maxiter": self.config.maxiter, "verbose": 3 if self.config.verbose else 0},
        )

        # fallback
        if (not res.success) and (self.config.method_fallback is not None):
            res2 = minimize(
                fun=self.objective,
                x0=res.x,
                jac=self.objective_grad,
                bounds=self.bounds,
                constraints=[{
                    "type": "ineq",
                    "fun": self.cons_fun,
                    # SLSQP expects dense Jacobian
                    "jac": (lambda x: self.cons_jac(x).toarray() if hasattr(self.cons_jac(x), "toarray") else np.asarray(self.cons_jac(x)))
                }],
                method=self.config.method_fallback,
                options={"maxiter": self.config.maxiter, "ftol": 1e-9, "disp": self.config.verbose},
            )
            if res2.success:
                res = res2

        _, _, diag = self._objective_and_grad(res.x)
        cons_vals, _ = self.constraints_all(res.x)
        info = {
            "success": bool(res.success),
            "status": int(getattr(res, "status", -1)),
            "message": str(getattr(res, "message", "")),
            "niter": int(getattr(res, "nit", -1)),
            "fun": float(getattr(res, "fun", np.nan)),
            "L_fit": float(diag.get("L_fit", np.nan)),
            "L_smooth": float(diag.get("L_smooth", np.nan)),
            "L_sim": float(diag.get("L_sim", np.nan)),
            "max_constraint_violation": float(np.max(np.maximum(0.0, -cons_vals))) if cons_vals.size > 0 else 0.0,
        }
        return res.x, info

    def compute_checks(self, theta_flat: np.ndarray) -> Dict[str, float]:
        thetas = self._theta_flat_to_dict(theta_flat)
        xs = self.x_const

        # mono
        mono_vals = []
        for g in self.groups:
            th = thetas[float(g)]
            d1 = self.model.dfdx(xs, self.i_value, th)
            mono_vals.append(self.config.eps_mono - self.s_i * d1)
        mono_violation = float(np.max(np.concatenate(mono_vals))) if mono_vals else 0.0

        # ordering
        ord_vals = []
        pairs = list(zip(self.g_order[:-1], self.g_order[1:]))
        for (g_c, g_n) in pairs:
            th_c = thetas[float(g_c)]
            th_n = thetas[float(g_n)]
            f_c = self.model.f(xs, self.i_value, th_c)
            f_n = self.model.f(xs, self.i_value, th_n)
            ord_vals.append(self.config.eps_ord - (f_n - f_c))
        ord_violation = float(np.max(np.concatenate(ord_vals))) if ord_vals else 0.0

        # denom
        denom_min_abs = math.inf
        if self.model.denom_lam is not None:
            for g in self.groups:
                th = thetas[float(g)]
                den = self.model.denom(xs, self.i_value, th)
                denom_min_abs = min(denom_min_abs, float(np.min(np.abs(den))))
        else:
            denom_min_abs = float("inf")

        # smooth penalty (unweighted)
        w = self.w_smooth
        L_smooth = 0.0
        for g in self.groups:
            th = thetas[float(g)]
            d2 = self.model.d2fdx2(self.x_dense, self.i_value, th)
            L_smooth += float(np.sum(w * (d2**2)))

        return {
            "monotonicity_violation_max": mono_violation,
            "ordering_violation_max": ord_violation,
            "denominator_min_abs": denom_min_abs,
            "smooth_penalty_value": L_smooth,
        }

    def export_plots(self, theta_flat: np.ndarray, fname: str) -> None:
        thetas = self._theta_flat_to_dict(theta_flat)
        plt.figure()
        for g in self.groups:
            th = thetas[float(g)]
            xs = self.x_dense
            ys = self.model.f(xs, self.i_value, th)
            plt.plot(xs, ys, label=f"g={g}")
        for g in self.groups:
            obs = self.obs_by_g[float(g)]
            if obs["x"].size > 0:
                plt.scatter(obs["x"], obs["y"], marker="o", s=16, alpha=0.7)
        plt.title(f"i = {self.i_value} — fitted f(x) by g")
        plt.xlabel("x")
        plt.ylabel("f(x) / y")
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(fname, dpi=180)
        plt.close()

    def export_params(self, theta_flat: np.ndarray) -> pd.DataFrame:
        rows = []
        for idx_g, g in enumerate(self.groups):
            th = theta_flat[idx_g*self.P:(idx_g+1)*self.P]
            for p_name, val in zip(self.model.param_names, th):
                rows.append({"i": self.i_value, "g": float(g), "param": p_name, "value": float(val)})
        return pd.DataFrame(rows)

    def export_grid(self, theta_flat: np.ndarray) -> pd.DataFrame:
        thetas = self._theta_flat_to_dict(theta_flat)
        rows = []
        for g in self.groups:
            th = thetas[float(g)]
            xs = self.x_dense
            ys = self.model.f(xs, self.i_value, th)
            for x_val, y_val in zip(xs, ys):
                rows.append({"i": self.i_value, "g": float(g), "x": float(x_val), "f_x": float(y_val)})
        return pd.DataFrame(rows)

    def export_fitted(self, theta_flat: np.ndarray) -> pd.DataFrame:
        thetas = self._theta_flat_to_dict(theta_flat)
        df = self.df_i.copy()
        y_hat_vals = []
        for _, row in df.iterrows():
            g = float(row["g"]); x = float(row["x"])
            th = thetas[g]
            y_hat_vals.append(float(self.model.f(np.array([x]), self.i_value, th)[0]))
        df["y_hat"] = np.array(y_hat_vals, dtype=float)
        df["source"] = np.where(df["y"].isna(), "imputed", "observed")
        return df[["g", "i", "x", "y", "y_hat", "source"]].copy()


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint constrained curve fitting per i and g (SymPy-defined f(i,x,θ)).")
    p.add_argument("--data", required=True, help="Path to data.csv (columns: g,i,x,y; optional: w).")
    p.add_argument("--func", required=True, help="Path to function.txt (first line: SymPy expression in i and x).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--grid-points", type=int, default=200, help="Baseline equal-spaced grid points per i (default: 200).")
    p.add_argument("--auto-direction", type=str, default="true", help="Auto infer monotonic direction s_i (true/false).")
    p.add_argument("--direction", type=str, choices=["inc", "dec"], default=None, help="Force monotonic direction (inc/dec).")
    p.add_argument("--order-mode", type=str, choices=["auto", "g-asc", "g-desc"], default="auto",
                   help="Cross-g ordering: 'auto' (by median fitted f), or by numeric g ascending/descending.")
    p.add_argument("--lambda-s", type=float, default=1e-4, dest="lambda_s", help="Smoothness penalty weight.")
    p.add_argument("--lambda-sim", type=float, default=1e-3, dest="lambda_sim", help="Cross-g similarity penalty weight.")
    p.add_argument("--eps-mono", type=float, default=1e-6, dest="eps_mono", help="Monotonicity margin epsilon.")
    p.add_argument("--eps-ord", type=float, default=1e-6, dest="eps_ord", help="Ordering margin epsilon.")
    p.add_argument("--eps-den", type=float, default=1e-8, dest="eps_den", help="Denominator lower-bound epsilon.")
    p.add_argument("--lower", type=float, default=-1e6, help="Lower bound for all parameters.")
    p.add_argument("--upper", type=float, default=1e6, help="Upper bound for all parameters.")
    p.add_argument("--maxiter", type=int, default=2000, help="Max iterations for optimizer.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    p.add_argument("--verbose", action="store_true", help="Verbose solver logs.")
    p.add_argument("--method-primary", type=str, default="trust-constr", choices=["trust-constr", "SLSQP"], help="Primary optimizer method.")
    p.add_argument("--method-fallback", type=str, default="SLSQP", choices=["SLSQP", "none"], help="Fallback optimizer (use 'none' to disable).")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    with open(args.func, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("function.txt first line is empty.")
        expr_str = first_line

    model = SympyModel.parse(expr_str)

    na = ["NA", "NaN", ""]
    df = pd.read_csv(args.data, na_values=na)
    for col in ["g", "i", "x"]:
        if col not in df.columns:
            raise ValueError(f"data.csv missing required column '{col}'.")
    if "y" not in df.columns:
        raise ValueError("data.csv missing required column 'y'.")
    df["g"] = pd.to_numeric(df["g"], errors="coerce")
    df["i"] = pd.to_numeric(df["i"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "w" in df.columns:
        df["w"] = pd.to_numeric(df["w"], errors="coerce").fillna(1.0)

    i_values = unique_sorted(df["i"].to_numpy())
    if i_values.size == 0:
        raise ValueError("No i values found in data.")

    cfg = SolverConfig(
        eps_mono=float(args.eps_mono),
        eps_ord=float(args.eps_ord),
        eps_den=float(args.eps_den),
        lambda_s=float(args.lambda_s),
        lambda_sim=float(args.lambda_sim),
        lower_bound=float(args.lower),
        upper_bound=float(args.upper),
        maxiter=int(args.maxiter),
        verbose=bool(args.verbose),
        method_primary=str(args.method_primary),
        method_fallback=(None if str(args.method_fallback).lower()=="none" else str(args.method_fallback)),
        order_mode=args.order_mode,
        auto_direction=(str(args.auto_direction).lower() in ("true", "1", "yes", "y")),
        s_i_override=(+1 if args.direction == "inc" else (-1 if args.direction == "dec" else None)),
        seed=int(args.seed),
    )

    all_params = []
    all_grid = []
    all_fitted = []
    checks = {
        "spec_version": "2025-08-12",
        "function": expr_str,
        "params": model.param_names,
        "per_i": [],
        "overall": {},
        "config": {
            "eps_mono": cfg.eps_mono,
            "eps_ord": cfg.eps_ord,
            "eps_den": cfg.eps_den,
            "lambda_s": cfg.lambda_s,
            "lambda_sim": cfg.lambda_sim,
            "order_mode": cfg.order_mode,
            "auto_direction": cfg.auto_direction,
            "direction_override": cfg.s_i_override,
            "lower": cfg.lower_bound,
            "upper": cfg.upper_bound,
            "maxiter": cfg.maxiter,
            "method_primary": cfg.method_primary,
            "method_fallback": cfg.method_fallback,
        }
    }

    any_failure = False
    for i_val in i_values:
        df_i = df[df["i"] == i_val].copy()
        fitter = FitSingleI(
            df_i=df_i, i_value=float(i_val), model=model,
            grid_points=int(args.grid_points), config=cfg, outdir=outdir
        )
        theta_hat, info = fitter.solve()
        ch = fitter.compute_checks(theta_hat)

        plot_path = os.path.join(outdir, f"plots_i={i_val}.png")
        try:
            fitter.export_plots(theta_hat, plot_path)
        except Exception as e:
            warnings.warn(f"Plotting failed for i={i_val}: {e}")

        all_params.append(fitter.export_params(theta_hat))
        all_grid.append(fitter.export_grid(theta_hat))
        all_fitted.append(fitter.export_fitted(theta_hat))

        checks["per_i"].append({
            "i": float(i_val),
            "monotonicity_violation_max": ch["monotonicity_violation_max"],
            "ordering_violation_max": ch["ordering_violation_max"],
            "denominator_min_abs": ch["denominator_min_abs"],
            "smooth_penalty_value": ch["smooth_penalty_value"],
            "solver": info,
            "g_order": list(map(float, fitter.g_order)),
            "s_i": int(fitter.s_i),
            "x_min": float(fitter.x_min),
            "x_max": float(fitter.x_max),
            "n_groups": int(fitter.n_g),
            "n_params_per_group": int(fitter.P),
        })

        violated = (
            (ch["monotonicity_violation_max"] > 1e-9) or
            (ch["ordering_violation_max"] > 1e-9) or
            (np.isfinite(ch["denominator_min_abs"]) and (ch["denominator_min_abs"] < cfg.eps_den - 1e-12)) or
            (not info.get("success", False))
        )
        if violated:
            any_failure = True

    params_df = pd.concat(all_params, ignore_index=True) if all_params else pd.DataFrame(columns=["i","g","param","value"])
    grid_df = pd.concat(all_grid, ignore_index=True) if all_grid else pd.DataFrame(columns=["i","g","x","f_x"])
    fitted_df = pd.concat(all_fitted, ignore_index=True) if all_fitted else pd.DataFrame(columns=["g","i","x","y","y_hat","source"])

    params_df.to_csv(os.path.join(outdir, "params.csv"), index=False)
    grid_df.to_csv(os.path.join(outdir, "grid.csv"), index=False)
    fitted_df.to_csv(os.path.join(outdir, "fitted.csv"), index=False)
    with open(os.path.join(outdir, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    if any_failure:
        print("Constraint violation or solver failure detected. See checks.json for details.", file=sys.stderr)
        sys.exit(2)
    else:
        print("Fitting completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
