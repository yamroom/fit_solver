#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit.py — Joint constrained curve fitting per i, multi-group parameters per g.

Implements the specification:
- Read `data.csv` (columns: g,i,x,y; optional weight column `w`) and `function.txt` (first line SymPy-parsable expression in i, x, and free parameter symbols).
- For each fixed i, jointly estimate one parameter vector per g by minimizing:
      L = L_fit (Huber) + lambda_s * L_smooth + lambda_sim * L_sim
  with hard inequality constraints on:
      1) Monotonicity in x for each (g, i)
      2) Cross-g non-crossing / non-tangency (global ordering across g curves)
      3) Denominator safety across x-domain (if function has a denominator)
- After fitting, impute missing y with f(x; i, θ_g) and write outputs:
    a) plots_i=<value>.png: curves per g with observed points overlaid
    b) params.csv: long-form table (i,g,param,value)
    c) fitted.csv: long-form table (g,i,x,y,y_hat,source)
    d) grid.csv: long-form table (i,g,x,f_x) on a dense x-grid
    e) checks.json: validation report (max violations, min |den|, smooth penalty, solver info)

Usage example:
    python fit.py \
      --data data.csv \
      --func function.txt \
      --outdir out \
      --grid-points 200 \
      --auto-direction true \
      --lambda-s 1e-4 \
      --lambda-sim 1e-3

Exit codes: 0 success; 2 constraint violation beyond tolerance or solver failure.

Notes:
- Function must include symbols `i` and `x` and **not** include `g`. Per-group differences
  are represented by separate parameter vectors (one set per g).
- Monotonic direction s_i is auto-inferred by default; ordering across g is inferred by
  sorting groups by median fitted values (using an initial per-group prefit), then enforcing
  strict separation between adjacent curves across the entire x-range.
- Denominator safety is enforced via den(x)^2 - eps_den^2 >= 0 at grid points (smooth).
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
    """Return trapezoidal integration weights for possibly non-uniform grid x."""
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
    """
    Huber loss value and derivative wrt residual r.
    ρ_δ(r) = 0.5 r^2, if |r| <= δ; else δ(|r| - 0.5 δ).
    dρ/dr = r, if |r| <= δ; else δ * sign(r).
    """
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
    """Unique + sorted for 1D array (handles NaNs by dropping)."""
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
    # base derivatives
    dfdx_expr: sp.Expr
    d2fdx2_expr: sp.Expr
    # denominator (may be 1)
    denom_expr: sp.Expr
    # lambdified callables (vectorized, NumPy)
    f_lam: callable
    dfdx_lam: callable
    d2fdx2_lam: callable
    # per-parameter derivative callables (avoid Matrix lambdify to prevent shape issues)
    df_dp_lams: List[callable]              # list of P callables
    ddfdx_dp_lams: List[callable]           # list of P callables
    d2fdx2_dp_lams: List[callable]          # list of P callables
    dden_dp_lams: Optional[List[callable]]  # list of P callables, or None if denom==1
    denom_lam: Optional[callable]           # callable for denominator, or None if denom==1

    @staticmethod
    def parse(expr_str: str) -> "SympyModel":
        # Define the essential symbols
        i_sym, x_sym = symbols("i x", real=True)

        # Parse expression string safely with SymPy
        local_dict = {"i": i_sym, "x": x_sym}
        expr = sp.sympify(expr_str, locals=local_dict)

        # Identify free parameter symbols (exclude i and x)
        free_syms = list(expr.free_symbols)
        param_syms = [s for s in free_syms if s not in {i_sym, x_sym}]
        # Sort parameters by name for deterministic ordering
        param_syms = sorted(param_syms, key=lambda s: s.name)
        param_names = [s.name for s in param_syms]

        # Derivatives wrt x
        dfdx_expr = sp.diff(expr, x_sym)
        d2fdx2_expr = sp.diff(expr, x_sym, 2)

        # Denominator (if any) for safety constraints
        num, den = sp.fraction(sp.together(expr))
        denom_expr = sp.simplify(den)

        # Per-parameter derivative expressions (scalar or vector-valued when evaluated)
        df_dp_exprs = [sp.diff(expr, p) for p in param_syms] if param_syms else []
        ddfdx_dp_exprs = [sp.diff(dfdx_expr, p) for p in param_syms] if param_syms else []
        d2fdx2_dp_exprs = [sp.diff(d2fdx2_expr, p) for p in param_syms] if param_syms else []

        if denom_expr == 1:
            dden_dp_exprs = None
        else:
            dden_dp_exprs = [sp.diff(denom_expr, p) for p in param_syms] if param_syms else []

        # Lambdify (NumPy) – order of arguments will be (x, i, *params)
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

    # ---------- Helpers to evaluate per-parameter derivatives ----------
    def _eval_param_deriv_list(self, lams: Optional[List[callable]], x: np.ndarray, i_val: float, theta: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluate a list of per-parameter derivative lambdas at (x, i_val, *theta).
        Returns stacked (P, N) array. If no params (P==0), returns (0, N) zeros.
        If lams is None, returns None.
        """
        if lams is None:
            return None
        P = len(self.param_syms)
        x = np.asarray(x)
        x_len = int(np.size(x))
        if P == 0:
            return np.zeros((0, x_len), dtype=float)
        rows = []
        for lam in lams:
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

    # Convenience wrappers expecting parameter vector θ (np.ndarray shape (P,))
    def f(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self.f_lam(x, i_val, *theta), dtype=float)

    def dfdx(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self.dfdx_lam(x, i_val, *theta), dtype=float)

    def d2fdx2(self, x: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
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
    method_fallback: str = "SLSQP"
    order_mode: str = "auto"  # 'auto' | 'g-asc' | 'g-desc'
    auto_direction: bool = True  # if False, force monotonic 'inc' or 'dec' by s_i_override
    s_i_override: Optional[int] = None  # +1 increasing, -1 decreasing
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

        # Ensure needed columns exist
        for col in ["g", "x"]:
            if col not in self.df_i.columns:
                raise ValueError(f"Missing required column '{col}' in data for i={i_value}.")
        if "y" not in self.df_i.columns:
            raise ValueError("Missing required column 'y' in data.")
        # Optional weights
        self.df_i["w"] = 1.0 if "w" not in self.df_i.columns else self.df_i["w"].fillna(1.0)

        # Basic vectors
        self.groups = unique_sorted(self.df_i["g"].to_numpy())
        self.n_g = len(self.groups)
        self.P = len(self.model.param_syms)  # params per group
        if self.P == 0:
            raise ValueError("No free parameters found in the function; nothing to estimate.")

        # x-domain & grids
        x_all = unique_sorted(self.df_i["x"].to_numpy())
        if x_all.size == 0:
            raise ValueError(f"No x values found for i={i_value}.")
        self.x_min, self.x_max = float(np.min(x_all)), float(np.max(x_all))
        self.x_dense = self._build_dense_grid(x_all, self.grid_points)
        # For constraints, include observed x as well (union)
        self.x_const = unique_sorted(np.concatenate([x_all, self.x_dense]))

        # Grouped observed data and weights
        self.obs_by_g: Dict[float, Dict[str, np.ndarray]] = {}
        for g in self.groups:
            sub = self.df_i[(self.df_i["g"] == g) & (~self.df_i["y"].isna())]
            xg = sub["x"].to_numpy(dtype=float)
            yg = sub["y"].to_numpy(dtype=float)
            wg = sub["w"].to_numpy(dtype=float)
            self.obs_by_g[float(g)] = {"x": xg, "y": yg, "w": wg}

        # Initial θ per group (from independent robust least squares where possible)
        self.theta0 = self._build_initial_thetas()

        # Auto monotonic direction s_i
        self.s_i = self._infer_monotonic_direction()

        # Cross-g ordering (sequence of groups). 'auto' sorts by median fitted value.
        self.g_order = self._infer_group_order()

        # Bounds (same for all params)
        lb = np.full(self.P * self.n_g, self.config.lower_bound, dtype=float)
        ub = np.full(self.P * self.n_g, self.config.upper_bound, dtype=float)
        self.bounds = Bounds(lb, ub)

        # Precompute integration weights for smoothness on x_dense
        self.w_smooth = trapz_weights(self.x_dense)

    # ---------- Grids ----------

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

        # Equal-spacing baseline
        step_eq = (b - a) / max(grid_points - 1, 1)

        # Observation-based finer step: 1/5 of minimal observed gap
        if x_all.size >= 2:
            min_gap = np.min(np.diff(x_all))
            step_obs = min_gap / 5.0 if min_gap > 0 else step_eq
        else:
            step_obs = step_eq

        step = min(step_eq, step_obs)
        return self._make_linspace_inclusive(a, b, step)

    # ---------- Initialization ----------

    def _build_initial_thetas(self) -> Dict[float, np.ndarray]:
        rng = np.random.default_rng(self.config.seed if self.config.seed is not None else 123)
        thetas: Dict[float, np.ndarray] = {}
        for g in self.groups:
            xg = self.obs_by_g[float(g)]["x"]
            yg = self.obs_by_g[float(g)]["y"]
            if xg.size >= max(2, self.P):  # attempt per-group LS if some data
                def resid(theta):
                    f = self.model.f(xg, self.i_value, theta)
                    return f - yg

                def jac(theta):
                    J = self.model.df_dp(xg, self.i_value, theta).T  # (N, P)
                    return J

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
                        f_scale=self.config.huber_delta,
                        max_nfev=1000,
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

        # Use observed Spearman if available; else derivative sign from initial thetas
        obs = self.df_i[~self.df_i["y"].isna()]
        if obs.shape[0] >= 3:
            try:
                # Manual Spearman implementation to avoid importing scipy.stats here
                def rankdata(a):
                    a = np.asarray(a, dtype=float)
                    order = np.argsort(a, kind="mergesort")
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(1, a.size + 1, dtype=float)
                    # average ties
                    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
                    cumcounts = np.cumsum(counts)
                    starts = cumcounts - counts + 1
                    avg = (starts + cumcounts) / 2.0
                    ranks = avg[inv]
                    return ranks

                x_r = rankdata(obs["x"].to_numpy())
                y_r = rankdata(obs["y"].to_numpy())
                x_rc = x_r - x_r.mean()
                y_rc = y_r - y_r.mean()
                denom = np.sqrt(np.sum(x_rc**2) * np.sum(y_rc**2))
                rho = float(np.sum(x_rc * y_rc) / denom) if denom > 0 else 0.0
                return +1 if rho >= 0 else -1
            except Exception:
                pass

        # Derivative sign from initial thetas across groups and dense grid
        signs = []
        for g in self.groups:
            d1 = self.model.dfdx(self.x_dense, self.i_value, self.theta0[float(g)])
            med = np.nanmedian(d1)
            if np.isfinite(med) and med != 0.0:
                signs.append(np.sign(med))
        if len(signs) == 0:
            return +1
        s = int(np.sign(np.sum(signs)))
        return +1 if s >= 0 else -1

    def _infer_group_order(self) -> List[float]:
        mode = self.config.order_mode
        if mode in ("g-asc", "g-desc"):
            ordered = sorted(self.groups)
            return ordered if mode == "g-asc" else ordered[::-1]

        # Auto: sort by median of initial f(x) across dense grid
        med_vals = []
        for g in self.groups:
            fx = self.model.f(self.x_dense, self.i_value, self.theta0[float(g)])
            med_vals.append((float(g), float(np.nanmedian(fx))))
        ordered = [g for g, _ in sorted(med_vals, key=lambda t: t[1])]
        return ordered

    # ---------- Mapping θ between flat vector and per-group ----------

    def _theta_flat_to_dict(self, theta_flat: np.ndarray) -> Dict[float, np.ndarray]:
        out: Dict[float, np.ndarray] = {}
        for idx, g in enumerate(self.groups):
            start = idx * self.P
            out[float(g)] = theta_flat[start : start + self.P]
        return out

    def _theta_dict_to_flat(self, thetas: Dict[float, np.ndarray]) -> np.ndarray:
        arrs = [thetas[float(g)] for g in self.groups]
        return np.concatenate(arrs, axis=0)

    # ---------- Objective and gradient ----------

    def _objective_and_grad(self, theta_flat: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, float]]:
        cfg = self.config
        thetas = self._theta_flat_to_dict(theta_flat)

        L_fit = 0.0
        g_fit = np.zeros_like(theta_flat)

        # L_fit: robust Huber
        for idx_g, g in enumerate(self.groups):
            th = thetas[float(g)]
            obs = self.obs_by_g[float(g)]
            xg, yg, wg = obs["x"], obs["y"], obs["w"]

            if xg.size > 0:
                f = self.model.f(xg, self.i_value, th)
                r = f - yg
                v, dr = huber_value_and_grad(r, cfg.huber_delta)
                L_fit += float(np.sum(wg * v))

                # Gradient wrt params: sum_j w_j * dρ/dr * df/dθ
                J = self.model.df_dp(xg, self.i_value, th)  # (P, N)
                grad_g = J @ (wg * dr)  # (P,)
                g_fit[idx_g * self.P : (idx_g + 1) * self.P] = grad_g

        # L_smooth: ∫ (f''(x))^2 dx per group (trapz approximation)
        L_smooth = 0.0
        g_smooth = np.zeros_like(theta_flat)
        w = self.w_smooth  # (M,)
        for idx_g, g in enumerate(self.groups):
            th = thetas[float(g)]
            d2 = self.model.d2fdx2(self.x_dense, self.i_value, th)  # (M,)
            # Value
            L_smooth += float(np.sum(w * (d2 ** 2)))
            # Grad wrt params: ∫ 2 f'' * ∂(f'')/∂θ
            d2_dp = self.model.d2fdx2_dp(self.x_dense, self.i_value, th)  # (P, M)
            grad_g = d2_dp @ (w * (2.0 * d2))  # (P,)
            g_smooth[idx_g * self.P : (idx_g + 1) * self.P] = grad_g

        # L_sim: sum_g ||θ_g - θ̄||^2
        L_sim = 0.0
        g_sim = np.zeros_like(theta_flat)
        Theta = np.stack([thetas[float(g)] for g in self.groups], axis=0)  # (G,P)
        theta_bar = np.mean(Theta, axis=0)
        diffs = Theta - theta_bar  # (G,P)
        L_sim = float(np.sum(diffs ** 2))
        # Gradient: ∂/∂θ_g L_sim = 2 (θ_g - θ̄)
        for idx_g in range(self.n_g):
            g_sim[idx_g * self.P : (idx_g + 1) * self.P] = 2.0 * diffs[idx_g]

        # Total
        L = L_fit + cfg.lambda_s * L_smooth + cfg.lambda_sim * L_sim
        grad = g_fit + cfg.lambda_s * g_smooth + cfg.lambda_sim * g_sim

        # Diagnostics
        diag = {
            "L_fit": L_fit,
            "L_smooth": L_smooth,
            "L_sim": L_sim,
        }
        return L, grad, diag

    # ---------- Constraints (values & sparse Jacobians) ----------

    def _constraints_mono(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        """Monotonicity: s_i * f'(x) - eps_mono >= 0 for each g and x in x_const."""
        s = float(self.s_i)
        eps = float(self.config.eps_mono)
        xs = self.x_const
        G, P = self.n_g, self.P
        M = xs.size

        vals = []
        rows = []
        cols = []
        data = []
        # Build constraint vector by concatenating groups
        for idx_g, g in enumerate(self.groups):
            th = theta_flat[idx_g * P : (idx_g + 1) * P]
            d1 = self.model.dfdx(xs, self.i_value, th)  # (M,)
            cvals = s * d1 - eps
            vals.append(cvals)

            # Jacobian block: s * ∂(f')/∂θ at each x
            d1_dp = self.model.ddfdx_dp(xs, self.i_value, th)  # (P, M)
            # For each constraint row k, fill columns idx_g*P:(idx_g+1)*P with s * d1_dp[:,k]
            for k in range(M):
                row_idx = idx_g * M + k
                rows.extend([row_idx] * P)
                cols.extend(list(range(idx_g * P, (idx_g + 1) * P)))
                data.extend(list(s * d1_dp[:, k]))

        vals = np.concatenate(vals, axis=0) if len(vals) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(G * M, G * P))
        return vals, J

    def _constraints_order(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        """
        Cross-g ordering: for adjacent pairs in g_order: f_next(x) - f_curr(x) - eps_ord >= 0.
        """
        eps = float(self.config.eps_ord)
        xs = self.x_const
        P = self.P
        M = xs.size
        pairs = list(zip(self.g_order[:-1], self.g_order[1:]))
        n_pairs = len(pairs)

        vals = []
        rows = []
        cols = []
        data = []

        # Map g -> idx in groups (flat θ blocks)
        g_to_idx = {float(g): idx for idx, g in enumerate(self.groups)}

        for pidx, (g_curr, g_next) in enumerate(pairs):
            idx_c = g_to_idx[float(g_curr)]
            idx_n = g_to_idx[float(g_next)]
            th_c = theta_flat[idx_c * P : (idx_c + 1) * P]
            th_n = theta_flat[idx_n * P : (idx_n + 1) * P]

            f_c = self.model.f(xs, self.i_value, th_c)  # (M,)
            f_n = self.model.f(xs, self.i_value, th_n)  # (M,)
            cvals = (f_n - f_c) - eps
            vals.append(cvals)

            # Jacobian rows for this pair
            df_dp_c = self.model.df_dp(xs, self.i_value, th_c)  # (P, M)
            df_dp_n = self.model.df_dp(xs, self.i_value, th_n)  # (P, M)

            for k in range(M):
                row_idx = pidx * M + k
                # wrt next: +df/dθ
                rows.extend([row_idx] * P)
                cols.extend(list(range(idx_n * P, (idx_n + 1) * P)))
                data.extend(list(df_dp_n[:, k]))
                # wrt current: -df/dθ
                rows.extend([row_idx] * P)
                cols.extend(list(range(idx_c * P, (idx_c + 1) * P)))
                data.extend(list(-df_dp_c[:, k]))

        vals = np.concatenate(vals, axis=0) if len(vals) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(n_pairs * M, self.n_g * P))
        return vals, J

    def _constraints_den(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        """
        Denominator safety: den(x)^2 - eps_den^2 >= 0 at x in x_const, per g.
        If denom is 1, returns empty constraints.
        """
        if self.model.denom_lam is None or self.model.dden_dp_lams is None:
            return np.array([], dtype=float), sparse.csr_matrix((0, self.n_g * self.P))

        xs = self.x_const
        P = self.P
        M = xs.size
        eps2 = float(self.config.eps_den) ** 2

        vals = []
        rows = []
        cols = []
        data = []

        for idx_g, g in enumerate(self.groups):
            th = theta_flat[idx_g * P : (idx_g + 1) * P]
            den = self.model.denom(xs, self.i_value, th)  # (M,)
            cvals = (den ** 2) - eps2
            vals.append(cvals)

            dden_dp = self.model.dden_dp(xs, self.i_value, th)  # (P, M)
            # jac: 2*den * ∂den/∂θ
            for k in range(M):
                row_idx = idx_g * M + k
                rows.extend([row_idx] * P)
                cols.extend(list(range(idx_g * P, (idx_g + 1) * P)))
                data.extend(list(2.0 * den[k] * dden_dp[:, k]))

        vals = np.concatenate(vals, axis=0) if len(vals) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_g * M, self.n_g * P))
        return vals, J

    # ---------- Scipy hooks ----------

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

    # ---------- Solve ----------

    def solve(self) -> Tuple[np.ndarray, Dict]:
        theta0_flat = self._theta_dict_to_flat(self.theta0)
        cons = NonlinearConstraint(self.cons_fun, lb=0.0, ub=np.inf, jac=self.cons_jac)

        res = minimize(
            fun=self.objective,
            x0=theta0_flat,
            jac=self.objective_grad,
            bounds=self.bounds,
            constraints=[cons],
            method=self.config.method_primary,
            options={"maxiter": self.config.maxiter, "verbose": 3 if self.config.verbose else 0},
        )

        if (not res.success) and (self.config.method_fallback is not None):
            # Fallback with SLSQP (uses same constraint functions; SciPy accepts vector constraints here)
            if self.config.verbose:
                print(f"[i={self.i_value}] Primary solver failed ({res.message}). Trying SLSQP...")
            res2 = minimize(
                fun=self.objective,
                x0=res.x,  # warm start from previous attempt
                jac=self.objective_grad,
                bounds=self.bounds,
                constraints=[{"type": "ineq", "fun": self.cons_fun, "jac": self.cons_jac}],
                method=self.config.method_fallback,
                options={"maxiter": self.config.maxiter, "ftol": 1e-9, "disp": self.config.verbose},
            )
            if res2.success:
                res = res2

        # Diagnostics
        _, _, diag = self._objective_and_grad(res.x)
        cons_vals, _ = self.constraints_all(res.x)

        info = {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "niter": int(res.nit) if hasattr(res, "nit") else None,
            "fun": float(res.fun),
            "L_fit": float(diag.get("L_fit", np.nan)),
            "L_smooth": float(diag.get("L_smooth", np.nan)),
            "L_sim": float(diag.get("L_sim", np.nan)),
            "max_constraint_violation": float(np.max(np.maximum(0.0, -cons_vals))) if cons_vals.size > 0 else 0.0,
        }
        return res.x, info

    # ---------- Post-solve checks ----------

    def compute_checks(self, theta_flat: np.ndarray) -> Dict[str, float]:
        thetas = self._theta_flat_to_dict(theta_flat)
        xs = self.x_const
        P = self.P

        # Monotonicity violation: max over g,x of (eps - s_i * f'(x))
        mono_vals = []
        for g in self.groups:
            th = thetas[float(g)]
            d1 = self.model.dfdx(xs, self.i_value, th)
            mono_vals.append(self.config.eps_mono - self.s_i * d1)
        mono_violation = float(np.max(np.concatenate(mono_vals))) if mono_vals else 0.0

        # Ordering violation: max over adj pairs, x of (eps - (f_next - f_curr))
        ord_vals = []
        pairs = list(zip(self.g_order[:-1], self.g_order[1:]))
        for (g_c, g_n) in pairs:
            th_c = thetas[float(g_c)]
            th_n = thetas[float(g_n)]
            f_c = self.model.f(xs, self.i_value, th_c)
            f_n = self.model.f(xs, self.i_value, th_n)
            ord_vals.append(self.config.eps_ord - (f_n - f_c))
        ord_violation = float(np.max(np.concatenate(ord_vals))) if ord_vals else 0.0

        # Denominator min abs across all g,x
        denom_min_abs = math.inf
        if self.model.denom_lam is not None:
            for g in self.groups:
                th = thetas[float(g)]
                den = self.model.denom(xs, self.i_value, th)
                denom_min_abs = min(denom_min_abs, float(np.min(np.abs(den))))
        else:
            denom_min_abs = float("inf")

        # Smooth penalty (unweighted) value
        w = self.w_smooth
        L_smooth = 0.0
        for g in self.groups:
            th = thetas[float(g)]
            d2 = self.model.d2fdx2(self.x_dense, self.i_value, th)
            L_smooth += float(np.sum(w * (d2 ** 2)))

        return {
            "monotonicity_violation_max": mono_violation,
            "ordering_violation_max": ord_violation,
            "denominator_min_abs": denom_min_abs,
            "smooth_penalty_value": L_smooth,
        }

    # ---------- Outputs ----------

    def export_plots(self, theta_flat: np.ndarray, fname: str) -> None:
        thetas = self._theta_flat_to_dict(theta_flat)

        plt.figure()
        # Curves per g
        for g in self.groups:
            th = thetas[float(g)]
            xs = self.x_dense
            ys = self.model.f(xs, self.i_value, th)
            plt.plot(xs, ys, label=f"g={g}")

        # Observed points (non-missing y) per g
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
            th = theta_flat[idx_g * self.P : (idx_g + 1) * self.P]
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
        # Compute y_hat for all rows in df_i
        df = self.df_i.copy()
        y_hat_vals = []
        for _, row in df.iterrows():
            g = float(row["g"])
            x = float(row["x"])
            th = thetas[g]
            y_hat_vals.append(float(self.model.f(np.array([x]), self.i_value, th)[0]))
        df["y_hat"] = np.array(y_hat_vals, dtype=float)
        df["source"] = np.where(df["y"].isna(), "imputed", "observed")
        # Keep only required columns in order
        return df[["g", "i", "x", "y", "y_hat", "source"]].copy()


# ----------------------------- CLI & Orchestration -----------------------------

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
    return p.parse_args()


def main():
    args = parse_args()

    # Validate & prepare output dir
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Read function
    with open(args.func, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("function.txt first line is empty.")
        expr_str = first_line

    model = SympyModel.parse(expr_str)

    # Read data
    na = ["NA", "NaN", ""]
    df = pd.read_csv(args.data, na_values=na)
    # Basic validation
    for col in ["g", "i", "x"]:
        if col not in df.columns:
            raise ValueError(f"data.csv missing required column '{col}'.")
    if "y" not in df.columns:
        raise ValueError("data.csv missing required column 'y'.")
    # Ensure numeric
    df["g"] = pd.to_numeric(df["g"], errors="coerce")
    df["i"] = pd.to_numeric(df["i"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "w" in df.columns:
        df["w"] = pd.to_numeric(df["w"], errors="coerce").fillna(1.0)

    # Unique i values
    i_values = unique_sorted(df["i"].to_numpy())
    if i_values.size == 0:
        raise ValueError("No i values found in data.")

    # Config
    cfg = SolverConfig(
        eps_mono=float(args.eps_mono),
        eps_ord=float(args.eps_ord),
        eps_den=float(args.eps_den),
        lambda_s=float(args.lambda_s),
        lambda_sim=float(args.lambda_sim),
        huber_delta=1.0,
        lower_bound=float(args.lower),
        upper_bound=float(args.upper),
        maxiter=int(args.maxiter),
        verbose=bool(args.verbose),
        method_primary="trust-constr",
        method_fallback="SLSQP",
        order_mode=args.order_mode,
        auto_direction=(str(args.auto_direction).lower() in ("true", "1", "yes", "y")),
        s_i_override=(+1 if args.direction == "inc" else (-1 if args.direction == "dec" else None)),
        seed=int(args.seed),
    )

    # Global collectors
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
        }
    }

    any_failure = False

    for i_val in i_values:
        df_i = df[df["i"] == i_val].copy()
        fitter = FitSingleI(
            df_i=df_i,
            i_value=float(i_val),
            model=model,
            grid_points=int(args.grid_points),
            config=cfg,
            outdir=outdir,
        )

        theta_hat, info = fitter.solve()

        # Diagnostics / checks
        ch = fitter.compute_checks(theta_hat)

        # Export plots
        plot_path = os.path.join(outdir, f"plots_i={i_val}.png")
        try:
            fitter.export_plots(theta_hat, plot_path)
        except Exception as e:
            warnings.warn(f"Plotting failed for i={i_val}: {e}")

        # Export tables
        all_params.append(fitter.export_params(theta_hat))
        all_grid.append(fitter.export_grid(theta_hat))
        all_fitted.append(fitter.export_fitted(theta_hat))

        # Compose per-i checks
        per_i_entry = {
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
        }
        checks["per_i"].append(per_i_entry)

        # Violation logic: any positive violation or solver failure triggers non-zero exit
        violated = (
            (ch["monotonicity_violation_max"] > 1e-9) or
            (ch["ordering_violation_max"] > 1e-9) or
            (np.isfinite(ch["denominator_min_abs"]) and (ch["denominator_min_abs"] < cfg.eps_den - 1e-12)) or
            (not info.get("success", False))
        )
        if violated:
            any_failure = True

    # Save combined outputs
    params_df = pd.concat(all_params, ignore_index=True) if all_params else pd.DataFrame(columns=["i","g","param","value"])
    grid_df = pd.concat(all_grid, ignore_index=True) if all_grid else pd.DataFrame(columns=["i","g","x","f_x"])
    fitted_df = pd.concat(all_fitted, ignore_index=True) if all_fitted else pd.DataFrame(columns=["g","i","x","y","y_hat","source"])

    params_path = os.path.join(outdir, "params.csv")
    grid_path = os.path.join(outdir, "grid.csv")
    fitted_path = os.path.join(outdir, "fitted.csv")
    checks_path = os.path.join(outdir, "checks.json")

    params_df.to_csv(params_path, index=False)
    grid_df.to_csv(grid_path, index=False)
    fitted_df.to_csv(fitted_path, index=False)

    # Overall checks
    if checks["per_i"]:
        checks["overall"] = {
            "monotonicity_violation_max": float(np.max([c["monotonicity_violation_max"] for c in checks["per_i"]])),
            "ordering_violation_max": float(np.max([c["ordering_violation_max"] for c in checks["per_i"]])),
            "denominator_min_abs": float(np.min([c["denominator_min_abs"] for c in checks["per_i"]])),
            "smooth_penalty_value": float(np.sum([c["smooth_penalty_value"] for c in checks["per_i"]])),
            "any_solver_failure": bool(any(not c["solver"]["success"] for c in checks["per_i"])),
        }
    else:
        checks["overall"] = {}

    with open(checks_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    # Exit code
    if any_failure:
        print("Constraint violation or solver failure detected. See checks.json for details.", file=sys.stderr)
        sys.exit(2)
    else:
        print("Fitting completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
