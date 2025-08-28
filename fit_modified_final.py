#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit.py — Joint constrained curve fitting across all i, with one parameter vector per g
=====================================================================================

Change from previous version:
- Parameters θ_g are **shared across all i** (same g, same θ), even though f depends on i.
- Objective sums over all (g, i) observations; smoothness & constraints are enforced per i.
- Outputs keep the same files; `params.csv` will list a copy of the same θ_g under each i
  (to preserve schema `i,g,param,value`). Values are identical across i by design.

Other features kept:
- Robust Huber loss
- Smoothness penalty ∫ (f''(x))^2 dx via trapezoid rule
- Cross-g similarity penalty on θ_g vs θ̄
- Hard constraints:
    * Monotonicity per (i, g): s_i * f'(x; i, θ_g) >= eps_mono
    * Cross-g non-crossing per i: f_{g(j+1)}(x) - f_{g(j)}(x) >= eps_ord
    * Denominator safety per (i, g): den(x; i, θ_g)^2 - eps_den^2 >= 0
- SymPy derivatives per-parameter (avoid shape issues), SLSQP dense Jacobian fallback,
  and np.errstate-wrapped numeric evaluations to silence transient warnings.

CLI example
-----------
python fit.py --data data.csv --func function.txt --outdir out \
  --grid-points 200 --auto-direction true --lambda-s 1e-4 --lambda-sim 1e-3

Exit code: 0 on success; 2 on violation/solver failure.
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

        param_syms = sorted([s for s in expr.free_symbols if s not in {i_sym, x_sym}], key=lambda s: s.name)
        param_names = [s.name for s in param_syms]

        dfdx_expr = sp.diff(expr, x_sym)
        d2fdx2_expr = sp.diff(expr, x_sym, 2)

        num, den = sp.fraction(sp.together(expr))
        denom_expr = sp.simplify(den)

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


# ----------------------------- Global fitter (shared θ per g) -----------------------------

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
    s_i_override: Optional[int] = None
    seed: Optional[int] = 42

    # If True, weights and smoothness penalties are rescaled on a per‑i basis
    # according to the dynamic range of the observed y values.  See the
    # description of the --balance-i CLI flag for details.
    balance_i: bool = False


class FitGlobal:
    def __init__(
        self,
        df: pd.DataFrame,
        model: SympyModel,
        grid_points: int,
        config: SolverConfig,
        outdir: str,
    ):
        self.df = df.copy()
        self.model = model
        self.grid_points = int(grid_points)
        self.config = config
        self.outdir = outdir

        # sets
        self.i_values = unique_sorted(self.df["i"].to_numpy())
        self.groups = unique_sorted(self.df["g"].to_numpy())
        self.n_i = len(self.i_values)
        self.n_g = len(self.groups)
        self.P = len(self.model.param_syms)
        if self.P == 0:
            raise ValueError("No free parameters found in the function; nothing to estimate.")

        # collect observed and grids per i, per g
        self.obs_by_gi: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}
        self.x_dense: Dict[float, np.ndarray] = {}
        self.x_const: Dict[float, np.ndarray] = {}
        self.w_smooth: Dict[float, np.ndarray] = {}

        for i_val in self.i_values:
            df_i = self.df[self.df["i"] == i_val]
            # observed x union
            x_all = unique_sorted(df_i["x"].to_numpy())
            if x_all.size == 0:
                raise ValueError(f"No x values found for i={i_val}.")
            xs_dense = self._build_dense_grid(x_all, self.grid_points)
            xs_const = unique_sorted(np.concatenate([x_all, xs_dense]))
            self.x_dense[float(i_val)] = xs_dense
            self.x_const[float(i_val)] = xs_const
            self.w_smooth[float(i_val)] = trapz_weights(xs_dense)

            for g in self.groups:
                sub = df_i[(df_i["g"] == g) & (~df_i["y"].isna())]
                self.obs_by_gi[(float(g), float(i_val))] = {
                    "x": sub["x"].to_numpy(dtype=float),
                    "y": sub["y"].to_numpy(dtype=float),
                    # Make a copy of weights so that per‑i balancing can safely modify them
                    "w": (sub["w"].to_numpy(dtype=float) if "w" in sub.columns else np.ones(sub.shape[0], dtype=float)).copy(),
                }

        # infer s_i per i
        self.s_i: Dict[float, int] = {}
        for i_val in self.i_values:
            self.s_i[float(i_val)] = self._infer_monotonic_direction(i_val)

        # Optional: scale weights and smoothness per i to balance contributions across i
        # The user can enable this behaviour via the --balance-i flag.  The scaling
        # factor for each i is defined as 1/(max(y_i) - min(y_i)), where y_i are
        # the observed y values across all groups for that i.  If the range is
        # zero or undefined, the scale factor defaults to 1.  Both the
        # observation weights and the smoothness weights are multiplied by this
        # factor.  This ensures that data sets with small magnitude y values do
        # not get overwhelmed in the objective by those with much larger y values.
        self.balance_scales: Dict[float, float] = {}
        if config.balance_i:
            for i_val in self.i_values:
                # gather all observed y across groups for this i
                ys_i = self.df[(self.df["i"] == i_val) & (~self.df["y"].isna())]["y"].to_numpy(dtype=float)
                if ys_i.size == 0:
                    scale = 1.0
                else:
                    finite_ys = ys_i[np.isfinite(ys_i)]
                    if finite_ys.size == 0:
                        scale = 1.0
                    else:
                        y_min = float(np.min(finite_ys))
                        y_max = float(np.max(finite_ys))
                        y_range = y_max - y_min
                        # Avoid division by zero when all y are equal
                        scale = 1.0 / y_range if y_range > 0.0 else 1.0
                self.balance_scales[float(i_val)] = scale
                # Apply scale to weights for each (g,i)
                for g in self.groups:
                    obs_key = (float(g), float(i_val))
                    obs = self.obs_by_gi.get(obs_key)
                    if obs is not None and obs["w"].size > 0:
                        obs["w"] = obs["w"] * scale
                # Scale smoothness weight vector
                self.w_smooth[float(i_val)] = self.w_smooth[float(i_val)] * scale
        else:
            for i_val in self.i_values:
                self.balance_scales[float(i_val)] = 1.0

        # initial θ_g using all i's data for each g (robust LS)
        self.theta0 = self._init_thetas_global()

        # infer order per i (sequence of g)
        self.g_order: Dict[float, List[float]] = {}
        for i_val in self.i_values:
            self.g_order[float(i_val)] = self._infer_group_order(i_val)

        # bounds
        lb = np.full(self.P * self.n_g, self.config.lower_bound, dtype=float)
        ub = np.full(self.P * self.n_g, self.config.upper_bound, dtype=float)
        self.bounds = Bounds(lb, ub)

    # ---------- Helpers ----------

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

    def _infer_monotonic_direction(self, i_val: float) -> int:
        cfg = self.config
        if (not cfg.auto_direction) and (cfg.s_i_override in (+1, -1)):
            return int(cfg.s_i_override)

        obs = self.df[(self.df["i"] == i_val) & (~self.df["y"].isna())]
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
        # fallback: assume increasing
        return +1

    def _init_thetas_global(self) -> Dict[float, np.ndarray]:
        rng = np.random.default_rng(self.config.seed if self.config.seed is not None else 123)
        thetas: Dict[float, np.ndarray] = {}
        for g in self.groups:
            # stack all observations over i for this g
            xs_all = []
            ys_all = []
            for i_val in self.i_values:
                obs = self.obs_by_gi[(float(g), float(i_val))]
                xs_all.append(obs["x"]); ys_all.append(obs["y"])
            xg = np.concatenate(xs_all) if len(xs_all) else np.array([], dtype=float)
            yg = np.concatenate(ys_all) if len(ys_all) else np.array([], dtype=float)
            if xg.size >= max(2, self.P):
                def resid(theta):
                    r_list = []
                    for i_val in self.i_values:
                        obs = self.obs_by_gi[(float(g), float(i_val))]
                        if obs["x"].size == 0: continue
                        f = self.model.f(obs["x"], float(i_val), theta)
                        r_list.append(f - obs["y"])
                    return np.concatenate(r_list) if len(r_list) else np.zeros(0, dtype=float)
                def jac(theta):
                    J_rows = []
                    for i_val in self.i_values:
                        obs = self.obs_by_gi[(float(g), float(i_val))]
                        if obs["x"].size == 0: continue
                        J = self.model.df_dp(obs["x"], float(i_val), theta).T  # (N_i, P)
                        J_rows.append(J)
                    return np.vstack(J_rows) if len(J_rows) else np.zeros((0, self.P), dtype=float)
                try:
                    res = least_squares(
                        resid, x0=np.zeros(self.P, dtype=float), jac=jac,
                        bounds=(np.full(self.P, self.config.lower_bound), np.full(self.P, self.config.upper_bound)),
                        loss="huber", f_scale=self.config.huber_delta, max_nfev=800, verbose=0
                    )
                    theta_g = np.clip(res.x, self.config.lower_bound, self.config.upper_bound)
                except Exception:
                    theta_g = rng.normal(scale=1e-2, size=self.P)
            else:
                theta_g = rng.normal(scale=1e-2, size=self.P)
            thetas[float(g)] = theta_g.astype(float)
        return thetas

    def _infer_group_order(self, i_val: float) -> List[float]:
        mode = self.config.order_mode
        if mode in ("g-asc", "g-desc"):
            ordered = sorted(self.groups)
            return ordered if mode == "g-asc" else ordered[::-1]
        # auto: sort by median f(x; i, θ_g) using initial θ0 on dense grid of this i
        xs = self.x_dense[float(i_val)]
        med_vals = []
        for g in self.groups:
            fx = self.model.f(xs, float(i_val), self.theta0[float(g)])
            med_vals.append((float(g), float(np.nanmedian(fx))))
        ordered = [g for g, _ in sorted(med_vals, key=lambda t: t[1])]
        return ordered

    # ---------- θ mapping ----------
    def _theta_flat_to_dict(self, theta_flat: np.ndarray) -> Dict[float, np.ndarray]:
        out: Dict[float, np.ndarray] = {}
        for idx, g in enumerate(self.groups):
            out[float(g)] = theta_flat[idx*self.P:(idx+1)*self.P]
        return out

    def _theta_dict_to_flat(self, thetas: Dict[float, np.ndarray]) -> np.ndarray:
        return np.concatenate([thetas[float(g)] for g in self.groups], axis=0)

    # ---------- Objective & grad ----------
    def _objective_and_grad(self, theta_flat: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, float]]:
        cfg = self.config
        thetas = self._theta_flat_to_dict(theta_flat)

        L_fit = 0.0
        g_fit = np.zeros_like(theta_flat)

        # Fit across all i and g
        for idx_g, g in enumerate(self.groups):
            th = thetas[float(g)]
            grad_acc = np.zeros(self.P, dtype=float)
            for i_val in self.i_values:
                obs = self.obs_by_gi[(float(g), float(i_val))]
                xg, yg, wg = obs["x"], obs["y"], obs["w"]
                if xg.size == 0: 
                    continue
                f = self.model.f(xg, float(i_val), th)
                r = f - yg
                v, dr = huber_value_and_grad(r, cfg.huber_delta)
                L_fit += float(np.sum(wg * v))
                J = self.model.df_dp(xg, float(i_val), th)  # (P,N)
                grad_acc += J @ (wg * dr)
            g_fit[idx_g*self.P:(idx_g+1)*self.P] = grad_acc

        # Smoothness across all i and g
        L_smooth = 0.0
        g_smooth = np.zeros_like(theta_flat)
        for idx_g, g in enumerate(self.groups):
            th = thetas[float(g)]
            grad_acc = np.zeros(self.P, dtype=float)
            for i_val in self.i_values:
                xs = self.x_dense[float(i_val)]
                w = self.w_smooth[float(i_val)]
                d2 = self.model.d2fdx2(xs, float(i_val), th)  # (M,)
                L_smooth += float(np.sum(w * (d2**2)))
                d2_dp = self.model.d2fdx2_dp(xs, float(i_val), th)  # (P,M)
                grad_acc += d2_dp @ (w * (2.0*d2))
            g_smooth[idx_g*self.P:(idx_g+1)*self.P] = grad_acc

        # Similarity (θ_g vs θ̄)
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

    # ---------- Constraints ----------
    def _constraints_mono(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        P = self.P
        vals_all = []
        rows = []
        cols = []
        data = []
        row_cursor = 0

        for i_val in self.i_values:
            s = float(self.s_i[float(i_val)])
            eps = float(self.config.eps_mono)
            xs = self.x_const[float(i_val)]
            M = xs.size
            for idx_g, g in enumerate(self.groups):
                th = theta_flat[idx_g*P:(idx_g+1)*P]
                d1 = self.model.dfdx(xs, float(i_val), th)  # (M,)
                cvals = s * d1 - eps
                vals_all.append(cvals)
                d1_dp = self.model.ddfdx_dp(xs, float(i_val), th)  # (P,M)
                for k in range(M):
                    row = row_cursor + k
                    rows.extend([row]*P)
                    cols.extend(range(idx_g*P, (idx_g+1)*P))
                    data.extend(list(s * d1_dp[:, k]))
                row_cursor += M

        vals = np.concatenate(vals_all, axis=0) if len(vals_all) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(row_cursor, self.n_g*P))
        return vals, J

    def _constraints_order(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        P = self.P
        vals_all = []
        rows = []
        cols = []
        data = []
        row_cursor = 0

        for i_val in self.i_values:
            eps = float(self.config.eps_ord)
            xs = self.x_const[float(i_val)]
            M = xs.size
            pairs = list(zip(self.g_order[float(i_val)][:-1], self.g_order[float(i_val)][1:]))
            g_to_idx = {float(g): idx for idx, g in enumerate(self.groups)}
            for (g_c, g_n) in pairs:
                ic = g_to_idx[float(g_c)]
                inn = g_to_idx[float(g_n)]
                th_c = theta_flat[ic*P:(ic+1)*P]
                th_n = theta_flat[inn*P:(inn+1)*P]
                f_c = self.model.f(xs, float(i_val), th_c)
                f_n = self.model.f(xs, float(i_val), th_n)
                cvals = (f_n - f_c) - eps
                vals_all.append(cvals)
                df_dp_c = self.model.df_dp(xs, float(i_val), th_c)  # (P,M)
                df_dp_n = self.model.df_dp(xs, float(i_val), th_n)  # (P,M)
                for k in range(M):
                    row = row_cursor + k
                    rows.extend([row]*P); cols.extend(range(inn*P, (inn+1)*P)); data.extend(list(df_dp_n[:, k]))
                    rows.extend([row]*P); cols.extend(range(ic*P, (ic+1)*P)); data.extend(list(-df_dp_c[:, k]))
                row_cursor += M

        vals = np.concatenate(vals_all, axis=0) if len(vals_all) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(row_cursor, self.n_g*P))
        return vals, J

    def _constraints_den(self, theta_flat: np.ndarray) -> Tuple[np.ndarray, sparse.csr_matrix]:
        if self.model.denom_lam is None or self.model.dden_dp_lams is None:
            return np.array([], dtype=float), sparse.csr_matrix((0, self.n_g*self.P))

        P = self.P
        vals_all = []
        rows = []
        cols = []
        data = []
        row_cursor = 0

        eps2 = float(self.config.eps_den)**2
        for i_val in self.i_values:
            xs = self.x_const[float(i_val)]
            M = xs.size
            for idx_g, g in enumerate(self.groups):
                th = theta_flat[idx_g*P:(idx_g+1)*P]
                den = self.model.denom(xs, float(i_val), th)  # (M,)
                cvals = (den**2) - eps2
                vals_all.append(cvals)
                dden_dp = self.model.dden_dp(xs, float(i_val), th)  # (P,M)
                for k in range(M):
                    row = row_cursor + k
                    rows.extend([row]*P)
                    cols.extend(range(idx_g*P, (idx_g+1)*P))
                    data.extend(list(2.0 * den[k] * dden_dp[:, k]))
                row_cursor += M

        vals = np.concatenate(vals_all, axis=0) if len(vals_all) else np.array([], dtype=float)
        J = sparse.csr_matrix((data, (rows, cols)), shape=(row_cursor, self.n_g*P))
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
            res2 = minimize(
                fun=self.objective,
                x0=res.x,
                jac=self.objective_grad,
                bounds=self.bounds,
                constraints=[{
                    "type": "ineq",
                    "fun": self.cons_fun,
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

    # ---------- Checks & Exports ----------
    def compute_checks_per_i(self, theta_flat: np.ndarray) -> List[Dict[str, float]]:
        thetas = self._theta_flat_to_dict(theta_flat)
        out = []
        for i_val in self.i_values:
            xs = self.x_const[float(i_val)]
            # mono
            mono_vals = []
            for g in self.groups:
                th = thetas[float(g)]
                d1 = self.model.dfdx(xs, float(i_val), th)
                mono_vals.append(self.config.eps_mono - self.s_i[float(i_val)] * d1)
            mono_violation = float(np.max(np.concatenate(mono_vals))) if mono_vals else 0.0
            # ordering
            ord_vals = []
            pairs = list(zip(self.g_order[float(i_val)][:-1], self.g_order[float(i_val)][1:]))
            for (g_c, g_n) in pairs:
                th_c = thetas[float(g_c)]
                th_n = thetas[float(g_n)]
                f_c = self.model.f(xs, float(i_val), th_c)
                f_n = self.model.f(xs, float(i_val), th_n)
                ord_vals.append(self.config.eps_ord - (f_n - f_c))
            ord_violation = float(np.max(np.concatenate(ord_vals))) if ord_vals else 0.0
            # denom
            denom_min_abs = math.inf
            if self.model.denom_lam is not None:
                for g in self.groups:
                    th = thetas[float(g)]
                    den = self.model.denom(xs, float(i_val), th)
                    denom_min_abs = min(denom_min_abs, float(np.min(np.abs(den))))
            else:
                denom_min_abs = float("inf")
            # smooth (unweighted report)
            xs_d = self.x_dense[float(i_val)]
            w = trapz_weights(xs_d)
            L_smooth = 0.0
            for g in self.groups:
                th = thetas[float(g)]
                d2 = self.model.d2fdx2(xs_d, float(i_val), th)
                L_smooth += float(np.sum(w * (d2**2)))
            out.append({
                "i": float(i_val),
                "monotonicity_violation_max": mono_violation,
                "ordering_violation_max": ord_violation,
                "denominator_min_abs": denom_min_abs,
                "smooth_penalty_value": L_smooth,
            })
        return out

    def export_plots(self, theta_flat: np.ndarray, outdir: str) -> None:
        thetas = self._theta_flat_to_dict(theta_flat)
        for i_val in self.i_values:
            plt.figure()
            xs = self.x_dense[float(i_val)]
            for g in self.groups:
                th = thetas[float(g)]
                ys = self.model.f(xs, float(i_val), th)
                plt.plot(xs, ys, label=f"g={g}")
            # scatter observed
            df_i = self.df[self.df["i"] == i_val]
            for g in self.groups:
                sub = df_i[(df_i["g"] == g) & (~df_i["y"].isna())]
                if sub.shape[0] > 0:
                    plt.scatter(sub["x"], sub["y"], s=16, alpha=0.7)
            plt.title(f"i = {i_val} — fitted f(x) by g (shared θ)")
            plt.xlabel("x"); plt.ylabel("f(x) / y")
            plt.legend(loc="best", fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"plots_i={i_val}.png"), dpi=180)
            plt.close()

    def export_params(self, theta_flat: np.ndarray) -> pd.DataFrame:
        # Duplicate the same θ_g values for each i to preserve schema (i,g,param,value)
        rows = []
        for i_val in self.i_values:
            for idx_g, g in enumerate(self.groups):
                th = theta_flat[idx_g*self.P:(idx_g+1)*self.P]
                for p_name, val in zip(self.model.param_names, th):
                    rows.append({"i": float(i_val), "g": float(g), "param": p_name, "value": float(val)})
        return pd.DataFrame(rows)

    def export_grid(self, theta_flat: np.ndarray) -> pd.DataFrame:
        thetas = self._theta_flat_to_dict(theta_flat)
        rows = []
        for i_val in self.i_values:
            xs = self.x_dense[float(i_val)]
            for g in self.groups:
                th = thetas[float(g)]
                ys = self.model.f(xs, float(i_val), th)
                for x_val, y_val in zip(xs, ys):
                    rows.append({"i": float(i_val), "g": float(g), "x": float(x_val), "f_x": float(y_val)})
        return pd.DataFrame(rows)

    def export_fitted(self, theta_flat: np.ndarray) -> pd.DataFrame:
        thetas = self._theta_flat_to_dict(theta_flat)
        df = self.df.copy()
        y_hat_vals = []
        for _, row in df.iterrows():
            g = float(row["g"]); i_val = float(row["i"]); x = float(row["x"])
            th = thetas[g]
            y_hat_vals.append(float(self.model.f(np.array([x]), i_val, th)[0]))
        df["y_hat"] = np.array(y_hat_vals, dtype=float)
        df["source"] = np.where(df["y"].isna(), "imputed", "observed")
        return df[["g", "i", "x", "y", "y_hat", "source"]].copy()


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global constrained fitting with shared θ per g (function of i and x).")
    p.add_argument("--data", required=True, help="Path to data.csv (columns: g,i,x,y; optional: w).")
    p.add_argument("--func", required=True, help="Path to function.txt (first line: SymPy expression in i and x).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--grid-points", type=int, default=200, help="Baseline equal-spaced grid points per i (default: 200).")
    p.add_argument("--auto-direction", type=str, default="true", help="Auto infer monotonic direction s_i (true/false).")
    p.add_argument("--direction", type=str, choices=["inc", "dec"], default=None, help="Force monotonic direction per i (inc/dec).")
    p.add_argument("--order-mode", type=str, choices=["auto", "g-asc", "g-desc"], default="auto",
                   help="Cross-g ordering within each i: auto by median f, or by numeric g.")
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

    # --- multi-start: number of random restarts ---
    p.add_argument("--restarts", type=int, default=1,
                   help=("Number of random restarts to perform.  When greater than 1, the solver runs multiple "
                         "optimizations with different random seeds and returns the solution with the lowest "
                         "objective value.  Each restart increments the seed by 1 relative to the specified --seed. "
                         "Useful for non-convex problems to escape poor local minima.  Default: 1 (no restarts)."))

    # When enabled, automatically compute a per‑i scaling factor based on the
    # dynamic range of the observed y values for each i.  This scaling is
    # applied to the observation weights and the smoothness weights so that
    # each i contributes roughly equally to the overall objective, even when
    # the magnitudes of y differ substantially.  Without balancing, data sets
    # with larger y values can dominate the loss and lead to poor fits for
    # small‑magnitude series.  See README for details.
    p.add_argument("--balance-i", dest="balance_i", action="store_true", default=False,
                   help="Scale contributions across different i values based on the range of y."
                        " When set, each observation weight and smoothness weight is multiplied"
                        " by 1/(max(y_i) - min(y_i)) for its i.  Disabled by default.")
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
        balance_i=bool(getattr(args, "balance_i", False)),
    )

    # Handle multi-start restarts.  We perform `args.restarts` runs with successive seeds and
    # keep the solution with the lowest objective value (fit + penalties).  This is useful when the
    # objective is non-convex and the solver can get stuck in poor local minima.  Each restart
    # increments the random seed so that initial θ0 differs across runs.  If restarts == 1, we
    # simply perform a single run with the provided seed.
    num_restarts = max(1, int(getattr(args, "restarts", 1)))
    best_L = None
    best_theta_hat = None
    best_info = None
    best_fitter = None
    # We need to preserve the original seed; if None, we use None for the first run and 1,2,... for subsequent
    base_seed = cfg.seed
    for restart_idx in range(num_restarts):
        # Determine seed for this restart; if base_seed is provided, increment it; otherwise use restart index
        seed_k = (base_seed + restart_idx) if (base_seed is not None) else restart_idx
        # Create a fresh config for this restart with the updated seed
        cfg_k = SolverConfig(
            eps_mono=cfg.eps_mono,
            eps_ord=cfg.eps_ord,
            eps_den=cfg.eps_den,
            lambda_s=cfg.lambda_s,
            lambda_sim=cfg.lambda_sim,
            lower_bound=cfg.lower_bound,
            upper_bound=cfg.upper_bound,
            maxiter=cfg.maxiter,
            verbose=cfg.verbose,
            method_primary=cfg.method_primary,
            method_fallback=cfg.method_fallback,
            order_mode=cfg.order_mode,
            auto_direction=cfg.auto_direction,
            s_i_override=cfg.s_i_override,
            seed=seed_k,
            balance_i=getattr(cfg, "balance_i", False),
        )
        fitter_k = FitGlobal(df=df, model=model, grid_points=int(args.grid_points), config=cfg_k, outdir=outdir)
        theta_hat_k, info_k = fitter_k.solve()
        # Compute objective value for this solution (fit + smoothness + similarity).  The solver returns a flat
        # parameter vector; we can pass it directly to _objective_and_grad.
        L_k, _, diag_k = fitter_k._objective_and_grad(theta_hat_k)
        if (best_L is None) or (L_k < best_L):
            best_L = L_k
            best_theta_hat = theta_hat_k
            best_info = info_k
            best_fitter = fitter_k
    # After restarts, use the best solution for exports and checks
    theta_hat = best_theta_hat
    info = best_info
    fitter = best_fitter

    # Checks
    per_i_checks = fitter.compute_checks_per_i(theta_hat)
    checks = {
        "spec_version": "2025-08-27-shared-theta",
        "function": expr_str,
        "params": model.param_names,
        "per_i": per_i_checks,
        "overall": {
            "monotonicity_violation_max": float(np.max([c["monotonicity_violation_max"] for c in per_i_checks])) if per_i_checks else np.nan,
            "ordering_violation_max": float(np.max([c["ordering_violation_max"] for c in per_i_checks])) if per_i_checks else np.nan,
            "denominator_min_abs": float(np.min([c["denominator_min_abs"] for c in per_i_checks])) if per_i_checks else np.inf,
            "smooth_penalty_value_sum": float(np.sum([c["smooth_penalty_value"] for c in per_i_checks])) if per_i_checks else 0.0,
            "solver": info,
        },
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

    # Exports
    fitter.export_plots(theta_hat, outdir)
    params_df = fitter.export_params(theta_hat)
    grid_df = fitter.export_grid(theta_hat)
    fitted_df = fitter.export_fitted(theta_hat)

    params_df.to_csv(os.path.join(outdir, "params.csv"), index=False)
    grid_df.to_csv(os.path.join(outdir, "grid.csv"), index=False)
    fitted_df.to_csv(os.path.join(outdir, "fitted.csv"), index=False)
    with open(os.path.join(outdir, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    violated = (
        (checks["overall"]["monotonicity_violation_max"] > 1e-9) or
        (checks["overall"]["ordering_violation_max"] > 1e-9) or
        (np.isfinite(checks["overall"]["denominator_min_abs"]) and (checks["overall"]["denominator_min_abs"] < cfg.eps_den - 1e-12)) or
        (not info.get("success", False))
    )
    if violated:
        print("Constraint violation or solver failure detected. See checks.json for details.", file=sys.stderr)
        sys.exit(2)
    else:
        print("Fitting completed successfully (shared θ per g).")
        sys.exit(0)


if __name__ == "__main__":
    main()
