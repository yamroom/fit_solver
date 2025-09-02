#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opt1_twostage.py — IPOPT/CasADi implementation that mirrors fit_modified_final_rel.py I/O & behavior
====================================================================================================

- CLI interface, inputs, outputs aligned with fit_modified_final_rel.py
- One parameter vector θ_g per g, **shared across all i**
- Objective = Huber(relative residual) + λ_s * smoothness + λ_sim * similarity
- Hard constraints:
    * Monotonicity per (i, g): s_i * d/dx f(i,x;θ_g) >= eps_mono on a grid
    * Cross-g non-crossing per i: f_{g(j+1)}(x) - f_{g(j)}(x) >= eps_ord on a grid
    * Denominator safety: den(i,x;θ_g)^2 >= eps_den^2 on a grid (if the function is a fraction)
- Two-stage solve (both with IPOPT): Stage 1 objective-only init, Stage 2 with constraints
- Multi-start restarts; pick best feasible solution
- Outputs: params.csv, grid.csv, fitted.csv, checks.json, plots_i=*.png
- Exit code: 0 success, 2 failure (infeasible or solver failure)
"""

from __future__ import annotations
import os, sys, json, math, argparse, textwrap, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CasADi & SymPy ---
try:
    import casadi as ca
except Exception as e:
    raise RuntimeError("CasADi is required. Please install casadi and ensure IPOPT is available.") from e

import sympy as sp
from sympy import symbols

# ----------------------------- CLI -----------------------------

def str2bool(s: str) -> bool:
    return str(s).strip().lower() in {"1","true","t","yes","y","on"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Global constrained fitting (IPOPT/CasADi) with shared θ per g; mirror of fit_modified_final_rel.py"
    )
    p.add_argument("--data", required=True, help="Path to data.csv (columns: g,i,x,y; optional: w).")
    p.add_argument("--func", required=True, help="Path to function.txt (first line: SymPy expression in i and x).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--grid-points", type=int, default=200, help="Equal-spaced grid points per i (default: 200).")
    p.add_argument("--auto-direction", type=str, default="true", help="Auto infer monotonic direction s_i (true/false).")
    p.add_argument("--direction", type=str, choices=["inc","dec"], default=None, help="Force monotonic direction per i (inc/dec).")
    p.add_argument("--order-mode", type=str, choices=["auto", "g-asc", "g-desc"], default="auto",
                   help="Cross-g ordering within each i: auto by median f, or by numeric g.")
    p.add_argument("--lambda-s", type=float, default=1e-4, dest="lambda_s", help="Smoothness penalty weight.")
    p.add_argument("--lambda-sim", type=float, default=1e-3, dest="lambda_sim", help="Cross-g similarity penalty weight.")
    p.add_argument("--eps-mono", type=float, default=1e-6, dest="eps_mono", help="Monotonicity margin epsilon.")
    p.add_argument("--eps-ord", type=float, default=1e-6, dest="eps_ord", help="Ordering margin epsilon.")
    p.add_argument("--eps-den", type=float, default=1e-8, dest="eps_den", help="Denominator lower-bound epsilon.")
    p.add_argument("--lower", type=float, default=-1e6, help="Lower bound for all parameters.")
    p.add_argument("--upper", type=float, default= 1e6, help="Upper bound for all parameters.")
    p.add_argument("--maxiter", type=int, default=3000, help="Max iterations for IPOPT (Stage 1 & 2).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for initialization & restarts.")
    p.add_argument("--verbose", action="store_true", help="Verbose solver output.")
    p.add_argument("--restarts", type=int, default=3, help="Number of multi-start restarts (>=1).")
    p.add_argument("--balance-i", dest="balance_i", action="store_true", help="Per‑i balancing of weights/smoothness.")
    p.add_argument("--relative-error", action="store_true", help="Use relative error residuals (default ON here).")
    p.add_argument("--rel-eps", type=float, default=1e-5, dest="rel_eps", help="Small epsilon for relative residuals.")
    # IPOPT options (optional)
    p.add_argument("--ipopt.tol", type=float, default=1e-6, dest="ipopt_tol", help="IPOPT tol.")
    p.add_argument("--ipopt.acceptable_tol", type=float, default=1e-4, dest="ipopt_acc_tol", help="IPOPT acceptable_tol.")
    p.add_argument("--ipopt.hessian_approximation", type=str, default="limited-memory", dest="ipopt_hess",
                   choices=["exact","limited-memory"], help="Hessian approximation mode.")
    p.add_argument("--ipopt.linear_solver", type=str, default="mumps", dest="ipopt_linear_solver",
                   help="Underlying linear solver (e.g., mumps, ma57 if available).")

    args = p.parse_args()

    # Defaults aligned to your choices
    if not args.relative_error:
        # In this script we default to ON per your preference; still honor explicit flag states.
        args.relative_error = True
    return args

# ----------------------------- SymPy → CasADi Model -----------------------------

@dataclass
class CasadiModel:
    expr_str: str
    expr: sp.Expr
    x_sym: sp.Symbol
    i_sym: sp.Symbol
    param_syms: List[sp.Symbol]
    param_names: List[str]
    f_fun: ca.Function          # (x,i,theta) -> f
    dfdx_fun: ca.Function       # (x,i,theta) -> df/dx
    d2fdx2_fun: ca.Function     # (x,i,theta) -> d2f/dx2
    df_dp_fun: ca.Function      # (x,i,theta) -> (P,N)
    ddfdx_dp_fun: ca.Function   # (x,i,theta) -> (P,N)
    d2fdx2_dp_fun: ca.Function  # (x,i,theta) -> (P,N)
    denom_fun: Optional[ca.Function]        # (x,i,theta) -> den
    dden_dp_fun: Optional[ca.Function]      # (x,i,theta) -> (P,N)

    @staticmethod
    def parse(expr_str: str) -> "CasadiModel":
        # Build SymPy expression, allow ln to mean log
        i_sym, x_sym = symbols("i x", real=True)
        expr = sp.sympify(expr_str, locals={"i": i_sym, "x": x_sym, "ln": sp.log, "exp": sp.exp})

        # Collect parameter symbols by name (deterministic order)
        param_syms = sorted([s for s in expr.free_symbols if s not in {i_sym, x_sym}], key=lambda s: s.name)
        param_names = [s.name for s in param_syms]

        # Build CasADi symbols
        x = ca.SX.sym("x")
        i = ca.SX.sym("i")
        P = len(param_syms)
        theta = ca.SX.sym("theta", P)

        # SymPy lambdify to CasADi using a mapping table
        modules = [{
            "sin": ca.sin, "cos": ca.cos, "tan": ca.tan,
            "asin": ca.asin, "acos": ca.acos, "atan": ca.atan,
            "sinh": ca.sinh, "cosh": ca.cosh, "tanh": ca.tanh,
            "exp": ca.exp, "log": ca.log, "ln": ca.log, "sqrt": ca.sqrt,
            "Abs": ca.fabs, "sign": ca.sign,
            "Min": lambda a,b: ca.if_else(a<b, a, b),
            "Max": lambda a,b: ca.if_else(a>b, a, b),
            "Pow": ca.power,
        }]

        def _lamb(expr_):
            # arguments order: (x, i, *params)
            args = (x, i, *[ca.SX.sym(p.name) for p in param_syms])
            f = sp.lambdify((x_sym, i_sym, *param_syms), expr_, modules=modules)
            # Call with CasADi symbols to get a CasADi SX expression
            val = f(args[0], args[1], *args[2:])
            return ca.Function("F", [x, i, theta], [ca.substitute(val, ca.vertcat(*args[2:]), theta)], ["x","i","theta"], ["out"])

        # Base function
        f_fun = _lamb(expr)

        # Use CasADi AD for derivatives w.r.t x and params
        # df/dx
        dfdx_expr = ca.jacobian(f_fun(x,i,theta), x)
        dfdx_fun = ca.Function("dfdx", [x,i,theta], [dfdx_expr])
        # d2f/dx2
        d2fdx2_expr = ca.jacobian(dfdx_expr, x)
        d2fdx2_fun = ca.Function("d2fdx2", [x,i,theta], [d2fdx2_expr])
        # df/dp (P x 1) at scalar x; we will vectorize later
        df_dp_expr = ca.jacobian(f_fun(x,i,theta), theta).T  # (P,1)
        df_dp_fun = ca.Function("df_dp", [x,i,theta], [df_dp_expr])
        # d(dfdx)/dp
        ddfdx_dp_expr = ca.jacobian(dfdx_expr, theta).T  # (P,1)
        ddfdx_dp_fun = ca.Function("ddfdx_dp", [x,i,theta], [ddfdx_dp_expr])
        # d(d2fdx2)/dp
        d2fdx2_dp_expr = ca.jacobian(d2fdx2_expr, theta).T
        d2fdx2_dp_fun = ca.Function("d2fdx2_dp", [x,i,theta], [d2fdx2_dp_expr])

        # Denominator detection via SymPy
        num, den = sp.fraction(sp.together(expr))
        denom_fun = None
        dden_dp_fun = None
        if not (den == 1):
            den_fun = _lamb(sp.simplify(den))
            denom_fun = ca.Function("den", [x,i,theta], [den_fun(x,i,theta)])
            dden_dp_expr = ca.jacobian(denom_fun(x,i,theta), theta).T
            dden_dp_fun = ca.Function("dden_dp", [x,i,theta], [dden_dp_expr])

        return CasadiModel(
            expr_str=str(expr_str),
            expr=expr,
            x_sym=x_sym,
            i_sym=i_sym,
            param_syms=param_syms,
            param_names=param_names,
            f_fun=f_fun,
            dfdx_fun=dfdx_fun,
            d2fdx2_fun=d2fdx2_fun,
            df_dp_fun=df_dp_fun,
            ddfdx_dp_fun=ddfdx_dp_fun,
            d2fdx2_dp_fun=d2fdx2_dp_fun,
            denom_fun=denom_fun,
            dden_dp_fun=dden_dp_fun,
        )

    # Vectorized helpers (numpy arrays in, numpy arrays out), using mapaccum
    def f(self, x_vec: np.ndarray, i_val: float, theta_vec: np.ndarray) -> np.ndarray:
        xx = ca.DM(x_vec.reshape(-1,1))
        ii = float(i_val)
        th = ca.DM(theta_vec)
        out = [float(self.f_fun(x=float(xv), i=ii, theta=th)["out"]) for xv in np.ravel(x_vec)]
        return np.asarray(out, dtype=float)

    def dfdx(self, x_vec: np.ndarray, i_val: float, theta_vec: np.ndarray) -> np.ndarray:
        ii = float(i_val); th = ca.DM(theta_vec)
        return np.asarray([float(self.dfdx_fun(x=float(xv), i=ii, theta=th)) for xv in np.ravel(x_vec)], dtype=float)

    def d2fdx2(self, x_vec: np.ndarray, i_val: float, theta_vec: np.ndarray) -> np.ndarray:
        ii = float(i_val); th = ca.DM(theta_vec)
        return np.asarray([float(self.d2fdx2_fun(x=float(xv), i=ii, theta=th)) for xv in np.ravel(x_vec)], dtype=float)

    def denom(self, x_vec: np.ndarray, i_val: float, theta_vec: np.ndarray) -> Optional[np.ndarray]:
        if self.denom_fun is None:
            return None
        ii = float(i_val); th = ca.DM(theta_vec)
        return np.asarray([float(self.denom_fun(x=float(xv), i=ii, theta=th)) for xv in np.ravel(x_vec)], dtype=float)

# ----------------------------- Utilities -----------------------------

def unique_sorted(a: np.ndarray) -> np.ndarray:
    return np.unique(a.astype(float))

def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 0: return np.array([])
    if n == 1: return np.array([1.0])
    w = np.zeros(n, dtype=float)
    w[0]  = (x[1] - x[0]) / 2.0
    w[-1] = (x[-1] - x[-2]) / 2.0
    if n > 2:
        w[1:-1] = (x[2:] - x[:-2]) / 2.0
    return w

def huber_value(r: ca.SX, delta: float) -> ca.SX:
    a = ca.fabs(r)
    return ca.if_else(a <= delta, 0.5*r*r, delta*(a - 0.5*delta))

# ----------------------------- Config -----------------------------

@dataclass
class SolverConfig:
    lambda_s: float
    lambda_sim: float
    eps_mono: float
    eps_ord: float
    eps_den: float
    lower: float
    upper: float
    maxiter: int
    seed: int
    verbose: bool
    restarts: int
    balance_i: bool
    relative_error: bool
    rel_eps: float
    ipopt_tol: float
    ipopt_acc_tol: float
    ipopt_hess: str
    ipopt_linear_solver: str

# ----------------------------- Fitter -----------------------------

class FitGlobalCasadi:
    def __init__(
        self,
        df: pd.DataFrame,
        model: CasadiModel,
        grid_points: int,
        config: SolverConfig,
        outdir: str,
        direction: Optional[str],
        order_mode: str,
    ):
        self.df = df.copy()
        self.model = model
        self.grid_points = int(grid_points)
        self.config = config
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        # sets
        self.i_values = unique_sorted(self.df["i"].to_numpy())
        self.groups = unique_sorted(self.df["g"].to_numpy())
        self.n_i = len(self.i_values)
        self.n_g = len(self.groups)
        self.P = len(self.model.param_syms)
        if self.P == 0:
            raise ValueError("No free parameters in function; nothing to estimate.")

        # observations with weights
        df2 = self.df.copy()
        if "w" not in df2.columns:
            df2["w"] = 1.0
        # relative error scaling; applied at residual construction time
        self.obs_by_gi: Dict[Tuple[float,float], pd.DataFrame] = {}
        for g in self.groups:
            for i_val in self.i_values:
                sub = df2[(df2["g"]==g) & (df2["i"]==i_val) & (~df2["y"].isna())][["x","y","w"]].copy()
                sub.sort_values("x", inplace=True)
                self.obs_by_gi[(float(g), float(i_val))] = sub.reset_index(drop=True)

        # grids per i
        self.x_dense: Dict[float, np.ndarray] = {}
        self.w_smooth: Dict[float, np.ndarray] = {}
        for i_val in self.i_values:
            obs_i = df2[df2["i"]==i_val]
            x_min = float(np.nanmin(obs_i["x"])) if obs_i.shape[0] else 0.0
            x_max = float(np.nanmax(obs_i["x"])) if obs_i.shape[0] else 1.0
            xs = np.linspace(x_min, x_max, self.grid_points)
            self.x_dense[float(i_val)] = xs
            self.w_smooth[float(i_val)] = trapezoid_weights(xs)

        # s_i direction
        if direction is not None:
            s_val = 1.0 if direction == "inc" else -1.0
            self.s_i = {float(i_val): s_val for i_val in self.i_values}
        else:
            self.s_i = self._infer_direction()

        # order indices per i
        self.order_mode = order_mode
        self.order_by_i = {}  # i -> list of g sorted
        # initial provisional ordering by numeric g; may be overridden by auto later
        for i_val in self.i_values:
            self.order_by_i[float(i_val)] = list(self.groups.astype(float))

        # bounds & initial theta (flat)
        self.bounds = (np.full(self.n_g*self.P, config.lower, dtype=float),
                       np.full(self.n_g*self.P, config.upper, dtype=float))
        self.theta0 = self._init_theta()

    def _infer_direction(self) -> Dict[float,float]:
        # sign of correlation between x and y across all g for each i
        out = {}
        for i_val in self.i_values:
            sub = self.df[(self.df["i"]==i_val) & (~self.df["y"].isna())][["x","y"]]
            if sub.shape[0] < 2:
                out[float(i_val)] = 1.0
                continue
            x = sub["x"].to_numpy(); y = sub["y"].to_numpy()
            # simple slope sign via linear fit
            try:
                A = np.vstack([x, np.ones_like(x)]).T
                m, b = np.linalg.lstsq(A, y, rcond=None)[0]
                out[float(i_val)] = 1.0 if m >= 0 else -1.0
            except Exception:
                out[float(i_val)] = 1.0
        return out

    def _init_theta(self) -> np.ndarray:
        # simple robust init from small random values
        rng = np.random.default_rng(self.config.seed)
        scale = 0.1
        return rng.normal(loc=0.0, scale=scale, size=self.n_g*self.P)

    # ----- helper to split/join theta -----
    def _theta_flat_to_dict(self, theta_flat: np.ndarray) -> Dict[float, np.ndarray]:
        out: Dict[float, np.ndarray] = {}
        for idx, g in enumerate(self.groups):
            out[float(g)] = theta_flat[idx*self.P:(idx+1)*self.P]
        return out

    def _theta_dict_to_flat(self, thetas: Dict[float, np.ndarray]) -> np.ndarray:
        return np.concatenate([thetas[float(g)] for g in self.groups], axis=0)

    # ----- objective (CasADi graph) -----
    def _objective_expr(self, theta: ca.SX, order_ready: bool=False) -> ca.SX:
        cfg = self.config
        P = self.P
        obj = ca.SX(0)

        # Optionally set ordering using median f from current theta for auto mode
        if (self.order_mode == "auto") and (not order_ready):
            self._update_order_by_median(theta)

        # Fit term
        for idx_g, g in enumerate(self.groups):
            th = theta[idx_g*P:(idx_g+1)*P]
            for i_val in self.i_values:
                obs = self.obs_by_gi[(float(g), float(i_val))]
                if obs.shape[0] == 0:
                    continue
                xs = obs["x"].to_numpy()
                ys = obs["y"].to_numpy().astype(float)
                ws = obs["w"].to_numpy().astype(float)
                for k in range(xs.size):
                    xk = float(xs[k]); yk = float(ys[k]); wk = float(ws[k])
                    pred = self.model.f_fun(x=xk, i=float(i_val), theta=th)["out"]
                    r = pred - yk
                    if cfg.relative_error:
                        r = r / (abs(yk) + cfg.rel_eps)
                    # balance-i scales both fitting and smoothness inside each i
                    if cfg.balance_i:
                        # scale by dynamic range of y in this i
                        yi = self.df[(self.df["i"]==i_val) & (~self.df["y"].isna())]["y"].to_numpy()
                        rng = float(np.max(yi) - np.min(yi) + 1e-12)
                        wk = wk / rng
                    obj = obj + huber_value(ca.sqrt(wk) * r, 1.0)  # δ=1.0

        # Smoothness term
        if cfg.lambda_s and cfg.lambda_s > 0:
            for idx_g, g in enumerate(self.groups):
                th = theta[idx_g*P:(idx_g+1)*self.P]
                for i_val in self.i_values:
                    xs = self.x_dense[float(i_val)]
                    ws = self.w_smooth[float(i_val)]
                    for k in range(xs.size):
                        xk = float(xs[k]); wk = float(ws[k])
                        d2 = self.model.d2fdx2_fun(x=xk, i=float(i_val), theta=th)
                        if cfg.balance_i:
                            yi = self.df[(self.df["i"]==i_val) & (~self.df["y"].isna())]["y"].to_numpy()
                            rng = float(np.max(yi) - np.min(yi) + 1e-12)
                            wk = wk / rng
                        obj = obj + cfg.lambda_s * wk * (d2**2)

        # Similarity term
        if cfg.lambda_sim and cfg.lambda_sim > 0 and self.n_g > 1:
            # theta_bar
            theta_bar = ca.SX.zeros(self.P)
            for idx_g in range(self.n_g):
                theta_bar = theta_bar + theta[idx_g*P:(idx_g+1)*P]
            theta_bar = theta_bar / float(self.n_g)
            for idx_g in range(self.n_g):
                diff = theta[idx_g*P:(idx_g+1)*P] - theta_bar
                obj = obj + cfg.lambda_sim * ca.dot(diff, diff)

        return obj

    def _update_order_by_median(self, theta: ca.SX) -> None:
        # compute median f across each i for each g and sort
        P = self.P
        order_by_i = {}
        for i_val in self.i_values:
            xs = self.x_dense[float(i_val)]
            meds = []
            for idx_g, g in enumerate(self.groups):
                th = theta[idx_g*P:(idx_g+1)*P]
                ys = [float(self.model.f_fun(x=float(xk), i=float(i_val), theta=th)["out"]) for xk in xs]
                meds.append((float(np.median(np.asarray(ys))), float(g)))
            meds.sort(key=lambda t: t[0])  # ascending by median
            order_by_i[float(i_val)] = [g for _,g in meds]
        self.order_by_i = order_by_i

    # ----- constraints (CasADi graph) -----
    def _constraints_expr(self, theta: ca.SX) -> ca.SX:
        cfg = self.config
        P = self.P
        cons = []

        # Monotonicity per (i,g)
        for i_val in self.i_values:
            s = float(self.s_i[float(i_val)])
            xs = self.x_dense[float(i_val)]
            for idx_g, g in enumerate(self.groups):
                th = theta[idx_g*P:(idx_g+1)*P]
                for xk in xs:
                    d1 = self.model.dfdx_fun(x=float(xk), i=float(i_val), theta=th)
                    cons.append(s * d1 - cfg.eps_mono)

        # Cross-g non-crossing per i (adjacent pairs under chosen order)
        for i_val in self.i_values:
            xs = self.x_dense[float(i_val)]
            # decide order list
            if self.order_mode == "g-asc":
                order = list(sorted(self.groups))
            elif self.order_mode == "g-desc":
                order = list(sorted(self.groups, reverse=True))
            else:
                order = self.order_by_i[float(i_val)]
            for j in range(len(order)-1):
                g_lo = float(order[j])
                g_hi = float(order[j+1])
                idx_lo = int(np.where(self.groups==g_lo)[0][0])
                idx_hi = int(np.where(self.groups==g_hi)[0][0])
                th_lo = theta[idx_lo*P:(idx_lo+1)*P]
                th_hi = theta[idx_hi*P:(idx_hi+1)*P]
                for xk in xs:
                    y_hi = self.model.f_fun(x=float(xk), i=float(i_val), theta=th_hi)["out"]
                    y_lo = self.model.f_fun(x=float(xk), i=float(i_val), theta=th_lo)["out"]
                    cons.append(y_hi - y_lo - cfg.eps_ord)

        # Denominator safety (if denominator exists)
        if self.model.denom_fun is not None:
            for i_val in self.i_values:
                xs = self.x_dense[float(i_val)]
                for idx_g, g in enumerate(self.groups):
                    th = theta[idx_g*P:(idx_g+1)*P]
                    for xk in xs:
                        den = self.model.denom_fun(x=float(xk), i=float(i_val), theta=th)
                        cons.append((den*den) - (cfg.eps_den**2))

        if len(cons) == 0:
            return ca.SX.zeros(1)
        return ca.vertcat(*cons)

    # ----- solve paths -----
    def solve(self) -> Tuple[np.ndarray, Dict]:
        # decision variable: theta_flat
        theta = ca.SX.sym("theta", self.n_g*self.P)

        # Stage 1: objective only, to get init
        nlp1 = {"x": theta, "f": self._objective_expr(theta, order_ready=False)}
        opts1 = {
            "ipopt": {
                "tol": self.config.ipopt_tol,
                "acceptable_tol": self.config.ipopt_acc_tol,
                "hessian_approximation": self.config.ipopt_hess,
                "linear_solver": self.config.ipopt_linear_solver,
                "max_iter": int(self.config.maxiter),
                "print_level": 5 if self.config.verbose else 0,
            },
            "print_time": False,
        }
        solver1 = ca.nlpsol("solver1","ipopt", nlp1, opts1)

        # Stage 2: with constraints
        g_expr = self._constraints_expr(theta)
        nlp2 = {"x": theta, "f": self._objective_expr(theta, order_ready=True), "g": g_expr}
        opts2 = {
            "ipopt": {
                "tol": self.config.ipopt_tol,
                "acceptable_tol": self.config.ipopt_acc_tol,
                "hessian_approximation": self.config.ipopt_hess,
                "linear_solver": self.config.ipopt_linear_solver,
                "max_iter": int(self.config.maxiter),
                "print_level": 5 if self.config.verbose else 0,
            },
            "print_time": False,
        }
        solver2 = ca.nlpsol("solver2","ipopt", nlp2, opts2)

        # Prepare bounds
        lbx = ca.DM(np.full(self.n_g*self.P, self.config.lower, dtype=float))
        ubx = ca.DM(np.full(self.n_g*self.P, self.config.upper, dtype=float))
        lbg = ca.DM.zeros(int(g_expr.shape[0]))  # all >= 0
        ubg = ca.DM.inf(int(g_expr.shape[0]))

        # Multi-start
        rng = np.random.default_rng(self.config.seed)
        best = None
        trials = []
        for r in range(max(1, self.config.restarts)):
            if r == 0:
                x0 = ca.DM(self.theta0)
            else:
                perturb = rng.normal(0.0, 0.1, size=self.n_g*self.P)
                x0 = ca.DM(self.theta0 + perturb)

            # Stage 1
            try:
                sol1 = solver1(x0=x0, lbx=lbx, ubx=ubx)
                x1 = sol1["x"]
            except Exception as e:
                x1 = x0  # fallback

            # If auto order, update order_by_i using x1
            self._update_order_by_median(x1)

            # Stage 2
            try:
                sol2 = solver2(x0=x1, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                x2 = np.array(sol2["x"]).reshape(-1)
                f2 = float(sol2["f"])
                lam_g = np.array(sol2["lam_g"]).reshape(-1) if "lam_g" in sol2 else None
                # feasibility check: evaluate constraints
                G = self._constraints_expr(ca.DM(x2))
                G_num = np.array(ca.Function("Gnum", [ca.SX.sym("z", self.n_g*self.P)], [G])(ca.DM(x2))).reshape(-1)
                feas_violation = float(np.max(np.maximum(0.0, -G_num))) if G_num.size else 0.0
                stat = {"ok": True, "obj": f2, "feas_max_violation": feas_violation}
            except Exception as e:
                x2 = np.array(x1).reshape(-1)
                f2 = math.inf
                stat = {"ok": False, "error": str(e), "obj": f2, "feas_max_violation": math.inf}

            trials.append({"x": x2, "f": f2, "stat": stat})

            # choose best feasible (feas_max_violation <= 1e-8)
            if stat["ok"] and stat["feas_max_violation"] <= 1e-8:
                if (best is None) or (f2 < best["f"]):
                    best = {"x": x2, "f": f2, "stat": stat}

        # if no feasible, choose least infeasible (min feas violation, then min obj)
        if best is None:
            trials.sort(key=lambda d: (d["stat"]["feas_max_violation"], d["f"]))
            best = trials[0]

        return best["x"], {"trials": trials, "chosen": best}

    # ----- Exports -----
    def export_params(self, theta_flat: np.ndarray) -> pd.DataFrame:
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
        rows = []
        for (g,i_val), obs in self.obs_by_gi.items():
            th = theta_flat[int(np.where(self.groups==g)[0][0])*self.P : int(np.where(self.groups==g)[0][0]+1)*self.P]
            xs = obs["x"].to_numpy()
            ys = self.model.f(xs, float(i_val), th)
            for x_val, y_obs, y_hat in zip(xs, obs["y"].to_numpy(), ys):
                rows.append({"g": float(g), "i": float(i_val), "x": float(x_val), "y": float(y_obs), "y_hat": float(y_hat)})
        return pd.DataFrame(rows)

    def export_plots(self, theta_flat: np.ndarray) -> None:
        thetas = self._theta_flat_to_dict(theta_flat)
        for i_val in self.i_values:
            xs = self.x_dense[float(i_val)]
            plt.figure()
            for g in self.groups:
                th = thetas[float(g)]
                ys = self.model.f(xs, float(i_val), th)
                plt.plot(xs, ys, label=f"g={g}")
            df_i = self.df[self.df["i"]==i_val]
            for g in self.groups:
                sub = df_i[(df_i["g"] == g) & (~df_i["y"].isna())]
                if sub.shape[0] > 0:
                    plt.scatter(sub["x"], sub["y"], s=18, alpha=0.9, label=f"data g={g}")
            plt.title(f"i = {i_val} — fitted f(x) by g (shared θ, IPOPT)")
            plt.xlabel("x"); plt.ylabel("f(x) / y")
            plt.legend(loc="best", fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, f"plots_i={i_val}.png"), dpi=160)
            plt.close()

# ----------------------------- Main -----------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    required_cols = {"g","i","x","y"}
    if not required_cols.issubset(set(df.columns)):
        missing = sorted(list(required_cols - set(df.columns)))
        raise ValueError(f"data.csv missing columns: {missing}")
    # normalize types
    for col in ["g","i","x","y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "w" in df.columns:
        df["w"] = pd.to_numeric(df["w"], errors="coerce").fillna(1.0)
    else:
        df["w"] = 1.0

    # Parse function
    with open(args.func, "r", encoding="utf-8") as f:
        expr_line = f.readline().strip()
        if not expr_line:
            raise ValueError("function.txt first line is empty.")
    model = CasadiModel.parse(expr_line)

    # Build config
    cfg = SolverConfig(
        lambda_s=float(args.lambda_s),
        lambda_sim=float(args.lambda_sim),
        eps_mono=float(args.eps_mono),
        eps_ord=float(args.eps_ord),
        eps_den=float(args.eps_den),
        lower=float(args.lower),
        upper=float(args.upper),
        maxiter=int(args.maxiter),
        seed=int(args.seed),
        verbose=bool(args.verbose),
        restarts=int(max(1, args.restarts)),
        balance_i=bool(args.balance_i),
        relative_error=bool(args.relative_error),
        rel_eps=float(args.rel_eps),
        ipopt_tol=float(args.ipopt_tol),
        ipopt_acc_tol=float(args.ipopt_acc_tol),
        ipopt_hess=str(args.ipopt_hess),
        ipopt_linear_solver=str(args.ipopt_linear_solver),
    )

    # Determine direction flag
    direction = None
    if str2bool(args.auto_direction):
        direction = None  # auto
    else:
        direction = args.direction  # must be provided if auto false

    # Build fitter and solve
    fitter = FitGlobalCasadi(
        df=df,
        model=model,
        grid_points=int(args.grid_points),
        config=cfg,
        outdir=args.outdir,
        direction=direction,
        order_mode=args.order_mode,
    )

    theta_hat, info = fitter.solve()

    # Export
    params_df = fitter.export_params(theta_hat)
    grid_df   = fitter.export_grid(theta_hat)
    fitted_df = fitter.export_fitted(theta_hat)
    params_df.to_csv(os.path.join(args.outdir, "params.csv"), index=False)
    grid_df.to_csv(os.path.join(args.outdir, "grid.csv"), index=False)
    fitted_df.to_csv(os.path.join(args.outdir, "fitted.csv"), index=False)

    # Checks & plots
    checks = {
        "ipopt": {
            "tol": cfg.ipopt_tol,
            "acceptable_tol": cfg.ipopt_acc_tol,
            "hessian_approximation": cfg.ipopt_hess,
            "linear_solver": cfg.ipopt_linear_solver,
            "max_iter": cfg.maxiter,
        },
        "relative_error": cfg.relative_error,
        "rel_eps": cfg.rel_eps,
        "lambda_s": cfg.lambda_s,
        "lambda_sim": cfg.lambda_sim,
        "eps_mono": cfg.eps_mono,
        "eps_ord": cfg.eps_ord,
        "eps_den": cfg.eps_den,
        "trials": info.get("trials", []),
        "chosen_stat": info.get("chosen", {}).get("stat", {}),
    }
    with open(os.path.join(args.outdir, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    # Plots
    try:
        fitter.export_plots(theta_hat)
    except Exception as e:
        # plotting failure shouldn't crash pipeline
        pass

    # Exit code: 0 on success & feasibility, else 2
    feas = checks["chosen_stat"].get("feas_max_violation", 0.0)
    ok   = checks["chosen_stat"].get("ok", False)
    if ok and (feas is not None) and float(feas) <= 1e-8:
        sys.exit(0)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()
