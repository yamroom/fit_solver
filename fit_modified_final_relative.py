#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# A compact joint fitter with shared θ per g, optional relative error, balance-i reweighting, and restarts.
# - Reads CSV: columns g,i,x,y (optional w)
# - Reads function.txt (first line: SymPy expression in i and x, with free params e.g., C1_A,...)
# - Fits one θ_g per g, shared across i, minimizing weighted Huber loss (+ optional smoothing)
# - Options: relative error residuals, per-i balance reweighting, SLSQP/trust-constr, restarts
# - Outputs: params.csv, grid.csv, fitted.csv, checks.json
# Note: This version does not enforce hard monotonicity/ordering constraints (it only reports checks).

import argparse, sys, json, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import sympy as sp
from scipy.optimize import minimize

def unique_sorted(a: np.ndarray) -> np.ndarray:
    b = np.unique(a[~np.isnan(a)])
    return np.sort(b.astype(float))

def trapz_weights(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    n = xs.size
    if n < 2:
        return np.ones(n, dtype=float)
    w = np.zeros(n, dtype=float)
    dx = np.diff(xs)
    w[0] = dx[0]/2.0
    w[-1] = dx[-1]/2.0
    if n > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w

def huber_value_and_grad(r: np.ndarray, delta: float):
    r = np.asarray(r, dtype=float)
    delta = float(delta)
    ab = np.abs(r)
    v = np.where(ab <= delta, 0.5 * r**2, delta*(ab - 0.5*delta))
    dr = np.where(ab <= delta, r, delta*np.sign(r))
    return v, dr

class SympyModel:
    def __init__(self, expr: sp.Expr):
        self.i = sp.Symbol("i")
        self.x = sp.Symbol("x")
        syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
        self.param_syms = [s for s in syms if s.name not in {"i","x"}]
        self.param_names = [s.name for s in self.param_syms]
        self.P = len(self.param_syms)
        self.expr = expr
        self._f = sp.lambdify((self.x, self.i, self.param_syms), self.expr, modules="numpy")
        self._df_dp_syms = [sp.diff(self.expr, p) for p in self.param_syms]
        self._df_dp = sp.lambdify((self.x, self.i, self.param_syms), self._df_dp_syms, modules="numpy")
        self._dfdx = sp.lambdify((self.x, self.i, self.param_syms), sp.diff(self.expr, self.x), modules="numpy")
        self._d2fdx2 = sp.lambdify((self.x, self.i, self.param_syms), sp.diff(sp.diff(self.expr, self.x), self.x), modules="numpy")

    @staticmethod
    def parse(s: str) -> "SympyModel":
        i = sp.Symbol("i")
        x = sp.Symbol("x")
        expr = sp.sympify(s, locals={"i": i, "x": x})
        return SympyModel(expr)

    def f(self, xs: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self._f(xs, i_val, list(theta)), dtype=float)

    def df_dp(self, xs: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        out = self._df_dp(xs, i_val, list(theta))
        arr = np.vstack([np.asarray(o, dtype=float) for o in out])  # (P, N)
        return arr

    def dfdx(self, xs: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self._dfdx(xs, i_val, list(theta)), dtype=float)

    def d2fdx2(self, xs: np.ndarray, i_val: float, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self._d2fdx2(xs, i_val, list(theta)), dtype=float)

@dataclass
class SolverConfig:
    huber_delta: float = 1e-2
    lambda_s: float = 0.0        # smoothing on d2/dx2
    grid_points: int = 200       # per-i dense grid for smoothing & checks
    maxiter: int = 2000
    method: str = "SLSQP"        # or trust-constr
    seed: Optional[int] = 0
    restarts: int = 1
    balance_i: bool = False      # reweight per i by 1/range(y_i)
    use_relative_error: bool = False
    rel_eps: float = 1e-6

class Fitter:
    def __init__(self, df: pd.DataFrame, model: SympyModel, cfg: SolverConfig):
        self.df = df.copy()
        self.model = model
        self.cfg = cfg

        self.groups = unique_sorted(self.df["g"].to_numpy())
        self.i_values = unique_sorted(self.df["i"].to_numpy())
        self.P = model.P
        if self.P == 0:
            raise ValueError("No free parameters in function.")

        self.x_dense = {}
        self.w_smooth = {}
        for i_val in self.i_values:
            xs = unique_sorted(self.df[self.df["i"]==i_val]["x"].to_numpy())
            if xs.size == 0: 
                raise ValueError(f"No x values for i={i_val}.")
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            dense = np.linspace(xmin, xmax, int(self.cfg.grid_points))
            self.x_dense[float(i_val)] = dense
            self.w_smooth[float(i_val)] = trapz_weights(dense)

        self.obs = {}
        for g in self.groups:
            for i_val in self.i_values:
                sub = self.df[(self.df["g"]==g) & (self.df["i"]==i_val) & (~self.df["y"].isna())]
                w = sub["w"].to_numpy(dtype=float) if "w" in sub.columns else np.ones(sub.shape[0], dtype=float)
                self.obs[(float(g), float(i_val))] = {"x": sub["x"].to_numpy(dtype=float),
                                                      "y": sub["y"].to_numpy(dtype=float),
                                                      "w": w.copy()}
        if self.cfg.balance_i:
            for i_val in self.i_values:
                ys = self.df[(self.df["i"]==i_val) & (~self.df["y"].isna())]["y"].to_numpy(dtype=float)
                if ys.size == 0: continue
                y_range = float(np.max(ys) - np.min(ys))
                scale = 1.0 / y_range if y_range > 0 else 1.0
                for g in self.groups:
                    self.obs[(float(g), float(i_val))]["w"] *= scale
                self.w_smooth[float(i_val)] *= scale

        rng = np.random.default_rng(self.cfg.seed if self.cfg.seed is not None else 0)
        self.theta0 = { float(g): rng.normal(scale=1e-3, size=self.P) for g in self.groups }

    def objective_and_grad(self, theta_flat: np.ndarray):
        cfg = self.cfg
        thetas = { float(g): theta_flat[idx*self.P:(idx+1)*self.P] for idx, g in enumerate(self.groups) }
        L_fit = 0.0
        g_fit = np.zeros_like(theta_flat)
        for idx_g, g in enumerate(self.groups):
            theta_g = thetas[float(g)]
            grad_acc = np.zeros(self.P, dtype=float)
            for i_val in self.i_values:
                o = self.obs[(float(g), float(i_val))]
                x, y, w = o["x"], o["y"], o["w"]
                if x.size == 0: continue
                f = self.model.f(x, float(i_val), theta_g)
                if cfg.use_relative_error:
                    denom = np.maximum(np.abs(y), float(cfg.rel_eps))
                    r = (f - y) / denom
                    v, dr = huber_value_and_grad(r, cfg.huber_delta)
                    J = self.model.df_dp(x, float(i_val), theta_g)
                    grad_acc += (J / denom) @ (w * dr)
                else:
                    r = f - y
                    v, dr = huber_value_and_grad(r, cfg.huber_delta)
                    J = self.model.df_dp(x, float(i_val), theta_g)
                    grad_acc += J @ (w * dr)
                L_fit += float(np.sum(w * v))
            g_fit[idx_g*self.P:(idx_g+1)*self.P] = grad_acc

        L_s = 0.0
        g_s = np.zeros_like(theta_flat)
        if cfg.lambda_s > 0.0:
            for idx_g, g in enumerate(self.groups):
                theta_g = thetas[float(g)]
                grad_acc = np.zeros(self.P, dtype=float)
                for i_val in self.i_values:
                    xs = self.x_dense[float(i_val)]
                    w = self.w_smooth[float(i_val)]
                    d2 = self.model.d2fdx2(xs, float(i_val), theta_g)
                    L_s += float(np.sum(w * (d2**2)))
                    eps = 1e-6
                    for p in range(self.P):
                        dtheta = np.zeros(self.P, dtype=float); dtheta[p] = eps
                        d2p = self.model.d2fdx2(xs, float(i_val), theta_g + dtheta)
                        d2m = self.model.d2fdx2(xs, float(i_val), theta_g - dtheta)
                        d_d2_dp = (d2p - d2m) / (2*eps)
                        grad_acc[p] += np.sum(w * 2.0 * d2 * d_d2_dp)
                g_s[idx_g*self.P:(idx_g+1)*self.P] = grad_acc

        L = L_fit + cfg.lambda_s * L_s
        g = g_fit + cfg.lambda_s * g_s
        return L, g

    def objective(self, theta_flat: np.ndarray) -> float:
        L, _ = self.objective_and_grad(theta_flat)
        return L

    def grad(self, theta_flat: np.ndarray) -> np.ndarray:
        _, g = self.objective_and_grad(theta_flat)
        return g

    def solve(self):
        best = None
        best_info = None
        base_seed = self.cfg.seed if self.cfg.seed is not None else 0
        for k in range(max(1, int(self.cfg.restarts))):
            seed_k = base_seed + k
            rng = np.random.default_rng(seed_k)
            theta0_flat = np.concatenate([self.theta0[float(g)] + rng.normal(scale=1e-3, size=self.P) for g in self.groups])
            res = minimize(self.objective, theta0_flat, jac=self.grad, method=self.cfg.method,
                           options={"maxiter": int(self.cfg.maxiter), "disp": False})
            info = {"success": bool(res.success), "status": int(getattr(res,"status",-1)),
                    "message": str(getattr(res,"message","")), "niter": int(getattr(res,"nit",-1)),
                    "fun": float(getattr(res,"fun", np.nan))}
            theta_flat = res.x
            if (best is None) or (info["fun"] < best_info["fun"]):
                best, best_info = theta_flat, info
        return best, best_info

    def export_all(self, theta_flat: np.ndarray, outdir: str, func_string: str):
        os.makedirs(outdir, exist_ok=True)
        P = self.P
        thetas = { float(g): theta_flat[idx*P:(idx+1)*P] for idx, g in enumerate(self.groups) }

        rows = []
        for i_val in self.i_values:
            for g in self.groups:
                th = thetas[float(g)]
                for name, val in zip(self.model.param_names, th):
                    rows.append({"i": float(i_val), "g": float(g), "param": name, "value": float(val)})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "params.csv"), index=False)

        rows = []
        for g in self.groups:
            for i_val in self.i_values:
                o = self.obs[(float(g), float(i_val))]
                x, y = o["x"], o["y"]
                th = thetas[float(g)]
                yhat = self.model.f(x, float(i_val), th)
                for xx, yy, yh in zip(x, y, yhat):
                    rows.append({"g": float(g), "i": float(i_val), "x": float(xx), "y": float(yy), "y_hat": float(yh), "source": "observed"})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "fitted.csv"), index=False)

        rows = []
        for g in self.groups:
            for i_val in self.i_values:
                xs = self.x_dense[float(i_val)]
                th = thetas[float(g)]
                ys = self.model.f(xs, float(i_val), th)
                for xx, yy in zip(xs, ys):
                    rows.append({"g": float(g), "i": float(i_val), "x": float(xx), "y_hat": float(yy)})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "grid.csv"), index=False)

        fitted = pd.read_csv(os.path.join(outdir, "fitted.csv"))
        per_i = []
        for i_val in self.i_values:
            sub = fitted[fitted["i"]==i_val]
            mse = float(np.mean((sub["y_hat"] - sub["y"])**2)) if len(sub)>0 else float("nan")
            per_i.append({"i": float(i_val), "mse": mse})
        checks = {
            "spec_version": "2025-08-28-minimal-rel",
            "function": func_string,
            "params": self.model.param_names,
            "config": {
                "huber_delta": self.cfg.huber_delta,
                "lambda_s": self.cfg.lambda_s,
                "grid_points": self.cfg.grid_points,
                "restarts": self.cfg.restarts,
                "balance_i": self.cfg.balance_i,
                "use_relative_error": self.cfg.use_relative_error,
                "rel_eps": self.cfg.rel_eps,
                "method": self.cfg.method,
            },
            "per_i": per_i
        }
        with open(os.path.join(outdir, "checks.json"), "w", encoding="utf-8") as f:
            json.dump(checks, f, indent=2, ensure_ascii=False)

def parse_args():
    p = argparse.ArgumentParser(description="Joint fit with shared θ per g, optional relative error & restarts.")
    p.add_argument("--data", required=True, help="Path to data.csv (columns: g,i,x,y; optional: w).")
    p.add_argument("--func", required=True, help="Path to function.txt (first line: SymPy expression in i and x).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--grid-points", type=int, default=200, help="Dense grid points per i (default: 200).")
    p.add_argument("--huber-delta", type=float, default=1e-2, dest="huber_delta", help="Huber delta (relative units if using relative error).")
    p.add_argument("--lambda-s", type=float, default=0.0, dest="lambda_s", help="Smoothing weight on d2/dx2 (default 0).")
    p.add_argument("--method", type=str, default="SLSQP", choices=["SLSQP","trust-constr"], help="Optimizer.")
    p.add_argument("--maxiter", type=int, default=2000, help="Max iterations.")
    p.add_argument("--seed", type=int, default=0, help="Base random seed.")
    p.add_argument("--restarts", type=int, default=1, help="Number of random restarts.")
    p.add_argument("--balance-i", action="store_true", dest="balance_i", default=False, help="Per-i reweighting by 1/range(y_i).")
    p.add_argument("--use-relative-error", action="store_true", dest="use_relative_error", default=False, help="Use relative residuals (f-y)/max(|y|, rel_eps).")
    p.add_argument("--rel-eps", type=float, default=1e-6, dest="rel_eps", help="Floor for |y| in relative residual.")
    return p.parse_args()

def main():
    args = parse_args()
    na = ["NA","NaN",""]
    df = pd.read_csv(args.data, na_values=na)
    for c in ["g","i","x","y"]:
        if c not in df.columns: raise ValueError(f"data.csv missing column '{c}'.")
    df["g"] = pd.to_numeric(df["g"], errors="coerce")
    df["i"] = pd.to_numeric(df["i"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "w" in df.columns:
        df["w"] = pd.to_numeric(df["w"], errors="coerce").fillna(1.0)

    with open(args.func, "r", encoding="utf-8") as f:
        expr_line = f.readline().strip()
    if not expr_line:
        raise ValueError("function.txt first line is empty.")
    model = SympyModel.parse(expr_line)

    cfg = SolverConfig(
        huber_delta=float(args.huber_delta),
        lambda_s=float(args.lambda_s),
        grid_points=int(args.grid_points),
        maxiter=int(args.maxiter),
        method=str(args.method),
        seed=int(args.seed),
        restarts=int(args.restarts),
        balance_i=bool(args.balance_i),
        use_relative_error=bool(args.use_relative_error),
        rel_eps=float(args.rel_eps),
    )

    fitter = Fitter(df=df, model=model, cfg=cfg)
    theta_hat, info = fitter.solve()
    os.makedirs(args.outdir, exist_ok=True)
    fitter.export_all(theta_hat, args.outdir, expr_line)
    print("Fitting completed.", json.dumps(info))

if __name__ == "__main__":
    main()
