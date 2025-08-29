
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols
from sympy.utilities.lambdify import lambdify
from scipy.optimize import least_squares

def build_model(expr_str: str):
    # Allow 'ln' alias
    local_dict = {"ln": sp.log, "exp": sp.exp}
    expr = sp.sympify(expr_str, locals=local_dict)

    # Identify symbols: parameters vs variables (i, x)
    free = sorted(list(expr.free_symbols), key=lambda s: str(s))
    i_sym = sp.Symbol("i")
    x_sym = sp.Symbol("x")
    param_syms = [s for s in free if s not in (i_sym, x_sym)]
    param_names = [str(s) for s in param_syms]

    # f(i, x, theta)
    f_lambda = lambdify((i_sym, x_sym, param_syms), expr, modules="numpy")
    # Jacobian wrt params (list of partials)
    jac_syms = [sp.diff(expr, p) for p in param_syms]
    jac_lambda = lambdify((i_sym, x_sym, param_syms), jac_syms, modules="numpy")

    return {
        "expr": expr,
        "i_sym": i_sym,
        "x_sym": x_sym,
        "param_syms": param_syms,
        "param_names": param_names,
        "f": f_lambda,
        "df_dp": jac_lambda,
    }

def residuals_relative(xv, iv, yv, wv, theta, model, use_relative: bool, rel_eps: float):
    # Compute residual vector and analytic Jacobian (N, P)
    f = model["f"](iv, xv, theta)  # shape (N,)
    r = f - yv
    if use_relative:
        den = np.maximum(np.abs(yv), rel_eps)
        r = r / den
    # Weight: least_squares loss acts on residual vector; use sqrt weights
    r = np.sqrt(wv) * r
    return r

def jacobian_relative(xv, iv, yv, wv, theta, model, use_relative: bool, rel_eps: float):
    # df/dp for each param -> stack to (P, N), then transpose to (N, P)
    df_list = model["df_dp"](iv, xv, theta)  # list or array-like length P, each (N,)
    J = np.vstack(df_list).T  # (N, P)
    if use_relative:
        den = np.maximum(np.abs(yv), rel_eps)[:, None]
        J = J / den
    J = (np.sqrt(wv)[:, None]) * J
    return J

def fit_with_restarts(xv, iv, yv, wv, model, args):
    best = None
    rng = np.random.default_rng(args.seed)
    P = len(model["param_syms"])
    names = model["param_names"]

    # Build a safe baseline init to avoid denominator = 0 in the provided expression:
    # - For parameters starting with 'C1_' (appear in denominator), start at 1.0
    # - Others start at 0.0
    baseline = np.array([1.0 if n.startswith("C1_") else 0.0 for n in names], dtype=float)

    for k in range(max(1, args.restarts)):
        if k == 0:
            theta0 = baseline.copy()
        else:
            theta0 = baseline + rng.normal(scale=0.1, size=P)

        res = least_squares(
            fun=lambda th: residuals_relative(xv, iv, yv, wv, th, model, args.relative_error, args.rel_eps),
            x0=theta0,
            jac=lambda th: jacobian_relative(xv, iv, yv, wv, th, model, args.relative_error, args.rel_eps),
            bounds=(np.full(P, args.lower), np.full(P, args.upper)),
            method="trf",
            loss="huber",
            f_scale=args.huber_delta,
            max_nfev=args.max_nfev,
            verbose=0
        )

        obj = 0.5 * np.sum(res.fun**2)
        cand = {"res": res, "obj": obj}
        if (best is None) or (obj < best["obj"]):
            best = cand

    return best


def main():
    ap = argparse.ArgumentParser(description="Curve fitting with optional relative error.")
    ap.add_argument("--data", required=True, help="Path to data.csv (cols: g,i,x,y[,w])")
    ap.add_argument("--func", required=True, help="Path to function.txt (first line is expression in i,x and params)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--relative-error", dest="relative_error", action="store_true", default=False, help="Use relative residuals (f-y)/max(|y|,rel_eps)")
    ap.add_argument("--rel-eps", type=float, default=1e-6, dest="rel_eps", help="Epsilon floor for relative residual denominator")
    ap.add_argument("--huber-delta", type=float, default=1.0, dest="huber_delta", help="Huber f_scale parameter")
    ap.add_argument("--lower", type=float, default=-1e6, help="Lower bound for params")
    ap.add_argument("--upper", type=float, default= 1e6, help="Upper bound for params")
    ap.add_argument("--restarts", type=int, default=1, help="Random restarts count")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--max-nfev", type=int, default=20000, dest="max_nfev", help="Max function evaluations")
    args = ap.parse_args()

    # Read function
    with open(args.func, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise ValueError("function.txt is empty.")
        expr_str = line

    model = build_model(expr_str)

    # Read data
    na = ["NA","NaN",""]
    df = pd.read_csv(args.data, na_values=na)
    for col in ["g","i","x","y"]:
        if col not in df.columns:
            raise ValueError(f"data.csv missing required column '{col}'")
    if "w" not in df.columns:
        df["w"] = 1.0
    df = df.dropna(subset=["g","i","x","y"]).copy()
    df["g"] = pd.to_numeric(df["g"], errors="coerce")
    df["i"] = pd.to_numeric(df["i"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["w"] = pd.to_numeric(df["w"], errors="coerce").fillna(1.0)

    xv = df["x"].to_numpy(float)
    iv = df["i"].to_numpy(float)
    yv = df["y"].to_numpy(float)
    wv = df["w"].to_numpy(float)

    best = fit_with_restarts(xv, iv, yv, wv, model, args)
    res = best["res"]
    theta_hat = res.x

    # Outputs
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # fitted.csv (observed points)
    y_hat = model["f"](iv, xv, theta_hat)
    fitted_df = df[["g","i","x","y"]].copy()
    fitted_df["y_hat"] = y_hat
    fitted_df["source"] = "observed"
    fitted_df.to_csv(os.path.join(outdir, "fitted.csv"), index=False)

    # params.csv
    rows = []
    for name, val in zip(model["param_names"], theta_hat):
        rows.append({"param": name, "value": float(val)})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "params.csv"), index=False)

    # meta.json
    meta = {
        "objective": float(best["obj"]),
        "relative_error": bool(args.relative_error),
        "rel_eps": float(args.rel_eps),
        "huber_delta": float(args.huber_delta),
        "restarts": int(args.restarts),
        "seed": int(args.seed),
        "expr": expr_str,
        "params": model["param_names"],
        "success": bool(res.success),
        "message": str(getattr(res, "message", "")),
        "nfev": int(getattr(res, "nfev", -1)),
        "cost": float(getattr(res, "cost", np.nan)),
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("OK: fitted, params, and meta saved to", outdir)

if __name__ == "__main__":
    main()
