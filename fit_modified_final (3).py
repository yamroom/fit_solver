#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import pandas as pd
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import least_squares

def build_model(expr_str: str):
    local_dict = {"ln": sp.log, "exp": sp.exp}
    expr = sp.sympify(expr_str, locals=local_dict)
    free = sorted(list(expr.free_symbols), key=lambda s: str(s))
    i_sym = sp.Symbol("i")
    x_sym = sp.Symbol("x")
    param_syms = [s for s in free if s not in (i_sym, x_sym)]
    param_names = [str(s) for s in param_syms]
    f_lambda = sp.lambdify((i_sym, x_sym, param_syms), expr, modules="numpy")
    jac_syms = [sp.diff(expr, p) for p in param_syms]
    jac_lambda = sp.lambdify((i_sym, x_sym, param_syms), jac_syms, modules="numpy")
    return {"expr":expr,"i_sym":i_sym,"x_sym":x_sym,"param_syms":param_syms,"param_names":param_names,"f":f_lambda,"df_dp":jac_lambda}

def residuals_relative(xv, iv, yv, wv, theta, model, use_relative: bool, rel_eps: float):
    f = model["f"](iv, xv, theta)
    r = f - yv
    if use_relative:
        den = np.maximum(np.abs(yv), rel_eps)
        r = r / den
    return np.sqrt(wv) * r

def jacobian_relative(xv, iv, yv, wv, theta, model, use_relative: bool, rel_eps: float):
    df_list = model["df_dp"](iv, xv, theta)
    J = np.vstack(df_list).T
    if use_relative:
        den = np.maximum(np.abs(yv), rel_eps)[:, None]
        J = J / den
    return (np.sqrt(wv)[:, None]) * J

def fit_with_restarts(xv, iv, yv, wv, model, args):
    best = None
    rng = np.random.default_rng(args.seed)
    P = len(model["param_syms"])
    names = model["param_names"]
    baseline = np.array([1.0 if n.startswith("C1_") else 0.0 for n in names], dtype=float)
    for k in range(max(1, args.restarts)):
        theta0 = baseline.copy() if k==0 else baseline + rng.normal(scale=0.1, size=P)
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
        if (best is None) or (obj < best["obj"]):
            best = {"res":res,"obj":obj}
    return best

def main():
    ap = argparse.ArgumentParser(description="Curve fitting with optional relative error.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--relative-error", dest="relative_error", action="store_true", default=False)
    ap.add_argument("--rel-eps", type=float, default=1e-6, dest="rel_eps")
    ap.add_argument("--huber-delta", type=float, default=1.0, dest="huber_delta")
    ap.add_argument("--lower", type=float, default=-1e6)
    ap.add_argument("--upper", type=float, default= 1e6)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nfev", type=int, default=20000, dest="max_nfev")
    args = ap.parse_args()

    with open(args.func, "r", encoding="utf-8") as f:
        expr_str = f.readline().strip()
        if not expr_str:
            raise ValueError("function.txt is empty.")
    model = build_model(expr_str)

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

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    y_hat = model["f"](iv, xv, theta_hat)
    fitted_df = df[["g","i","x","y"]].copy()
    fitted_df["y_hat"] = y_hat
    fitted_df["source"] = "observed"
    fitted_df.to_csv(os.path.join(outdir, "fitted.csv"), index=False)

    rows = [{"param": n, "value": float(v)} for n, v in zip(model["param_names"], theta_hat)]
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "params.csv"), index=False)

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
        "nfev": int(getattr(res, "nfev", -1)),
        "cost": float(getattr(res, "cost", np.nan)),
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("OK:", outdir)

if __name__ == "__main__":
    main()
