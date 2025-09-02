# opt1_twostage.py
import os, sys, subprocess, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "/mnt/data"
OUT_DIR  = os.path.join(BASE_DIR, "out_opt1_twostage")
os.makedirs(OUT_DIR, exist_ok=True)

rows = [
    (0, 0.5,   0.4568, 0.0162),
    (0, 0.5,   0.5597, 0.0122),
    (0, 0.5,   0.6571, float("nan")),
    (0, 0.5,   1.1,    0.01),
    (0, 0.999, 0.4568, float("nan")),
    (0, 0.999, 0.5597, 0.175),
    (0, 0.999, 0.6571, float("nan")),
    (0, 0.999, 1.1,    0.056),
    (1, 0.5,   0.3,    0.032),
    (1, 0.5,   0.4769, 0.0167),
    (1, 0.5,   0.5817, 0.0130),
    (1, 0.5,   0.6878, 0.0128),
    (1, 0.5,   1.1,    0.0130),
    (1, 0.999, 0.3,    0.9),
    (1, 0.999, 0.4769, 0.47),
    (1, 0.999, 0.5817, 0.2682),
    (1, 0.999, 0.6878, 0.1867),
    (1, 0.999, 1.1,    0.0571),
]
df = pd.DataFrame(rows, columns=["g","i","x","y"])
df.to_csv(os.path.join(OUT_DIR,"data_1.csv"), index=False)

try:
    import casadi as ca
except Exception:
    wheel = os.path.join(BASE_DIR, "casadi-3.7.1-cp311-none-manylinux2014_x86_64.whl")
    if not os.path.exists(wheel):
        raise RuntimeError("CasADi wheel not found. Please pip install casadi or provide wheel.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel], check=True)
    import casadi as ca

def f_sym(i, x, p):
    C1_A,C1_B,C1_C,C1_D,C2_A,C2_B,C2_C,C2_D = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    num = ca.log(-ca.log(i)) + (C2_A + C2_B*(1.1-x)*(1.1-x) + C2_C*ca.exp(1.1-x) + C2_D*ca.exp(x-1.1))
    den = (C1_A + C1_B*(1.1-x)*(1.1-x) + C1_C*ca.exp(1.1-x) + C1_D*ca.exp(x-1.1))
    return - num / den, den

def f_num(i, x, pv):
    C1_A,C1_B,C1_C,C1_D,C2_A,C2_B,C2_C,C2_D = pv
    num = (math.log(-math.log(i)) + (C2_A + C2_B*(1.1-x)*(1.1-x) + C2_C*math.exp(1.1-x) + C2_D*math.exp(x-1.1)))
    den = (C1_A + C1_B*(1.1-x)*(1.1-x) + C1_C*math.exp(1.1-x) + C1_D*math.exp(x-1.1))
    return - num / den

# Stage A: Unconstrained
p0 = ca.MX.sym("p0", 8); p1 = ca.MX.sym("p1", 8)
theta = ca.vertcat(p0,p1)
rel_eps = 1e-3
obs = df[~df["y"].isna()].copy()
resid=[]
for _,r in obs.iterrows():
    p = p1 if int(r.g)==1 else p0
    yhat,_ = f_sym(float(r.i), float(r.x), p)
    w = 1.0/max(abs(float(r.y)), rel_eps)
    resid.append(ca.sqrt(w)*(yhat - float(r.y)))
objA = 0.5*ca.dot(ca.vertcat(*resid), ca.vertcat(*resid)) + 1e-8*ca.dot(p1-p0, p1-p0)
solverA = ca.nlpsol("solverA","ipopt", {"x":theta,"f":objA}, {"ipopt.print_level":0,"print_time":0,"ipopt.max_iter":2500})
x0 = np.array([10,0,0,0,0,0,0,0,  12,0,0,0,0,0,0,0], dtype=float)
x_uc = np.array(solverA(x0=x0)["x"]).reshape(-1)

# Stage B: Non-crossing + denominator safety
p0 = ca.MX.sym("p0", 8); p1 = ca.MX.sym("p1", 8)
theta = ca.vertcat(p0,p1)
resid=[]
for _,r in obs.iterrows():
    p = p1 if int(r.g)==1 else p0
    yhat,_ = f_sym(float(r.i), float(r.x), p)
    w = 1.0/max(abs(float(r.y)), rel_eps)
    resid.append(ca.sqrt(w)*(yhat - float(r.y)))
objB = 0.5*ca.dot(ca.vertcat(*resid), ca.vertcat(*resid)) + 1e-7*ca.dot(p1-p0, p1-p0)

xs_obs = {iv: sorted(df[(df["i"]==iv)&df["x"].notna()]["x"].unique().tolist()) for iv in sorted(df["i"].unique())}
g_list=[]
ord_eps = 1e-5; den_eps = 1e-6
for iv, xs in xs_obs.items():
    xs_uni = np.linspace(0.3,1.1,35)
    xs_all = np.unique(np.concatenate([xs, xs_uni]))
    for xv in xs_all:
        y0,den0 = f_sym(float(iv), float(xv), p0)
        y1,den1 = f_sym(float(iv), float(xv), p1)
        g_list.append(y1 - y0)
        g_list.append(den0)
        g_list.append(den1)
g_con = ca.vertcat(*g_list)
lb=[]; ub=[]
count = sum(len(np.unique(np.concatenate([xs_obs[iv], np.linspace(0.3,1.1,35)]))) for iv in xs_obs)
for _ in range(count):
    lb += [ord_eps, den_eps, den_eps]; ub += [1e20, 1e20, 1e20]
lb_g = np.array(lb, dtype=float); ub_g = np.array(ub, dtype=float)

solverB = ca.nlpsol("solverB","ipopt", {"x":theta,"f":objB,"g":g_con},
                    {"ipopt.print_level":0,"print_time":0,"ipopt.max_iter":3000,"ipopt.tol":1e-8,"ipopt.constr_viol_tol":1e-8})
solB = solverB(x0=x_uc, lbg=lb_g, ubg=ub_g)
xopt = np.array(solB["x"]).reshape(-1)
p0_hat, p1_hat = xopt[:8], xopt[8:]

# Metrics
obs2 = obs.copy()
obs2["y_hat"] = obs2.apply(lambda r: f_num(r.i, r.x, p1_hat if int(r.g)==1 else p0_hat), axis=1)
obs2["ae"] = (obs2["y_hat"] - obs2["y"]).abs()
obs2["re"] = obs2["ae"] / obs2["y"].abs().clip(lower=1e-9)
rmse = float(np.sqrt((obs2["ae"]**2).mean()))
mape = float(obs2["re"].mean())*100.0
print({"RMSE": rmse, "MAPE_%": mape, "n_obs": int(len(obs2))})

# Dense non-crossing check
xs_dense = np.linspace(0.3,1.1,300)
def f_num_d(i,x,pv):
    C1_A,C1_B,C1_C,C1_D,C2_A,C2_B,C2_C,C2_D = pv
    den = (C1_A + C1_B*(1.1-x)*(1.1-x) + C1_C*np.exp(1.1-x) + C1_D*np.exp(x-1.1))
    num = (np.log(-np.log(i)) + (C2_A + C2_B*(1.1-x)*(1.1-x) + C2_C*np.exp(1.1-x) + C2_D*np.exp(x-1.1)))
    return - num / den
def margin(iv):
    y0 = np.array([f_num_d(iv, xv, p0_hat) for xv in xs_dense])
    y1 = np.array([f_num_d(iv, xv, p1_hat) for xv in xs_dense])
    return float(np.min(y1 - y0))
margins = {iv: margin(iv) for iv in sorted(df["i"].unique())}
print({"min_noncross_margin_by_i": margins})

# Plots + save
for iv in sorted(df["i"].unique()):
    y0_line = np.array([f_num_d(iv, xv, p0_hat) for xv in xs_dense])
    y1_line = np.array([f_num_d(iv, xv, p1_hat) for xv in xs_dense])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(xs_dense, y0_line, label=f"g=0 (i={iv})")
    plt.plot(xs_dense, y1_line, label=f"g=1 (i={iv})")
    sub = df[(df["i"]==iv) & df["y"].notna()]
    plt.scatter(sub[sub["g"]==0]["x"], sub[sub["g"]==0]["y"], s=40, label="g=0 data")
    plt.scatter(sub[sub["g"]==1]["x"], sub[sub["g"]==1]["y"], s=40, label="g=1 data", marker="x")
    plt.title(f"Two-stage (Option 1) — NON-CROSSING only — i={iv}")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    png = os.path.join(OUT_DIR, f"opt1_twostage_i={iv}.png")
    plt.savefig(png, dpi=130, bbox_inches="tight")
    plt.close()

# Save CSVs
params_df = pd.DataFrame({
    "param": ["C1_A","C1_B","C1_C","C1_D","C2_A","C2_B","C2_C","C2_D"]*2,
    "g":     [0]*8 + [1]*8,
    "value": list(p0_hat) + list(p1_hat)
})
params_df.to_csv(os.path.join(OUT_DIR, "params_opt1_twostage.csv"), index=False)
obs2.to_csv(os.path.join(OUT_DIR, "fitted_opt1_twostage.csv"), index=False)
print("Saved:", {"params": os.path.join(OUT_DIR, "params_opt1_twostage.csv"),
                "fitted": os.path.join(OUT_DIR, "fitted_opt1_twostage.csv"),
                "pngs_dir": OUT_DIR})
