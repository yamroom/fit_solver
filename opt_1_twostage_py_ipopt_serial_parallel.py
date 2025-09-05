
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serial + Parallel Launcher for multistart runs

功能：
- 保留原本單次求解（serial）：直接呼叫子腳本一次
- 並行 multistart（parallel）：用多進程一次跑多個 seed，每個子進程執行子腳本（--restarts 1）
- 自動評分每個 run（以 data.csv vs fitted.csv 的 RMSE 為準），挑選最佳結果，彙整輸出到主 outdir
- 避免過度併行：在子進程環境限制 BLAS/OMP 執行緒數為 1

相容：Windows/Unix（使用 ProcessPoolExecutor）

使用方式（例）：
python opt_1_twostage_py_ipopt_serial_parallel.py \
  --child-script opt_1_twostage_py_ipopt_solutionA.py \
  --data data.csv --func function.txt --outdir out_ipopt \
  --grid-points 200 --restarts 6 --workers 3 \
  --auto-direction true --order-mode auto \
  --lambda-s 1e-5 --lambda-sim 1e-5 \
  --eps-mono 1e-6 --eps-ord 1e-6 --eps-den 1e-8 \
  --relative-error --rel-eps 1e-5 \
  --maxiter 3000 --seed 42 --verbose \
  --ipopt.tol 1e-6 --ipopt.acceptable_tol 1e-4 \
  --ipopt.hessian_approximation limited-memory --ipopt.linear_solver mumps \
  --plot-mode both

說明：
- workers=1 時，等同「原本 serial 單次執行」；restarts 會傳給子腳本（由子腳本自己處理 multistart 串行）
- workers>1 時，外層會把 restarts 分拆成多個子進程（每個 --restarts 1 + 不同 seed），平行跑完後自動挑選最佳 run，
  並把該 run 的輸出複製/彙整到主 outdir（params.csv/fitted.csv/grid.csv/checks.json/圖檔）。
"""
import argparse, os, sys, shutil, json, math, tempfile, uuid, time, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-script", default="opt_1_twostage_py_ipopt_solutionA.py",
                    help="要呼叫的子腳本檔案（預設為你修好的 SolutionA 版本）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=1, help=">1 則使用平行多進程跑多個 seeds")
    ap.add_argument("--restarts", type=int, default=1, help="總共要嘗試的種子數（workers>1 時會分拆）")
    ap.add_argument("--seed", type=int, default=42)
    # 其餘旗標原封轉給子腳本（不在這裡解析），包含：grid-points/auto-direction/order-mode/…/ipopt.* 等
    # 我們只在 CLI 解析到 '--' 前的固定旗標，剩餘用 parse_known_args 接續傳遞
    args, rest = ap.parse_known_args()
    return args, rest

def _env_for_child():
    env = dict(os.environ)
    # 限制每個子進程的 BLAS/OMP 執行緒數，避免多進程 * 多執行緒過度併行
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_MAX_THREADS", "1")
    return env

def _run_child_once(py, script, base_args, extra_args, outdir, timeout=0):
    cmd = [py, str(script)] + base_args + extra_args
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=_env_for_child(), timeout=(timeout or None), universal_newlines=True)
        ok = True; msg = cp.stdout
    except subprocess.TimeoutExpired as e:
        ok = False; msg = f"[TIMEOUT] {e}"
    except subprocess.CalledProcessError as e:
        ok = False; msg = f"[RET={e.returncode}] stdout:\n{e.stdout}\n--- stderr:\n{e.stderr}"
    dt = time.perf_counter() - t0
    return ok, dt, msg

def _score_run(outdir: Path, data_csv: Path):
    """以 RMSE(data.y, fitted.y_hat) 當作分數（愈小愈好）。如果檔案缺失，回傳 +inf。"""
    import pandas as pd, numpy as np
    fitted_fp = outdir/"fitted.csv"
    if not fitted_fp.exists():
        return float("inf")
    try:
        data = pd.read_csv(data_csv).dropna(subset=["y"])
        fit = pd.read_csv(fitted_fp)
        m = data.merge(fit[["g","i","x","y_hat"]], on=["g","i","x"], how="left")
        r = (m["y_hat"] - m["y"]).to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(r**2)))
        return rmse
    except Exception as e:
        return float("inf")

def _collect_outputs(best_dir: Path, final_outdir: Path):
    final_outdir.mkdir(parents=True, exist_ok=True)
    for name in ["params.csv", "fitted.csv", "grid.csv", "checks.json"]:
        src = best_dir/name
        if src.exists():
            shutil.copy2(src, final_outdir/name)
    # 圖檔
    for p in list(best_dir.glob("plots_by_i_i=*.png")) + list(best_dir.glob("residuals_g*.png")):
        shutil.copy2(p, final_outdir/p.name)

def main():
    args, rest = parse_args()
    py = sys.executable
    child = Path(args.child_script).resolve()
    assert child.exists(), f"找不到子腳本：{child}"
    data = Path(args.data).resolve()
    func = Path(args.func).resolve()
    final_out = Path(args.outdir).resolve()
    final_out.mkdir(parents=True, exist_ok=True)

    # 先組出 child 的「共同參數」（基礎部分）
    base = [
        "--data", str(data),
        "--func", str(func),
        "--outdir", "",            # 由外層分配子 outdir
        "--restarts", "1",         # 子進程一律只跑 1 次
    ]

    # 把我們這層解析掉的標準旗標回填（其餘在 rest 中）
    # - workers 不往下傳
    # - restarts 由外層拆分，不往下傳
    # - seed 每個 run 各自指定
    # 其餘 rest 原封傳下去
    others = [x for x in rest]

    if args.workers <= 1:
        # 純 serial：把 outdir 指到最終 outdir，restarts 交給子腳本自己跑（串行）
        base_serial = base.copy()
        base_serial[base_serial.index("--outdir")+1] = str(final_out)
        # 這時把 '--restarts 1' 換成使用者指定的 restarts 數量
        idx = base_serial.index("--restarts"); base_serial[idx+1] = str(args.restarts)
        ok, dt, msg = _run_child_once(py, child, base_serial, others + ["--seed", str(args.seed)], final_out)
        print(f"[serial] elapsed={dt:.3f}s ok={ok}")
        if not ok:
            print(msg)
        sys.exit(0 if ok else 1)

    # 並行：拆成多個子 run，每個 --restarts 1 + 不同 seed，各自寫到 outdir_child_k
    # 種子序列
    seeds = [args.seed + k for k in range(args.restarts)]
    tmp_root = final_out.parent / f"{final_out.name}_workers_tmp_{uuid.uuid4().hex[:8]}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    def task(seed_val: int):
        child_out = tmp_root / f"run_seed{seed_val}"
        child_out.mkdir(parents=True, exist_ok=True)
        base_par = base.copy()
        base_par[base_par.index("--outdir")+1] = str(child_out)
        ok, dt, msg = _run_child_once(py, child, base_par, others + ["--seed", str(seed_val)], child_out)
        score = _score_run(child_out, data)
        return {"seed": seed_val, "ok": ok, "elapsed_s": dt, "score_rmse": score, "outdir": str(child_out), "log": msg if not ok else ""}

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(task, s) for s in seeds]
        for fu in as_completed(futs):
            results.append(fu.result())
            r = results[-1]
            print(f"[parallel] seed={r['seed']} ok={r['ok']} elapsed={r['elapsed_s']:.2f}s score(RMSE)={r['score_rmse']:.6g} out={r['outdir']}")

    elapsed = time.perf_counter() - t0
    # 揀最小 RMSE 的 run
    ok_runs = [r for r in results if r["ok"] and math.isfinite(r["score_rmse"])]
    if not ok_runs:
        print("[parallel] 所有子 run 皆失敗或無法評分，請查看各子目錄 log")
        sys.exit(2)

    best = min(ok_runs, key=lambda r: r["score_rmse"])
    best_dir = Path(best["outdir"])
    _collect_outputs(best_dir, final_out)

    # 輸出總結
    summary = {
        "mode": "serial+parallel",
        "workers": args.workers,
        "restarts_total": args.restarts,
        "elapsed_s_total": elapsed,
        "best_seed": best["seed"],
        "best_rmse": best["score_rmse"],
        "best_outdir": str(best_dir),
        "runs": results,
    }
    with open(final_out/"parallel_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[parallel] done. total={elapsed:.2f}s best_seed={best['seed']} best_rmse={best['score_rmse']:.6g}")
    print(f"[parallel] 最佳結果已彙整到：{final_out}")
    print(f"[parallel] 各子 run 的輸出在：{tmp_root}")
    # 不自動刪除 tmp_root，方便排錯；若要清理可自行刪除該資料夾。

if __name__ == "__main__":
    import math, time, uuid
    main()
