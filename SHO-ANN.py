import os
import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import Parallel, delayed

# ===========================
# Silence warning spam
# ===========================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(42)

# ===========================
# Load data
# ===========================
def load_any(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {candidates} found in {os.getcwd()}")

data_path = load_any("data_fixed.csv", "data.csv")
data = pd.read_csv(data_path, encoding="cp1252")
print(f"Loaded: {data_path}")

# Quick label encoding for object cols (fast)
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = LabelEncoder().fit_transform(data[col])

X_full = data.iloc[:, :-1].to_numpy(dtype=np.float32)
y_full = data.iloc[:, -1].to_numpy(dtype=np.float32).reshape(-1, 1)

# Precompute full-data splits (used for final evaluation)
KFOLDS = 5
full_splits = list(KFold(n_splits=KFOLDS, shuffle=True, random_state=42).split(X_full))

# ===========================
# Fold evaluation with leakage-safe preprocessing
# y is scaled to [0,1] on TRAIN only, metrics reported in both spaces
# ===========================
def eval_fold(X, y, train_idx, test_idx, neurons, lr, alpha, noise_ratio=0.03, max_iter=240):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr_raw, y_te_raw = y[train_idx], y[test_idx]

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr).astype(np.float32)
    X_te = sx.transform(X_te).astype(np.float32)

    sy = MinMaxScaler()
    y_tr = sy.fit_transform(y_tr_raw)
    y_te = sy.transform(y_te_raw)

    if noise_ratio > 0:
        yp = y_tr.flatten()
        yp = (yp + np.random.normal(0.0, noise_ratio, size=yp.shape)).clip(0.0, 1.0)
        y_tr = yp.reshape(-1, 1)

    model = MLPRegressor(
        hidden_layer_sizes=(neurons,),
        learning_rate_init=lr,
        alpha=alpha,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=8,
        tol=1e-4
    )
    model.fit(X_tr, y_tr.ravel())

    pred_norm = model.predict(X_te).astype(np.float32).reshape(-1, 1)
    mse_norm  = mean_squared_error(y_te, pred_norm)
    rmse_norm = float(np.sqrt(mse_norm))
    mae_norm  = mean_absolute_error(y_te, pred_norm)

    pred_raw = sy.inverse_transform(pred_norm)
    mse_raw  = mean_squared_error(y_te_raw, pred_raw)
    rmse_raw = float(np.sqrt(mse_raw))
    mae_raw  = mean_absolute_error(y_te_raw, pred_raw)
    r2_raw   = r2_score(y_te_raw, pred_raw)

    return {
        "mse_norm": float(mse_norm),
        "rmse_norm": rmse_norm,
        "mae_norm": float(mae_norm),
        "mse_raw": float(mse_raw),
        "rmse_raw": rmse_raw,
        "mae_raw": float(mae_raw),
        "r2_raw": float(r2_raw),
    }

# ===========================
# Parallel CV wrapper
# No globals, no mutation. Subsample is local-only.
# ===========================
def make_splits(n, k=5, seed=42):
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))

def parallel_cv(X, y, neurons, lr, alpha, fold_ids=None, subsample=None, max_iter=200, n_jobs=-1):
    if subsample is not None and subsample < len(X):
        idx = np.random.choice(len(X), size=subsample, replace=False)
        X_use = X[idx]
        y_use = y[idx]
        use_splits = make_splits(len(idx), k=KFOLDS, seed=42)
    else:
        X_use = X
        y_use = y
        use_splits = full_splits

    if fold_ids is not None:
        use_splits = [use_splits[i] for i in fold_ids if i < len(use_splits)]

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(eval_fold)(X_use, y_use, tr, te, neurons, lr, alpha, max_iter=max_iter)
        for tr, te in use_splits
    )
    return results

# ===========================
# SHO (quiet)
# ===========================
def SHO(objf, lb, ub, dim, n_hyenas, max_iter):
    pop = np.random.uniform(lb, ub, (n_hyenas, dim)).astype(np.float32)
    fit = np.array([objf(x) for x in pop], dtype=np.float32)
    best = pop[int(np.argmin(fit))].copy()

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(n_hyenas):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            D = np.abs(C * best - pop[i])
            pop[i] = best - A * D
            pop[i] = np.clip(pop[i], lb, ub)
        fit = np.array([objf(x) for x in pop], dtype=np.float32)
        best = pop[int(np.argmin(fit))].copy()
    return best

# Speed settings for objective
FAST_FOLDS = [0, 2, 4]               # 3 folds during search
SUBSAMPLE_ROWS = min(4000, len(X_full))
FAST_MAX_ITER = 180

def objective(params):
    neurons = int(params[0])
    lr = float(abs(params[1]))
    alpha = float(abs(params[2]))
    neurons = max(4, min(neurons, 64))
    lr      = max(1e-4, min(lr, 2e-2))
    alpha   = max(1e-5, min(alpha, 1e-1))

    res = parallel_cv(
        X_full, y_full, neurons, lr, alpha,
        fold_ids=FAST_FOLDS, subsample=SUBSAMPLE_ROWS,
        max_iter=FAST_MAX_ITER, n_jobs=-1
    )
    return float(np.mean([r["mse_norm"] for r in res]))

# ===========================
# Run SHO
# ===========================
dim = 3
lb = np.array([4, 1e-4, 1e-5], dtype=np.float32)
ub = np.array([64, 2e-2, 1e-1], dtype=np.float32)

best = SHO(objective, lb, ub, dim, n_hyenas=6, max_iter=12)
bn, blr, ba = int(best[0]), float(best[1]), float(best[2])

print("\n✅ Best Hyperparameters")
print(f"Neurons       : {bn}")
print(f"Learning Rate : {blr:.5f}")
print(f"Alpha         : {ba:.5f}")

# ===========================
# Final full 5-fold evaluation
# ===========================
final = parallel_cv(X_full, y_full, bn, blr, ba, fold_ids=None, subsample=None, max_iter=320, n_jobs=-1)

r2_vals    = [r["r2_raw"] for r in final]
rmse_vals  = [r["rmse_raw"] for r in final]
mae_vals   = [r["mae_raw"] for r in final]
mse_vals   = [r["mse_raw"] for r in final]

rmse_n_vals = [r["rmse_norm"] for r in final]
mae_n_vals  = [r["mae_norm"] for r in final]
mse_n_vals  = [r["mse_norm"] for r in final]

print("\n🎯 Final Average Performance (K = 5)")
print(f"Average R²            : {np.mean(r2_vals):.4f}")
print(f"Average RMSE (raw)    : {np.mean(rmse_vals):.4f}")
print(f"Average MAE  (raw)    : {np.mean(mae_vals):.4f}")
print(f"Average MSE  (raw)    : {np.mean(mse_vals):.4f}")

print("\n📏 Normalized Target-Space (y in [0,1] per fold)")
print(f"Average RMSE_norm     : {np.mean(rmse_n_vals):.4f}")
print(f"Average MAE_norm      : {np.mean(mae_n_vals):.4f}")
print(f"Average MSE_norm      : {np.mean(mse_n_vals):.4f}")

# Targets: normalized errors ≤ 0.10
hit_rmse_norm = np.mean(rmse_n_vals) <= 0.10
hit_mse_norm  = np.mean(mse_n_vals)  <= 0.10
print("\n✅ Target checks (normalized space)")
print(f"RMSE_norm ≤ 0.10 : {'✅' if hit_rmse_norm else '❌'}")
print(f"MSE_norm  ≤ 0.10 : {'✅' if hit_mse_norm  else '❌'}")
