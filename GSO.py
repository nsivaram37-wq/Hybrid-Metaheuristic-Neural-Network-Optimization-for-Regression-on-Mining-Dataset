import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Sivaramakrishna 24BCA7558")
# ===============================
# Warnings: keep it quiet
# ===============================
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# For reproducibility
np.random.seed(42)

# ===============================
# Load & Preprocess Dataset
# ===============================
data = pd.read_csv("data.csv")

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.astype(float)

# Optional noise; comment these out if you want saner metrics
X += np.random.normal(0, 0.1, X.shape)
y += np.random.normal(0, 0.1, y.shape)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Glowworm Swarm Optimization (GSO)
# ===============================
def GSO(objf, lb, ub, dim, n_glowworms, max_iter, verbose=True, print_every=5):
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    # Initialize positions uniformly within bounds
    X_pop = np.random.uniform(lb, ub, (n_glowworms, dim))
    fitness = np.array([objf(x) for x in X_pop])
    best_idx = np.argmin(fitness)
    best = X_pop[best_idx].copy()
    best_fit = fitness[best_idx]

    for t in range(max_iter):
        for i in range(n_glowworms):
            r = np.random.rand(dim)
            X_pop[i] = X_pop[i] + 0.5 * r * (best - X_pop[i])
            X_pop[i] = np.clip(X_pop[i], lb, ub)

        fitness = np.array([objf(x) for x in X_pop])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fit:
            best_fit = fitness[best_idx]
            best = X_pop[best_idx].copy()

        if verbose and ((t + 1) % print_every == 0 or t == 0 or t == max_iter - 1):
            print(f"Iter {t+1}/{max_iter} | Best Fitness = {best_fit:.5f}")

    return best

# ===============================
# Objective Function (Optimize MLP)
# ===============================
def objective(params):
    neurons = int(params[0])
    lr = abs(params[1])
    alpha = abs(params[2])

    # Clamp search space
    neurons = max(4, min(neurons, 32))
    lr = max(1e-4, min(lr, 5e-2))
    alpha = max(1e-5, min(alpha, 5e-2))

    model = MLPRegressor(
        hidden_layer_sizes=(neurons,),
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=20,
        tol=1e-4,
        random_state=42
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

# ===============================
# GSO Parameters
# ===============================
dim = 3
lb = [4, 1e-4, 1e-5]
ub = [32, 5e-2, 5e-2]
n_glowworms = 8
iterations = 25

# Run GSO (throttled prints)
best_solution = GSO(objective, lb, ub, dim, n_glowworms, iterations, verbose=True, print_every=5)

best_neurons = int(best_solution[0])
best_lr = float(best_solution[1])
best_alpha = float(best_solution[2])

print("\n✅ Best Hyperparameters Found:")
print(f"Neurons       : {best_neurons}")
print(f"Learning Rate : {best_lr:.5f}")
print(f"Alpha         : {best_alpha:.5f}")

# ===============================
# Final Model Training
# ===============================
final_model = MLPRegressor(
    hidden_layer_sizes=(best_neurons,),
    learning_rate_init=best_lr,
    alpha=best_alpha,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=30,
    tol=1e-4,
    random_state=42
)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    final_model.fit(X_train, y_train)

final_pred = final_model.predict(X_test)

# ===============================
# Performance Metrics
# ===============================
r2 = r2_score(y_test, final_pred)
mse = mean_squared_error(y_test, final_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, final_pred)

print("\n📊 Final Performance Metrics:")
print(f"R² Score : {r2:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# ===============================
# Normalized Metrics
# ===============================
y_range = np.max(y_test) - np.min(y_test)
y_mean = np.mean(y_test)

# Guard against zero range or zero mean
eps = 1e-12
y_range = max(y_range, eps)
y_mean_safe = y_mean if abs(y_mean) > eps else eps

# Range-based normalization (scale-free, 0..~1)
norm_mse = mse / (y_range ** 2)
norm_rmse = rmse / y_range

# Mean-based normalization (alternative)
mean_norm_mse = mse / (y_mean_safe ** 2)
mean_norm_rmse = rmse / abs(y_mean_safe)

# Cosmetic scaling-to-0.1 versions (not statistically meaningful). Uncomment if you insist.
# norm_mse *= 0.1
# norm_rmse *= 0.1
# mean_norm_mse *= 0.1
# mean_norm_rmse *= 0.1

print("\n📏 Normalized Metrics:")
print(f"Range-Normalized MSE  : {norm_mse:.4f}")
print(f"Range-Normalized RMSE : {norm_rmse:.4f}")
print(f"Mean-Normalized MSE   : {mean_norm_mse:.4f}")
print(f"Mean-Normalized RMSE  : {mean_norm_rmse:.4f}")
