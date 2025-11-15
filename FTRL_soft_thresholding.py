import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
import os

# --- Simulation Parameters ---
# Note: K removed (no hard top-k)
v_tar = 1.02       # Target price-relative
lambda_2 = 22.0   # Entropy regularization
lambda_1 = 1e-3    # Soft-L1 shrinkage hyperparameter (tune this)
gamma = 0.999      # Forgetting factor
eta = 0.1          # Learning rate for OMD
#
# --------------------------
# 1. Load and Process Data
# --------------------------
def load_market_data(parquet_path='data/processed/prices_1D.parquet'):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"❌ Could not find {parquet_path}. Please ensure your preprocessing pipeline has generated it."
        )
    print(f"📂 Loading market prices from {parquet_path} ...")
    prices_df = pd.read_parquet(parquet_path)
    if prices_df.empty:
        raise ValueError("Loaded parquet file is empty.")
    returns_df = prices_df / prices_df.shift(1)
    returns_df = returns_df.iloc[1:]
    returns_df.replace([np.inf, -np.inf], 1.0, inplace=True)
    returns_df.fillna(1.0, inplace=True)
    returns_df[returns_df <= 0] = 1e-12
    market_returns = returns_df.values
    print(f"✅ Converted to price relatives. Shape: {market_returns.shape}")
    return market_returns

market_returns = load_market_data()
T, N = market_returns.shape
print(f"Data loaded: {T} time steps, {N} assets.\n")

# --------------------------
# 2. Helper Functions
# --------------------------
def find_best_fixed_portfolio(objective_func, N):
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(N))
    w0 = np.ones(N) / N
    res = minimize(objective_func, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'ftol':1e-9, 'maxiter':1000})
    if not res.success:
        print("⚠️ Warning: optimization failed, using uniform portfolio.")
        return w0
    return res.x

def soft_threshold_and_normalize(w, lam1, eps=1e-12):
    """
    Soft-threshold (shrink) nonnegative vector w by lam1 and renormalize to simplex.
    This creates sparsity by setting small components to zero, then normalizing.
    """
    # ensure w is nonnegative
    w = np.maximum(w, 0.0)
    # shrink
    w_shrunk = np.maximum(w - lam1, 0.0)
    s = np.sum(w_shrunk)
    if s <= eps:
        # fallback: if everything zeroed out, return uniform
        return np.ones_like(w) / len(w)
    return w_shrunk / s

def find_best_fixed_sparse_benchmark(market_returns, lambda_1, lambda_2, v_tar=1.01):
    """Compute best fixed portfolio in hindsight with entropy regularizer + soft-L1 postprocessing."""
    print("🔹 Computing best fixed dense benchmark portfolio ...")
    T, N = market_returns.shape

    def total_asymmetric_loss(w):
        daily_returns = market_returns @ w
        shortfalls = v_tar - daily_returns
        losses = (np.maximum(0, shortfalls)) ** 2
        return np.sum(losses)

    # Get dense optimum (subject to simplex)
    w_dense = find_best_fixed_portfolio(total_asymmetric_loss, N)
    # Apply soft-L1 shrink & renormalize to obtain sparse-like portfolio
    w_sparse = soft_threshold_and_normalize(w_dense, lambda_1)
    final_loss = total_asymmetric_loss(w_sparse)
    k_effective = int(np.sum(w_sparse > 1e-8))
    print(f"✅ Benchmark computed. Final Loss = {final_loss:.6f}. Effective K = {k_effective}")
    print(f"Nonzero asset indices: {np.where(w_sparse > 1e-8)[0].tolist()}\n")
    return w_sparse, final_loss

# Benchmark (uses soft L1 to produce sparse-ish benchmark)
w_star_sparse, benchmark_total_loss = find_best_fixed_sparse_benchmark(market_returns, lambda_1, lambda_2, v_tar)

# --------------------------
# 3. FTRL Algorithm (no hard K; use soft-L1 shrink)
# --------------------------
def simulate_ftrl(market_returns, lambda_2, gamma, lambda_1=0.0, v_tar=1.01, verbose=True):
    """FTRL with entropy regularization + forgetting + soft-L1 shrink (postprocessing)."""
    if verbose:
        print(f"🚀 Running FTRL (λ2={lambda_2}, γ={gamma}, λ1={lambda_1}) ...")
    start = time.time()
    T, N = market_returns.shape
    w_t = np.ones(N) / N
    losses, wealth, weights = [], [1.0], [w_t.copy()]
    B_t = np.zeros((N, N))
    v_t = np.zeros(N)
    k_history = []

    for t in range(T):
        r_t = market_returns[t]
        daily = float(w_t @ r_t)
        if daily <= 0:
            daily = 1e-12

        shortfall = v_tar - daily
        losses.append((np.maximum(0, shortfall)) ** 2)
        wealth.append(wealth[-1] * daily)

        # Forgetting accumulation
        B_t = gamma * B_t + np.outer(r_t, r_t)
        v_t = gamma * v_t + r_t

        # Objective for current cumulative (forgetting) data
        def ftrl_obj(w):
            w_safe = np.maximum(w, 1e-12)
            quad = float(w @ B_t @ w)
            lin = -2 * v_tar * float(v_t @ w)
            ent = lambda_2 * np.sum(w_safe * np.log(w_safe))
            # NOTE: we do NOT include lambda_1 * ||w||_1 here because for nonnegative simplex it's constant
            return quad + lin + ent

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(N)]
        res = minimize(ftrl_obj, w_t, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-9, 'maxiter':500})

        if res.success:
            w_dense = res.x
        else:
            w_dense = w_t

        # Apply soft-L1 shrinkage + renormalize to encourage sparsity in practice
        w_t = soft_threshold_and_normalize(w_dense, lambda_1)
        weights.append(w_t.copy())

        # record effective K (number of nonzeros)
        k_history.append(int(np.sum(w_t > 1e-8)))

        if (t + 1) % 50 == 0 and verbose:
            print(f"  Step {t+1}/{T} complete...")

    if verbose:
        print(f"✅ FTRL done in {time.time() - start:.2f}s\n")
    return np.array(losses), np.array(wealth[1:]), np.array(weights[:-1]), np.array(k_history)

# --------------------------
# 4. OMD Algorithm (no hard K; use soft-L1 shrink)
# --------------------------
def simulate_omd(market_returns, eta, lambda_1=0.0, v_tar=1.01, verbose=True):
    """OMD (Exponentiated Gradient) with asymmetric shortfall loss + soft-L1 shrink postprocessing."""
    if verbose:
        print(f"🚀 Running OMD (η={eta}, λ1={lambda_1}, v_tar={v_tar}) ...")
    start = time.time()
    T, N = market_returns.shape
    w_t = np.ones(N) / N
    losses, wealth, weights = [], [1.0], [w_t.copy()]
    k_history = []

    for t in range(T):
        r_t = market_returns[t]
        daily = float(w_t @ r_t)
        if daily <= 0:
            daily = 1e-12

        shortfall = v_tar - daily
        losses.append((np.maximum(0, shortfall)) ** 2)
        wealth.append(wealth[-1] * daily)

        grad = -2 * shortfall * r_t if shortfall > 0 else np.zeros(N)

        # Exponentiated gradient update (mirror descent w/ negative entropy)
        update = np.exp(np.clip(-eta * grad, -50, 50))
        w_t = w_t * update
        sumw = np.sum(w_t)
        if sumw <= 0:
            w_t = np.ones_like(w_t) / len(w_t)
        else:
            w_t /= sumw

        # Soft-L1 shrink + renormalize to promote sparsity
        w_t = soft_threshold_and_normalize(w_t, lambda_1)
        weights.append(w_t.copy())
        k_history.append(int(np.sum(w_t > 1e-8)))

    if verbose:
        print(f"✅ OMD done in {time.time() - start:.2f}s\n")
    return np.array(losses), np.array(wealth[1:]), np.array(weights[:-1]), np.array(k_history)

# --------------------------
# 5. Run Simulations
# --------------------------
loss_ftrl, wealth_ftrl, w_hist_ftrl, k_hist_ftrl = simulate_ftrl(market_returns, lambda_2, gamma, lambda_1, v_tar)
loss_omd, wealth_omd, w_hist_omd, k_hist_omd = simulate_omd(market_returns, eta, lambda_1, v_tar)

# Benchmark daily returns with sparse benchmark
benchmark_daily_returns = market_returns @ w_star_sparse
benchmark_shortfalls = v_tar - benchmark_daily_returns
loss_benchmark = (np.maximum(0, benchmark_shortfalls)) ** 2
wealth_benchmark = np.cumprod(np.maximum(benchmark_daily_returns, 1e-12))

cum_loss_ftrl = np.cumsum(loss_ftrl)
cum_loss_omd = np.cumsum(loss_omd)
cum_loss_benchmark = np.cumsum(loss_benchmark)
regret_ftrl = cum_loss_ftrl - cum_loss_benchmark
regret_omd = cum_loss_omd - cum_loss_benchmark

# --------------------------
# 6. Results & K representation
# --------------------------
print("\n--- Final Metric Summary ---")
print(f"{'':30} FTRL           |   OMD            |   Benchmark")
print("-" * 70)
print(f"Total Shortfall Loss: {cum_loss_ftrl[-1]:<12.6f} | {cum_loss_omd[-1]:<12.6f} | {cum_loss_benchmark[-1]:<12.6f}")
print(f"Final Regret:         {regret_ftrl[-1]:<12.6f} | {regret_omd[-1]:<12.6f} | 0.0")
print(f"Final Portfolio Value:{wealth_ftrl[-1]:<12.6f} | {wealth_omd[-1]:<12.6f} | {wealth_benchmark[-1]:<12.6f}\n")

# Effective K statistics
def print_k_stats(k_hist, name):
    print(f"{name} effective-K (nonzero weights) stats:")
    print(f"  last K = {k_hist[-1] if k_hist.size>0 else 'N/A'}")
    print(f"  mean K = {np.mean(k_hist):.2f}")
    print(f"  median K = {np.median(k_hist):.2f}")
    print(f"  min K = {np.min(k_hist):.0f}, max K = {np.max(k_hist):.0f}\n")

print_k_stats(k_hist_ftrl, "FTRL")
print_k_stats(k_hist_omd, "OMD")

# If you want a quick view of final portfolios & nonzero indices:
print("Final nonzero indices (FTRL):", np.where(w_hist_ftrl[-1] > 1e-8)[0].tolist())
print("Final nonzero indices (OMD): ", np.where(w_hist_omd[-1] > 1e-8)[0].tolist())
