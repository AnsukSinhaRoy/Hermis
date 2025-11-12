
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import os

# --- Plot settings ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# --- Simulation Parameters ---
K = 15             # K-set constraint
v_tar = 1.02      # Target price-relative (2% target return)
lambda_2 = 100.0   # Entropy regularization
gamma = 0.999     # Forgetting factor
eta = 0.1         # Learning rate for OMD

# --------------------------
# 1. Load and Process Data
# --------------------------
def load_market_data(parquet_path='data/processed/prices_1D.parquet'):
    """
    Load price data directly from prices_1D.parquet and
    convert it into price relatives (returns).
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"❌ Could not find {parquet_path}. "
            "Please ensure your preprocessing pipeline has generated it."
        )

    print(f"📂 Loading market prices from {parquet_path} ...")
    prices_df = pd.read_parquet(parquet_path)
    if prices_df.empty:
        raise ValueError("Loaded parquet file is empty.")

    # Convert Prices → Price Relatives
    returns_df = prices_df / prices_df.shift(1)
    returns_df = returns_df.iloc[1:]  # drop NaN first row

    # Clean anomalies
    returns_df.replace([np.inf, -np.inf], 1.0, inplace=True)
    returns_df.fillna(1.0, inplace=True)
    returns_df[returns_df <= 0] = 1e-12  # tiny positive for log stability

    market_returns = returns_df.values
    print(f"✅ Converted to price relatives. Shape: {market_returns.shape}")
    return market_returns

# Load data directly from prices_1D.parquet
market_returns = load_market_data()
T, N = market_returns.shape
print(f"Data loaded: {T} time steps, {N} assets.\n")

# --------------------------
# 2. Helper Functions
# --------------------------
def project_to_k_set(w, k):
    """Project vector w onto top-k sparse simplex."""
    w = np.array(w, dtype=float)
    w_sparse = np.zeros_like(w)
    top_k = np.argsort(w)[-k:]
    w_sparse[top_k] = w[top_k]
    s = np.sum(w_sparse)
    if s > 0:
        w_sparse /= s
    else:
        w_sparse[top_k] = 1.0 / k
    return w_sparse

def find_best_fixed_portfolio(objective_func, N):
    """Helper: find best dense portfolio minimizing objective_func."""
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(N))
    w0 = np.ones(N) / N
    res = minimize(objective_func, w0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        print("⚠️ Warning: optimization failed, using uniform portfolio.")
        return w0
    return res.x

def find_best_fixed_k_sparse_portfolio(market_returns, K, v_tar=1.01):
    """Compute best fixed K-sparse portfolio in hindsight (asymmetric shortfall loss)."""
    print("🔹 Computing best fixed K-sparse benchmark portfolio ...")
    T, N = market_returns.shape

    def total_asymmetric_loss(w):
        daily_returns = market_returns @ w
        shortfalls = v_tar - daily_returns
        losses = (np.maximum(0, shortfalls)) ** 2
        return np.sum(losses)

    w_dense = find_best_fixed_portfolio(total_asymmetric_loss, N)
    w_k = project_to_k_set(w_dense, K)
    final_loss = total_asymmetric_loss(w_k)

    print(f"✅ Benchmark computed. Final Loss = {final_loss:.6f}")
    print(f"Top-{K} assets (indices): {np.where(w_k > 0)[0].tolist()}\n")
    return w_k, final_loss

# Benchmark
w_star_k, benchmark_total_loss = find_best_fixed_k_sparse_portfolio(market_returns, K, v_tar)

# --------------------------
# 3. FTRL Algorithm
# --------------------------
def simulate_ftrl(market_returns, K, lambda_2, gamma, v_tar=1.01):
    """FTRL with entropy regularization + forgetting."""
    print(f"🚀 Running FTRL (λ={lambda_2}, γ={gamma}) ...")
    start = time.time()
    T, N = market_returns.shape
    w_t = np.ones(N) / N
    losses, wealth, weights = [], [1.0], [w_t.copy()]

    B_t = np.zeros((N, N))
    v_t = np.zeros(N)

    for t in range(T):
        r_t = market_returns[t]
        daily = w_t @ r_t
        shortfall = v_tar - daily
        losses.append((np.maximum(0, shortfall)) ** 2)
        wealth.append(wealth[-1] * daily)

        # Forgetting accumulation
        B_t = gamma * B_t + np.outer(r_t, r_t)
        v_t = gamma * v_t + r_t

        # Objective
        def ftrl_obj(w):
            w_safe = np.maximum(w, 1e-12)
            quad = w @ B_t @ w
            lin = -2 * v_tar * (v_t @ w)
            reg = lambda_2 * np.sum(w_safe * np.log(w_safe))
            return quad + lin + reg

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(N)]
        res = minimize(ftrl_obj, w_t, method='SLSQP', bounds=bounds, constraints=cons)

        if res.success:
            w_dense = res.x
        else:
            w_dense = w_t

        w_t = project_to_k_set(w_dense, K)
        weights.append(w_t.copy())

        if (t + 1) % 50 == 0:
            print(f"  Step {t+1}/{T} complete...")

    print(f"✅ FTRL done in {time.time() - start:.2f}s\n")
    return np.array(losses), np.array(wealth[1:]), np.array(weights[:-1])

# --------------------------
# 4. OMD Algorithm
# --------------------------
def simulate_omd(market_returns, K, eta, v_tar=1.01):
    """OMD (Exponentiated Gradient) with asymmetric shortfall loss."""
    print(f"🚀 Running OMD (η={eta}) ...")
    start = time.time()
    T, N = market_returns.shape
    w_t = np.ones(N) / N
    losses, wealth, weights = [], [1.0], [w_t.copy()]

    for t in range(T):
        r_t = market_returns[t]
        daily = w_t @ r_t
        shortfall = v_tar - daily
        losses.append((np.maximum(0, shortfall)) ** 2)
        wealth.append(wealth[-1] * daily)

        # Gradient (only penalize shortfall)
        grad = -2 * shortfall * r_t if shortfall > 0 else np.zeros(N)

        # Exponentiated gradient update
        w_t = w_t * np.exp(-eta * grad)
        w_t /= np.sum(w_t)
        w_t = project_to_k_set(w_t, K)
        weights.append(w_t.copy())

    print(f"✅ OMD done in {time.time() - start:.2f}s\n")
    return np.array(losses), np.array(wealth[1:]), np.array(weights[:-1])

# --------------------------
# 5. Run Simulations
# --------------------------
loss_ftrl, wealth_ftrl, w_hist_ftrl = simulate_ftrl(market_returns, K, lambda_2, gamma, v_tar)
loss_omd, wealth_omd, w_hist_omd = simulate_omd(market_returns, K, eta, v_tar)

# Benchmark
benchmark_daily_returns = market_returns @ w_star_k
benchmark_shortfalls = v_tar - benchmark_daily_returns
loss_benchmark = (np.maximum(0, benchmark_shortfalls)) ** 2
wealth_benchmark = np.cumprod(benchmark_daily_returns)

cum_loss_ftrl = np.cumsum(loss_ftrl)
cum_loss_omd = np.cumsum(loss_omd)
cum_loss_benchmark = np.cumsum(loss_benchmark)
regret_ftrl = cum_loss_ftrl - cum_loss_benchmark
regret_omd = cum_loss_omd - cum_loss_benchmark

# --------------------------
# 6. Plot Results
# --------------------------
print("📊 Plotting results...")

plt.figure()
plt.plot(cum_loss_ftrl, label=f'FTRL (λ={lambda_2}, γ={gamma})', linewidth=2.5)
plt.plot(cum_loss_omd, label=f'OMD (η={eta})', linestyle='--')
plt.plot(cum_loss_benchmark, label='Benchmark', linestyle=':', color='black')
plt.title('Cumulative Shortfall Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(regret_ftrl, label='FTRL', linewidth=2.5)
plt.plot(regret_omd, label='OMD', linestyle='--')
plt.title('Cumulative Regret')
plt.legend()
plt.show()

plt.figure()
plt.plot(wealth_ftrl, label='FTRL', linewidth=2.5)
plt.plot(wealth_omd, label='OMD', linestyle='--')
plt.plot(wealth_benchmark, label='Benchmark', linestyle=':', color='black')
plt.yscale('log')
plt.title('Portfolio Wealth Over Time')
plt.legend()
plt.show()

if w_hist_ftrl is not None:
    plt.figure()
    plt.stackplot(range(T), w_hist_ftrl.T)
    plt.title('FTRL Portfolio Allocation Over Time')
    plt.xlabel('Time')
    plt.ylabel('Weights')
    plt.show()

# --------------------------
# 7. Summary
# --------------------------
print("\n--- Final Metric Summary ---")
print(f"{'':30} FTRL           |   OMD            |   Benchmark")
print("-" * 70)
print(f"Total Shortfall Loss: {cum_loss_ftrl[-1]:<10.4f} | {cum_loss_omd[-1]:<10.4f} | {cum_loss_benchmark[-1]:<10.4f}")
print(f"Final Regret:         {regret_ftrl[-1]:<10.4f} | {regret_omd[-1]:<10.4f} | 0.0")
print(f"Final Portfolio Value:{wealth_ftrl[-1]:<10.4f} | {wealth_omd[-1]:<10.4f} | {wealth_benchmark[-1]:<10.4f}")
