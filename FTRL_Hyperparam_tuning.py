import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
import os
import itertools
import traceback

# --------------------------
# User hyperparameters
# --------------------------
v_tar_list = [1.02, 1.03, 1.04, 1.05]
lambda_2_list = [50.0, 100.0, 150.0, 200.0]

# Fixed global parameters (can be changed)
K = 15             # K-set constraint
gamma = 0.999      # Forgetting factor
eta = 0.1          # Learning rate for OMD

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
    res = minimize(objective_func, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-9, 'maxiter':1000})
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

# --------------------------
# 3. FTRL Algorithm
# --------------------------
def simulate_ftrl(market_returns, K, lambda_2, gamma, v_tar=1.01, verbose=True):
    """FTRL with entropy regularization + forgetting."""
    if verbose:
        print(f"🚀 Running FTRL (λ={lambda_2}, γ={gamma}, v_tar={v_tar}) ...")
    start = time.time()
    T, N = market_returns.shape
    w_t = np.ones(N) / N
    losses, wealth, weights = [], [1.0], [w_t.copy()]

    B_t = np.zeros((N, N))
    v_t = np.zeros(N)

    for t in range(T):
        r_t = market_returns[t]
        daily = float(w_t @ r_t)
        # Avoid degenerate daily returns
        if daily <= 0:
            daily = 1e-12

        shortfall = v_tar - daily
        losses.append((np.maximum(0, shortfall)) ** 2)
        wealth.append(wealth[-1] * daily)

        # Forgetting accumulation
        B_t = gamma * B_t + np.outer(r_t, r_t)
        v_t = gamma * v_t + r_t

        # Objective
        def ftrl_obj(w):
            w_safe = np.maximum(w, 1e-12)
            quad = float(w @ B_t @ w)
            lin = -2 * v_tar * float(v_t @ w)
            reg = lambda_2 * np.sum(w_safe * np.log(w_safe))
            return quad + lin + reg

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(N)]
        res = minimize(ftrl_obj, w_t, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-9, 'maxiter':500})

        if res.success:
            w_dense = res.x
        else:
            w_dense = w_t

        w_t = project_to_k_set(w_dense, K)
        weights.append(w_t.copy())

        # if (t + 1) % 50 == 0 and verbose:
        #     print(f"  Step {t+1}/{T} complete...")

    if verbose:
        print(f"✅ FTRL done in {time.time() - start:.2f}s\n")
    return np.array(losses), np.array(wealth[1:]), np.array(weights[:-1])

# --------------------------
# 4. OMD Algorithm
# --------------------------
def simulate_omd(market_returns, K, eta, v_tar=1.01, verbose=True):
    """OMD (Exponentiated Gradient) with asymmetric shortfall loss."""
    if verbose:
        print(f"🚀 Running OMD (η={eta}, v_tar={v_tar}) ...")
    start = time.time()
    T, N = market_returns.shape
    w_t = np.ones(N) / N
    losses, wealth, weights = [], [1.0], [w_t.copy()]

    for t in range(T):
        r_t = market_returns[t]
        daily = float(w_t @ r_t)
        if daily <= 0:
            daily = 1e-12

        shortfall = v_tar - daily
        losses.append((np.maximum(0, shortfall)) ** 2)
        wealth.append(wealth[-1] * daily)

        # Gradient (only penalize shortfall)
        grad = -2 * shortfall * r_t if shortfall > 0 else np.zeros(N)

        # Exponentiated gradient update
        # Clip exponent to avoid overflow
        update = np.exp(np.clip(-eta * grad, -50, 50))
        w_t = w_t * update
        sumw = np.sum(w_t)
        if sumw <= 0:
            w_t = np.ones_like(w_t) / len(w_t)
        else:
            w_t /= sumw

        w_t = project_to_k_set(w_t, K)
        weights.append(w_t.copy())

    if verbose:
        print(f"✅ OMD done in {time.time() - start:.2f}s\n")
    return np.array(losses), np.array(wealth[1:]), np.array(weights[:-1])

# --------------------------
# 5. Sweep runner
# --------------------------
def run_sweep(market_returns, v_tar_list, lambda_2_list, K, gamma, eta):
    T, N = market_returns.shape
    combo_index = 0
    total_combos = len(v_tar_list) * len(lambda_2_list)

    for v_tar_val, lambda_val in itertools.product(v_tar_list, lambda_2_list):
        combo_index += 1
        header = f"=== Run {combo_index}/{total_combos}: v_tar={v_tar_val}, lambda_2={lambda_val} ==="
        print("\n" + "="*len(header))
        print(header)
        print("="*len(header) + "\n")

        try:
            # Benchmark (depends on v_tar)
            w_star_k, benchmark_total_loss = find_best_fixed_k_sparse_portfolio(market_returns, K, v_tar=v_tar_val)
            # Simulate algorithms
            loss_ftrl, wealth_ftrl, w_hist_ftrl = simulate_ftrl(market_returns, K, lambda_val, gamma, v_tar=v_tar_val, verbose=True)
            loss_omd, wealth_omd, w_hist_omd = simulate_omd(market_returns, K, eta, v_tar=v_tar_val, verbose=True)

            # Benchmark arrays
            benchmark_daily_returns = market_returns @ w_star_k
            benchmark_daily_returns = np.maximum(benchmark_daily_returns, 1e-12)
            benchmark_shortfalls = v_tar_val - benchmark_daily_returns
            loss_benchmark = (np.maximum(0, benchmark_shortfalls)) ** 2
            wealth_benchmark = np.cumprod(benchmark_daily_returns)

            # Cumulative losses and regrets
            cum_loss_ftrl = np.cumsum(loss_ftrl)
            cum_loss_omd = np.cumsum(loss_omd)
            cum_loss_benchmark = np.cumsum(loss_benchmark)
            regret_ftrl = cum_loss_ftrl - cum_loss_benchmark
            regret_omd = cum_loss_omd - cum_loss_benchmark

            # Final metrics
            final_cum_loss_ftrl = float(cum_loss_ftrl[-1])
            final_cum_loss_omd = float(cum_loss_omd[-1])
            final_cum_loss_bench = float(cum_loss_benchmark[-1])

            final_regret_ftrl = float(regret_ftrl[-1])
            final_regret_omd = float(regret_omd[-1])

            final_wealth_ftrl = float(wealth_ftrl[-1]) if wealth_ftrl.size else np.nan
            final_wealth_omd = float(wealth_omd[-1]) if wealth_omd.size else np.nan
            final_wealth_bench = float(wealth_benchmark[-1]) if wealth_benchmark.size else np.nan

            # Print summary table for this run
            print("\n--- Final Metric Summary ---")
            print(f"{'':30} FTRL           |   OMD            |   Benchmark")
            print("-" * 70)
            print(f"Total Shortfall Loss: {final_cum_loss_ftrl:<12.6f} | {final_cum_loss_omd:<12.6f} | {final_cum_loss_bench:<12.6f}")
            print(f"Final Regret:         {final_regret_ftrl:<12.6f} | {final_regret_omd:<12.6f} | 0.0")
            print(f"Final Portfolio Value:{final_wealth_ftrl:<12.6f} | {final_wealth_omd:<12.6f} | {final_wealth_bench:<12.6f}\n")

        except Exception as e:
            print("❌ Exception during run:")
            traceback.print_exc()
            print("Continuing to next hyperparameter combination.\n")
            continue

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Load data once
    try:
        market_returns = load_market_data()
    except Exception as e:
        print("Fatal error loading market data:")
        traceback.print_exc()
        raise SystemExit(1)

    T, N = market_returns.shape
    print(f"Data loaded: {T} time steps, {N} assets.\n")

    run_sweep(market_returns, v_tar_list, lambda_2_list, K, gamma, eta)

    print("\nAll runs complete.")
