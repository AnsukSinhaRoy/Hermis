# smoke_test_backtest.py
import pandas as pd
import numpy as np

# Import the necessary components from your project files
from portfolio_sim.optimizer import mv_reg_optimize
from portfolio_sim.backtest import run_backtest
from portfolio_sim.data import generate_synthetic_prices, cov_matrix

def smoke_test():
    """
    A self-contained smoke test for the backtesting engine.
    It performs the following steps:
    1. Generates a small, reproducible synthetic price dataset.
    2. Defines simple estimators for expected returns and covariance.
    3. Defines a simple optimizer function.
    4. Sets up a configuration dictionary.
    5. Runs the backtest.
    6. Prints key results for manual verification.
    """
    print("--- Starting Backtest Smoke Test ---")

    # 1. Generate synthetic data
    # Use a small number of assets and a short time period for a quick test
    # A fixed seed ensures the data is the same every time you run the test
    prices = generate_synthetic_prices(
        n_assets=4, 
        start="2021-01-01", 
        end="2021-12-31", 
        seed=42
    )
    print(f"Generated synthetic prices for {prices.shape[1]} assets from {prices.index.min()} to {prices.index.max()}.")

    # 2. Define estimators
    # For simplicity, we'll use annualized mean log returns as the expected return
    def simple_expected_return_estimator(price_history: pd.DataFrame) -> pd.Series:
        log_returns = np.log(price_history).diff()
        # Annualize the mean return (assuming daily data)
        return log_returns.mean() * 252

    # We can use your cov_matrix function directly
    def simple_cov_estimator(price_history: pd.DataFrame) -> pd.DataFrame:
        # Annualize the covariance matrix
        return cov_matrix(price_history, method='log') * 252

    # 3. Define the optimizer function
    # The optimizer function passed to the backtest needs to match the expected signature
    def optimizer_func(expected_returns: pd.Series, cov: pd.DataFrame):
        # We can directly call one of your optimizer functions
        return mv_reg_optimize(expected_returns, cov, long_only=True)

    # 4. Set up the configuration
    config = {
        'rebalance': 'monthly', # Rebalance at the start of each month
        'transaction_costs': {
            'proportional': 0.001 # 0.1% or 10 bps transaction cost
        }
    }
    print(f"Backtest configuration: Rebalance '{config['rebalance']}', TC: {config['transaction_costs']['proportional']:.4f}")

    # 5. Run the backtest
    print("\n--- Running Backtest ---")
    result = run_backtest(
        prices=prices,
        expected_return_estimator=simple_expected_return_estimator,
        cov_estimator=simple_cov_estimator,
        optimizer_func=optimizer_func,
        config=config
    )
    print("--- Backtest Finished ---\n")

    # 6. Print key results for verification
    print("--- Verification ---")
    
    # Check NAV Series
    print("\n1. Net Asset Value (NAV):")
    print("Head:")
    print(result.nav.head())
    print("\nTail:")
    print(result.nav.tail())
    # You should expect NAV to start at 1.0.
    if result.nav.iloc[0] == 1.0:
        print("✅ NAV starts at 1.0, as expected.")
    else:
        print(f"❌ WARNING: NAV starts at {result.nav.iloc[0]}, expected 1.0.")

    # Check Weights DataFrame
    print("\n2. Portfolio Weights (last 5 days):")
    print(result.weights.tail())
    last_weights_sum = result.weights.iloc[-1].sum()
    print(f"Sum of weights on the last day: {last_weights_sum:.4f}")
    # The sum should be very close to 1.0.
    if np.isclose(last_weights_sum, 1.0):
        print("✅ Sum of last weights is approximately 1.0.")
    else:
        print("❌ WARNING: Sum of last weights is not 1.0.")

    # Check Trades DataFrame
    print("\n3. Trades Log:")
    print(result.trades)
    # Check if the number of trades matches the number of rebalances.
    # Check 'opt_status' column for any failures.
    if not result.trades.empty and "ok" in result.trades['opt_status'].iloc[0]:
         print("✅ First trade optimization status is 'ok'.")
    else:
         print("❌ WARNING: Check optimizer status in trades log.")

    print("\n--- Smoke Test Complete ---")


if __name__ == "__main__":
    smoke_test()