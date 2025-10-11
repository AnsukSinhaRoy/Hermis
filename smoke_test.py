from run_experiment import runner_func
import yaml
cfg = yaml.safe_load(open("configs/newconfig.yaml"))
res = runner_func(cfg)
print("NAV head:", res.nav.head())
print("Weights at first rebalance (if any):", res.weights.dropna(how='all').head())
print("Trades head:", res.trades.head())
