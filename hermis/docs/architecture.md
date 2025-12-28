# Hermis architecture

This repo intentionally supports two simulation "modes":

## 1) Portfolio rebalance backtesting (daily / bar-based)

This is what the project does today.

- Input data is typically **1D bars** (daily) in a wide `DataFrame` (index = datetime, columns = tickers).
- The simulator runs in **rebalance steps** (e.g., every day / every week):
  - estimate `mu` and `Sigma` from a rolling window
  - call an optimizer / strategy to produce weights
  - apply transaction costs and compute portfolio value

This mode is kept **simple and vectorized** (pandas / numpy), and is ideal for research and quick iteration.

## 2) Event-driven simulation (1s / derivatives / execution realism)

For derivatives and high-frequency data, a pure DataFrame-based approach tends to break down:

- memory blowups (1s across many symbols)
- microstructure realism (partial fills, slippage, queues)
- RL agents need **step-wise interaction** (observation → action → reward)

Hermis therefore has a **separate event-driven skeleton** under `hermis/sim/event_driven/`.

Key idea:
- daily portfolio backtests stay **bar/rebalance-driven**
- intraday derivatives / RL move to **event-driven**, using a `DataFeed` that yields `Bar` events

You can share some preprocessing and indicator code, but the engine loops and state handling should remain separate.

## Practical guidance

- Keep your *data loaders* frequency-aware (daily vs 1s).
- Keep *strategies/policies* decoupled from the engine:
  - portfolio mode: `optimizer(mu, Sigma, **context) -> weights`
  - event-driven/RL mode: `policy.act(obs) -> action`

