# Adversarial Audit

## Where non-stationarity harms us
- Regime shifts can invalidate feature-target relationships learned in historical windows.
- Fixed model hyperparameters may underperform as volatility and liquidity regimes change.

## Cost assumptions impact
- Linear bps + spread proxy may understate impact in stressed markets.
- Net-of-cost promotion can flip from pass to fail under modest cost increases.

## Repeated iteration overfitting risk
- Re-running backtests and selecting best variants can overfit validation windows.
- Promotion gates reduce but do not eliminate researcher degrees of freedom.

## Gate-to-failure-mode mapping
- Leakage tests + lag enforcement mitigate alignment bugs.
- Walk-forward harness mitigates single-split optimism.
- Baseline outperformance gate mitigates complexity bias.
- Calibration gate mitigates overconfident forecast distributions.
- Drawdown and turnover constraints mitigate optimizer error amplification.

## What would fool this system?
- Synthetic stationarity during development with real-world distribution break in production.
- Unmodeled jump risk and liquidity gaps that invalidate cost and volatility assumptions.
- Data vendor timestamp corrections arriving later and silently changing historical bars.

## Detection strategies
- Track live vs backtest calibration drift and rolling net alpha decay.
- Run stress scenarios with doubled costs and volatility shocks before promotion.
- Hash input datasets and compare across reruns for silent historical revisions.
