# Hybrid ExtraTrees + Ridge MT5 ONNX Strategy

This repository contains an MT5 Expert Advisor that combines two ONNX machine-learning models trained on the same scale-invariant feature set:

- **ExtraTreesClassifier**: primary signal generator.
- **RidgeClassifier**: confirmation / veto filter.

The goal is to reduce overfitting risk from the nonlinear Extra Trees model while keeping the Ridge model as a simple, more stable directional sanity check.

> This is research code. It is not financial advice and it should not be used on a live account before forward testing, spread/slippage testing, and broker-specific validation.

---

## Files

Recommended repository layout:

```text
.
├── MT5_Hybrid_ExtraTrees_Ridge_ONNX_Strategy_Scale_Invariant.mq5
├── train_mt5_extratrees_classifier_scale_invariant.py
├── train_mt5_ridge_classifier_scale_invariant.py
├── README.md
└── output folders from training
```

After training, the EA expects these ONNX files to be embedded as resources:

```text
ml_strategy_classifier_extratrees.onnx
ml_strategy_classifier_ridge.onnx
```

Place both `.onnx` files in the same MT5 folder as the `.mq5` file before compiling.

---

## Strategy idea

The hybrid decision flow is:

```text
1. Build the same 10 scale-invariant features on the last closed bar.
2. Run the Extra Trees ONNX model.
3. Run the Ridge ONNX model.
4. Extra Trees proposes BUY / SELL / FLAT.
5. Ridge confirms or vetoes the Extra Trees direction.
6. Trend filter, ATR volatility filter, and kill switch are applied.
7. If no position is open, the EA opens the filtered signal.
```

The intended architecture is:

```text
Extra Trees = entry timing / nonlinear signal generator
Ridge       = directional confirmation / anti-overfit filter
Trend EMA   = market regime filter
ATR filter  = volatility sanity filter
Kill switch = live-risk protection layer
```

---

## Scale-invariant features

Both models must be trained with exactly the same feature order:

```text
ret_1
ret_3
ret_5
ret_10
vol_10
vol_20
dist_sma_10
dist_sma_20
zscore_20
atr_pct_14
```

These features are mostly scale-invariant because they use returns, ratios, z-scores, and ATR as a percentage of price. This allows the same feature design to be tested on symbols with very different prices, such as XAGUSD and BTCUSD.

---

## Class order assumption

The EA assumes both ONNX models use this class order:

```text
0 = sell
1 = flat
2 = buy
```

This should match the Python metadata:

```json
"class_order": [-1, 0, 1]
```

Check `model_metadata.json` after training both models. If the ONNX output order differs, the EA signal mapping must be changed.

---

## Python installation

Install dependencies:

```powershell
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

Optional but useful:

```powershell
pip install onnxruntime
```

---

## Train Extra Trees

Example:

```powershell
python train_mt5_extratrees_classifier_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_et_XAGUSD_M15_h8
```

The important output is:

```text
output_et_XAGUSD_M15_h8/ml_strategy_classifier_extratrees.onnx
```

Also read:

```text
output_et_XAGUSD_M15_h8/run_in_mt5.txt
output_et_XAGUSD_M15_h8/model_metadata.json
```

Use the recommended Extra Trees thresholds from `run_in_mt5.txt` as starting EA inputs:

```text
InpETEntryProbThreshold
InpETMinProbGap
InpMaxBarsInTrade
```

---

## Train Ridge

Example:

```powershell
python train_mt5_ridge_classifier_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_ridge_XAGUSD_M15_h8
```

The important output is:

```text
output_ridge_XAGUSD_M15_h8/ml_strategy_classifier_ridge.onnx
```

Also read:

```text
output_ridge_XAGUSD_M15_h8/run_in_mt5.txt
output_ridge_XAGUSD_M15_h8/model_metadata.json
```

Use the recommended Ridge thresholds only for Ridge-only tests. For the hybrid, start with soft confirmation first.

---

## Copy files to MT5

Copy the EA:

```text
MT5_Hybrid_ExtraTrees_Ridge_ONNX_Strategy_Scale_Invariant.mq5
```

Copy both ONNX files next to it:

```text
ml_strategy_classifier_extratrees.onnx
ml_strategy_classifier_ridge.onnx
```

Then compile the `.mq5` file in MetaEditor.

The EA embeds the models using:

```cpp
#resource "ml_strategy_classifier_extratrees.onnx" as uchar ExtExtraTreesModel[]
#resource "ml_strategy_classifier_ridge.onnx"      as uchar ExtRidgeModel[]
```

---

## Hybrid modes

The EA exposes this input:

```cpp
input HybridMode InpHybridMode = HYBRID_ET_WITH_RIDGE_SOFT_CONFIRM;
```

Available modes:

| Mode | Value | Meaning |
|---|---:|---|
| `HYBRID_ET_ONLY` | 0 | Use only Extra Trees. Useful baseline. |
| `HYBRID_RIDGE_ONLY` | 1 | Use only Ridge. Useful baseline. |
| `HYBRID_ET_WITH_RIDGE_CONFIRM` | 2 | Extra Trees leads, Ridge must hard-confirm. |
| `HYBRID_ET_WITH_RIDGE_SOFT_CONFIRM` | 3 | Extra Trees leads, Ridge must not disagree directionally. Recommended first hybrid test. |

---

## Recommended first settings

Start with the Extra Trees parameters that worked best in your separate Extra Trees test.

Then test the hybrid with:

```text
InpHybridMode = HYBRID_ET_WITH_RIDGE_SOFT_CONFIRM

InpETEntryProbThreshold = value from Extra Trees run_in_mt5.txt
InpETMinProbGap         = value from Extra Trees run_in_mt5.txt

InpRidgeEntryScoreThreshold = 0.00
InpRidgeMinScoreGap         = 0.10
InpRidgeMinDirectionalGap   = 0.00

InpUseTrendFilter        = true
InpTrendMAPeriod         = 100
InpTrendRequireSlope     = true

InpUseAtrVolFilter       = true
InpAtrMinPercentile      = 0.25
InpAtrMaxPercentile      = 0.85

InpUseKillSwitch         = false for first backtests
```

Why soft confirmation first?

Because Extra Trees may already produce a small number of trades. Hard Ridge confirmation can reduce the trade count too much.

---

## Hard vs soft Ridge confirmation

### Hard confirmation

For BUY:

```text
ridge_buy_score > ridge_sell_score
ridge_buy_score > ridge_flat_score
ridge_buy_score - max(ridge_sell_score, ridge_flat_score) >= InpRidgeMinScoreGap
```

For SELL:

```text
ridge_sell_score > ridge_buy_score
ridge_sell_score > ridge_flat_score
ridge_sell_score - max(ridge_buy_score, ridge_flat_score) >= InpRidgeMinScoreGap
```

This is stricter and can improve trade quality, but it may reduce trade count too much.

### Soft confirmation

For BUY:

```text
ridge_buy_score - ridge_sell_score >= InpRidgeMinDirectionalGap
```

For SELL:

```text
ridge_sell_score - ridge_buy_score >= InpRidgeMinDirectionalGap
```

This ignores the Ridge flat score and checks only whether Ridge directionally agrees with Extra Trees.

This is usually the better first test.

---

## Recommended optimization workflow

Do not optimize all parameters at once.

### Stage 1: baselines

Run these modes with fixed non-ML filters:

```text
HYBRID_ET_ONLY
HYBRID_RIDGE_ONLY
HYBRID_ET_WITH_RIDGE_SOFT_CONFIRM
HYBRID_ET_WITH_RIDGE_CONFIRM
```

Compare:

```text
profit factor
net profit
equity drawdown %
recovery factor
number of trades
average trade
```

### Stage 2: Ridge filter only

Freeze Extra Trees thresholds. Optimize only:

```text
InpHybridMode
InpRidgeMinDirectionalGap
InpRidgeMinScoreGap
```

Suggested ranges:

```text
InpRidgeMinDirectionalGap: 0.00, 0.02, 0.05, 0.10, 0.15
InpRidgeMinScoreGap:       0.00, 0.05, 0.10, 0.15, 0.20
```

### Stage 3: Extra Trees thresholds

Only after Stage 2, optimize:

```text
InpETEntryProbThreshold
InpETMinProbGap
```

Suggested ranges:

```text
InpETEntryProbThreshold: 0.45 to 0.75
InpETMinProbGap:         0.00 to 0.30
```

### Stage 4: filters

Only after the model thresholds look stable, optimize:

```text
InpTrendMAPeriod
InpTrendRequireSlope
InpAtrMinPercentile
InpAtrMaxPercentile
InpStopAtrMultiple
InpTakeAtrMultiple
InpMaxBarsInTrade
```

---

## What would count as a good result?

For XAGUSD M15, a better hybrid should ideally improve robustness, not just net profit:

```text
Profit Factor:      >= 1.6 to 1.8
Equity DD:          lower than Extra Trees only
Recovery Factor:    > 2.0 preferred
Trades:             not too low; preferably 40+
```

For BTCUSD M15:

```text
Profit Factor:      >= 1.6
Equity DD:          < 10% to 15% preferred
Trades:             preferably 40+
```

A result with very high profit factor but only 10 to 20 trades is not enough evidence. Treat it as interesting, not proven.

---

## Important testing rules

1. Use an out-of-sample period that was not used for training.
2. Do not select only the single best optimizer pass.
3. Prefer broad stable parameter zones over isolated peaks.
4. Test both XAGUSD and BTCUSD.
5. Test with higher spread and slippage.
6. Forward-test on demo before real money.
7. Re-check class order after retraining.
8. Recompile the EA after replacing embedded ONNX files.

---

## Notes about Ridge scores

Ridge outputs are raw decision scores, not probabilities.

Therefore:

```text
Extra Trees thresholds use probabilities.
Ridge thresholds use score differences.
```

Do not compare Ridge scores to probability thresholds such as `0.60`.

---

## Common problems

### EA fails to initialize

Check that both files are next to the EA before compilation:

```text
ml_strategy_classifier_extratrees.onnx
ml_strategy_classifier_ridge.onnx
```

Then recompile.

### No trades

Try:

```text
InpHybridMode = HYBRID_ET_ONLY
```

If trades appear, the Ridge filter is too strict.

Then try:

```text
InpHybridMode = HYBRID_ET_WITH_RIDGE_SOFT_CONFIRM
InpRidgeMinDirectionalGap = 0.00
```

### Too few trades

Lower:

```text
InpETEntryProbThreshold
InpETMinProbGap
InpRidgeMinDirectionalGap
InpRidgeMinScoreGap
```

Or use soft confirmation instead of hard confirmation.

### Too many bad trades

Increase:

```text
InpETEntryProbThreshold
InpETMinProbGap
```

Then test hard confirmation:

```text
InpHybridMode = HYBRID_ET_WITH_RIDGE_CONFIRM
```

---

## Disclaimer

This project is for research and education. Backtest profitability does not guarantee future performance. Real trading can be affected by spread, slippage, broker execution, liquidity, symbol specification changes, and regime changes.
