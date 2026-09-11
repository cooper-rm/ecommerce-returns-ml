# Predicting e-commerce returns, and pricing coverage against them

**Pricing each order off its calibrated return probability doesn't just change prices — it prices adverse selection away.** Under a flat rate, the shoppers who buy returns coverage are **+0.123 riskier** than the population, because the offer is a bargain for anyone who knows they'll return. Quote each shopper their own expected cost and that collapses to **+0.002**: attach lands between 19.8% and 20.3% across *all ten* risk deciles, and expected claim per policy falls 14%.

An order-level return-prediction model on 2.3M real order lines, and the pricing rule it produces.

📄 **[Memo (PDF)](memo/returns_ml_memo.pdf)** — the short version, with the charts. Source is [`memo/returns_ml_memo.html`](memo/returns_ml_memo.html); edit it and run `python memo/build_memo.py` to re-render.

![Left: attach rate by risk decile, climbing steeply under a flat rate and almost perfectly flat under the price card. Right: share of policies sold by decile, concentrated in high-risk deciles under the flat rate and evenly spread under the card.](figures/selection_flip.png)

## Key results

| | |
|---|---|
| **Model** | 0.8526 AUC (vs 0.7959 raw-attribute baseline), log loss 0.4527 |
| **Calibration** | ECE 0.0047 — top decile predicts 0.993 against actual 0.994 |
| **Model choice is irrelevant** | XGBoost 0.8528, LightGBM 0.8526, predictions correlate 0.998 |
| **Features carry it** | Logistic regression reaches 0.8348 — within 0.018 of the boosters |
| **Strongest single signal** | Same article ordered in 2+ sizes → **96%** return rate vs 54% |
| **Selection effect** | Flat rate **+0.123** riskier than population → price card **+0.002** |
| **Claim cost per policy** | $6.24 flat → **$5.37** with the card (−14%) |
| **Baseline EV** | A 45% margin on a $100M book = **$45M**, at every return rate |

## The pricing rule

```
price = (probability of return × cost of a return) ÷ (1 − margin)
```

Because the probability is calibrated, this earns the target **by construction on every order** — per-order margin is `(p − c)/p = margin` regardless of what the claim costs. No optimiser, no dependence on any demand assumption. At $8 per return:

| Risk decile | Return probability | Expected claim | Price at 45% | Price at 60% |
|---|---|---|---|---|
| 1 (safest) | 0.15 | $1.89 | **$3.44** | $4.73 |
| 5 | 0.60 | $5.09 | $9.25 | $12.72 |
| 10 (riskiest) | 0.99 | $7.95 | **$14.45** | $19.87 |

The single flat price earning the same 45% margin is **$11.34**. The card sells to **20.1% of all shoppers against the flat rate's 18.0%** at identical profitability — and, more importantly, that uptake is near-uniform across risk deciles (19.8–20.3%) rather than concentrated in the riskiest. Same margin, more customers, and a far healthier book.

**A high quote carries no risk.** A shopper who declines pays their own return shipping, so the platform has no revenue *and* no claims from them. Quoting the riskiest decile $14.45 costs forgone attach, not money.

## Scaled to $100M

The EV is invariant — every order carries exactly the target margin, so a $100M coverage book returns **$45M at a 45% margin regardless of the book's return rate**. What changes is the price range required to get there:

| Return rate | Mean claim | Flat price at 45% | Decile range at 45% |
|---|---|---|---|
| 10% | $1.52 | $3.26 | $1.47 – $8.92 |
| 20% | $2.24 | $5.36 | $1.53 – $12.28 |
| 30% | $2.96 | $7.31 | $1.66 – $13.58 |
| 63% (this book) | $5.35 | $11.34 | $3.44 – $14.45 |

## How it works

**Data.** Data Mining Cup 2016 — 2,325,165 order lines from an anonymized European fashion retailer (Jan 2014 – Sep 2015), collapsed to 738,698 orders labeled `any_return`. Picked over Olist or H&M because it has a **real return label**, not a review-score proxy. This book returns 63.5% of orders, the global high-water mark, which is why the pricing is swept across lower rates.

**Leakage discipline.** Validation is a **time split** at 2015-07-01, not random K-fold. Every history feature is past-only via `src.features.past_rate`, which freezes each key's history at the start of the day so an order never sees itself or its same-day siblings — verified against a hand-checked example including a same-day tie.

**Models.** One fixed, untuned LightGBM throughout, with logistic regression and XGBoost as reference points. The study varies the data, not the hyperparameters.

**Pricing.** Claim costs use **actual** validation outcomes. The price card needs no demand assumption — it follows from the calibrated probability and the margin. Attach figures do depend on a demand curve, anchored at 35% attach for a $5 price and stated in one cell.

## Repository

```
notebooks/     run in order
  01           download DMC 2016 order lines into data/raw
  02           auto-EDA profile of the raw order lines
  03           order table and raw-attribute baseline
  04           basket features: bracketing, discount depth, mix
  05           customer history features (past-only)
  06           article / size / product-group rates (past-only)
  07           model: LightGBM, reliability diagram, precision@k
  08           pricing: sweep, strategies, adverse selection
  09           break-even frontier over return rate
  10           segment analysis: where the model wins and breaks
  11           interpretation: SHAP and permutation importance
  12           manual EDA: the four cuts, clustering, class separation
  13           logistic regression: what moves the needle, in odds ratios
  14           XGBoost: is the score model-limited or data-limited?
  15           the price card, the selection flip, and scaling to $100M
  16           further exploration: a heterogeneous multi-merchant portfolio
src/           data paths, past-only rates, time-split and OOF helpers
figures/       charts used in the memo and this README
memo/          memo (editable HTML source + PDF + build script)
data/          raw data and local outputs (gitignored)
eda/           generated auto-EDA reports (gitignored)
```

Every code cell across all 16 notebooks carries explanatory markdown above it.

## Getting started

```bash
# 1. create and activate the conda environment (Python 3.11 + the data stack)
conda env create -f environment.yml
conda activate returns-ml

# 2. register the Jupyter kernel (so it is selectable in Jupyter / VS Code)
python -m ipykernel install --user --name returns-ml \
  --display-name "Python (returns-ml)"
```

Run `notebooks/01_download.ipynb` to fetch the data into `data/raw/` (not committed), then run the notebooks in order. A pip `requirements.txt` is provided as an alternative to conda.

## Honest limitations

- **Demand is assumed, not measured.** The dataset has no opt-in or coverage data. The price card does not depend on it, but every attach figure does.
- **63.5% is an outlier return rate.** US brands run 15–30%, which is why notebooks 09 and 15 sweep the rate rather than reporting one number.
- **Margin here is gross margin on revenue**, `(price − cost)/price`. The markup-on-cost convention gives lower prices for the same headline percentage — $11.53 rather than $14.45 at "45%".
- **The top of the range is optically large.** $19.87 on a $106 order is ~19% of cart value. It carries no financial risk, but it is a merchant conversation.
- **Two payment segments show calibration drift** past 0.04 ECE and would need per-segment recalibration before their prices could be trusted.
- Amounts shown in dollars; the source retailer is euro-denominated, treated 1:1 with no FX applied.

## References

- **Dataset**: Data Mining Cup 2016 (prudsys AG), mirrored in [ISU-DMC/dmc2016](https://github.com/ISU-DMC/dmc2016). Downloaded at runtime, never committed.
