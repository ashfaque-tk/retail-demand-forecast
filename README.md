# Retail Demand Forecasting: M5 Walmart Dataset

A leak-free, recursive time-series forecasting pipeline built on the M5 Forecasting (Walmart) competition dataset, deployed as a FastAPI endpoint with quantile-based uncertainty intervals for inventory decision-making.

Scope: FOODS/HOBBIES/HOUSEHOLD items, store CA_1. Single-store scope was a deliberate decision made after full-dataset feature engineering (~59M rows) crashed on available hardware.
## Metric Used: WRMSSE (Weighted Root Mean Scaled Squared Error)

## What it does
Recursive, leak-free 28-day demand forecasting for CA_1 (M5 dataset) using LightGBM,
with 10th/90th percentile intervals for inventory decisions. Served via FastAPI.

## Notes
- WRMSSE ~0.89 (recursive, leak-free). An earlier version scored ~0.85 due to a leakage bug in lag/rolling feature computation — fixed by ensuring lag features are only ever built from real or previously predicted history, never future
  ground truth. Scoped to one store (CA_1) due to local memory constraints.

## Next steps
- Direct (non-recursive) model for a fair head-to-head comparison
- Scale to additional stores
