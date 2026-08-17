# Optimizing Retail Inventory & Demand Risk: A Walmart M5 Case Study

> **Core Business Questions:** 
> * *How can large-scale retailers accurately forecast 28-day item-level demand across thousands of volatile SKUs without falling victim to feature explosion, data leakage, or error accumulation?*
> * *How can probabilistic quantile forecasts (10th/90th percentiles) translate directly into optimal inventory safety stock buffers and risk-aware supply chain decisions?*
> * *How can retailers safely forecast and minimize demand uncertainty for newly launching items (the cold-start problem) where historical sales data and recursive lags are completely absent?*

---

This repository houses a production-grade, leak-free, recursive time-series forecasting pipeline built as an applied case study on the M5 Forecasting (Walmart) competition dataset. Focused initially on store `CA_1` (`FOODS`, `HOBBIES`, `HOUSEHOLD` departments), the system bridges raw historical point-of-sale data with downstream inventory decision-making, deploying the core model via a FastAPI endpoint with complete uncertainty quantification.

## Key Performance & Engineering Highlights
- **Optimized WRMSSE: 0.8178** (with near-zero bias of `0.0035`) achieved through rigorous feature ablation studies.
- **Recursive Error Mitigation:** Successfully neutralized recursive error-propagation loops by eliminating noisy short-term micro-lags (`lag_1` through `lag_3`, intermediate lags) and prioritizing robust structural anchors (`lag_7`, `lag_28`) alongside dynamic trend features (`selling_trend`, `demand_vs_historical_mean`).
- **Leak-Free Validation:** Guaranteed temporal integrity by ensuring lag and rolling features are built strictly from real historical sales or prior recursive predictions—completely eliminating future data contamination.
- **Hardware-Optimized Scope:** Scoped initially to store `CA_1` to handle massive feature engineering workflows (~59M rows) reliably within local resource boundaries.

## Tech Stack & Architecture
- **Core Engine:** Python (environment managed via `uv`), LightGBM (Point & Quantile regression)
- **Experiment Tracking:** MLflow (tracking hyperparameter iterations and model artifacts; PostgreSQL backend integration for live API data logging is currently in progress)
- **Serving & Database:** FastAPI, PostgreSQL, Docker & Docker Compose
- **Repository Structure:**
  - `src/`: Core pipeline logic, feature engineering classes, and recursive simulation loops
  - `scripts/`: Automated execution scripts (`train_and_eval.py`, `main.py`)
  - `notebooks/`: Exploratory data analysis and feature ablation experiments
  - `results/`: Evaluation outputs and performance metrics

## Current Status: In Progress 🚧
The core machine learning pipeline, validation framework, and prediction engine are fully functional, with advanced components actively expanding:
- [ ] **In Progress:** PostgreSQL integration for real-time API data ingestion and logging
- [ ] **In Progress:** Business case studies translating quantile outputs into actionable inventory policies, safety stock calculations, stockout-cost tradeoffs, and cold-start attribute mapping
- [ ] **Next step:** Direct (non-recursive) baseline model for head-to-head comparison and multi-store scaling extensions

## Evaluation Metric
- **WRMSSE** (Weighted Root Mean Scaled Squared Error) — The official multi-scale evaluation metric of the M5 competition.