# Retail Forecast System — Diagnostic & 7-Day Execution Plan

*Prepared for the action plan defined in `ACTION_PLAN.md`. This document is based on a direct audit of the repository (`src/`, `run_pipeline.py`, `config.py`, `main.py`, `models/deployments/`, `results/Experiments/`, notebooks, and git history).*

---

## 1. Diagnosis of the current project

The project is a **Walmart M5 (store CA_1) demand-forecasting + inventory case study** at roughly 300 SKUs in the current filtered data files (`train_filtered_ca1.parquet` = 300 items, 1,096 days; config comment claims 500 SKUs — the number has drifted between runs). **69% of SKUs have >50% zero-sales days and median daily sales per SKU is 0.78 units** — this is, in plain terms, an intermittent/erratic demand set, not a "fast-moving retail" dataset. That shape drives several of the recommendations below.

What actually exists, honestly:

- **Global model.** One LightGBM (tweedie point objective) fit across all SKUs, with `item_id`, `cat_id`, `dept_id` as native categoricals plus per-SKU lag/rolling features. This *is* a genuine global model, not per-SKU models.
- **Recursive and direct multi-horizon forecasters.** Both implemented; `direct_forecast` builds a stacked horizon frame, `recursive_forecaster` feeds predictions back day-by-day. Logged results: direct 1yr WRMSSE ≈ 0.864 vs recursive 1yr ≈ 0.865; direct MAE ≈ 1.15 vs recursive ≈ 2.04.
- **Walk-forward backtesting.** Rolling (reversed, most-recent-first) and expanding windows, step 28 days, storage of OOS errors for safety stock.
- **Metrics.** WRMSSE, MAE, Bias%, dept/cat aggregation, plus FVA vs seasonal naive.
- **Inventory cost layer.** Order-up-to `(S)` policies with `safety_stock_rmse`, `safety_stock_mae`, `safety_stock_classical`, and per-review-period holding cost.
- **API.** FastAPI that *reads precomputed* forecasts and policies from PostgreSQL tables.
- **Visuals.** Solid Plotly primitives in `src/utils_visuals.py`; static PNGs in `case_studies/`; no runnable dashboard app yet.

### State table

| Area | Current state | Gap | Priority |
|---|---|---|---|
| Forecasting methodology | Recursive + direct GBM; WRMSSE/MAE/Bias/FVA | No classical/relevant statistical competitor tuned for intermittent demand; metrics not presented at SKU/segment level | High |
| Global model design | Single LightGBM, categorical `item_id/dept_id/cat_id` — legitimately global | No explicit per-SKU static descriptors; no segment identity; "global" claim has no supporting feature story | High |
| Feature engineering | Leak-free lags/rolling/trend/price/calendar/event; 40 features persisted | No static SKU features (intermittency, volatility, demand level, revenue weight) | Medium |
| Baselines | Seasonal naive (lag 28), moving average (180 d) | No Croston/SBA/TSB — the data is 69% >50%-zero SKUs, so this is the single most relevant missing benchmark | High |
| Backtesting | Walk-forward rolling/expanding, OOS error carried across windows | Fine; only needs a consistent runnable entry point and cleaner FVA pivot | Low |
| Inventory simulation | **Analytical estimate only** (S = forecast + safety stock, holding cost ≈ cycle stock + SS). The `InventoryPolicy.daily_simulation` class is incomplete dead code | No day-by-day simulation of realized stockouts, lost sales, fill rate, or total cost → the headline "forecast→inventory→cost" story is not actually verifiable | High |
| Business metrics | Holding cost per review period, FVA on WRMSSE + holding cost | No stockout/lost-sales cost, no achieved service level, no total relevant cost; "monthly_holding_cost_*" misnomer (computed per 7-day cycle) | High |
| Architecture | Config-driven Experiment/Deploy modes; clean `src/` separation; parquet artifact set + manifest per deployment | Some paths only work in one mode; duplicated JSON-log code between `run_pipeline.py` and `src/logs.py`; stale/legacy artifacts differ in schema from current code | Medium |
| Deployment/API | FastAPI reads precomputed from PostgreSQL; Docker compose with Postgres | **DB is never populated** — no path from Deploy artifacts → Postgres; API expects `q10/q90` but current ML run emits point only; `main.py:70` bug (`if not forecast_rows or policy_row`) | Low (means) |
| Visualisation | Plotly helpers + static case-study charts | No runnable interactive dashboard; case-study PNGs are not a portfolio demo | High |
| Client deliverable | Notebooks (error accumulation, safety stock/cost, cold start) + `fva_model_summary.html` | No client-style business report | High |
| Recruiter signal | README + case studies + results JSONs | README overclaims ("production-grade", MLflow, ~0.8178) vs. what runs; `data/` and `models/` are gitignored so a public repo can't reproduce anything | Medium |

### The single biggest weakness

**The pipeline, as committed, does not run end-to-end.** `src/inventory_policy.py` crashes at import time (line 50: `type:str='rmse'|'quantile'` — invalid; you cannot `|` two string literals), which takes down `run_pipeline.py`, the inventory layer, and every downstream artifact. Independent of that, the inventory "cost" numbers currently produced are a closed-form estimate that never simulates a stockout or measures achieved fill rate. So the one thing that separates this from "another forecast notebook" — the forecast→uncertainty→inventory→business-cost chain — is neither runnable nor verifiable right now.

### The single biggest opportunity

**The forecast→inventory→cost chain is genuinely differentiated and 80% of the pieces already exist.** The Plotly primitives, the global-model plumbing, the walk-forward engine, and the safety-stock concept are here. Repairing the broken module, adding one relevant intermittent-demand benchmark, and turning the analytical estimate into a real order-up-to simulation with stockout/service-level/total-cost outputs transforms this from "a global LGBM with extra metrics" into the exact end-to-end system clients and recruiters are asked to believe in. That is achievable in 7 days.

### What you should absolutely NOT spend these 7 days doing

- **DO NOT**: MLflow, Airflow, Kubernetes, cloud deployment, CI/CD, orchestration.
- **DO NOT**: polish the FastAPI/PostgreSQL layer (it is a *read* API over tables nothing writes to; it adds zero portfolio value this week).
- **DO NOT**: tune LightGBM hyperparameters, run feature-ablation sweeps, or chase one more decimal on WRMSSE.
- **DO NOT**: add ARIMA/ETS per-SKU modelling, multi-store scaling, or a second dataset.
- **DO NOT**: implement clustering (k-means) over SKU descriptors. Not needed; see below.

---

## 2. Central methodology questions — answered directly

### A. What does "global ML forecasting model" actually require?

A global model must (a) train **one** model on pooled rows across all SKUs, (b) encode *who the series is* and *where it is in its own history*, and (c) be evaluated out-of-sample across all series. Yours already satisfies (a) and (c), and partially (b) via `item_id` categoricals + per-SKU lags/rollings.

What makes the claim *credible and explainable* (and worth 2–3 hours):
- **A small static SKU descriptor block** — mean daily demand, zero-sales rate (intermittency), coefficient of variation, revenue share, active-history length — computed once per SKU and merged in as features. This is "technically useful" **and** "necessary to demonstrate competence": it is the readable evidence that the model knows an intermittent item from a smooth one, and it is exactly what a client or recruiter will ask ("how does it know these are different products?") And these descriptors also power the segment analysis on Day 3, so they are built once and reused twice.
- **A segment/category identity** (`smooth/intermittent/erratic/lumpy` or simpler `high/low volatility × smooth/intermittent`), which doubles as a cross-feature and as the grouping for the "where does it work" analysis.

Deliberately **NOT needed** for this portfolio:
- SKU-pattern clustering (k-means etc.). A deterministic rule-based segment tag is faster, explainable to a non-technical client, and easier to defend than a cluster centroid.
- Hundreds of micro-lags. Your lag set (7/28/60/90 + rollings) is the right shape; adding lag_1..3 reintroduces the noise you already documented.
- Explicit item/category/department hierarchy features beyond the categorical columns you already pass to LightGBM.

Rule of thumb for this week: **static descriptors = yes, one-day; clustering = no; more rolling/lag variants = no.**

### B. Do you need explicit SKU descriptors as static features, rolling features, or both?

Both, and they serve different purposes:
- **Static** (per-SKU, time-invariant): intermittency, CV, mean level, revenue share. Use in the global model and in reports ("which products are predictable").
- **Rolling** (already present): recent level and trend. Keep as-is.
Do not build *rolling* intermittency features — overengineering for 7 days.

### C. Baselines — which matter?

Your current two (seasonal naive, moving average) are correct and must stay; they are the methods a non-technical client's planners actually see. What is missing, given the data (69% of SKUs >50% zeros):

| Baseline | Freelance credibility | Recruiter credibility | Needed in 7 days? |
|---|---|---|---|
| Seasonal naive | Yes — the planner's default | Yes | Already present |
| Moving average | Yes | Yes | Already present |
| **Croston / TSB (intermittent)** | **Yes — this dataset *is* intermittent** | **Yes — distinguishes you from notebook-tutorial people** | **Add (Day 2)** |
| ETS | Marginal | Nice-to-have | No |
| ARIMA | Marginal | Nice-to-have | No |
| A second global model | No | Marginal | No |

How to present it to a non-technical client: never show metric tables first. Show a sentence: *"We compared the model against the two methods most retailers already use — 'this week = same time last period' and 'a recent average' — plus a method tuned for slow-selling items."* Then a single ranked bar chart. One chart, three bars, one headline: how much accuracy ML adds, and where it doesn't.

### D. Inventory / business decision layer — critical evaluation

- **Keep the error-based (RMSE/MAE-over-τ) safety stock.** This is the workhorse; it is honest and standard.
- **Keep the classical `z·σ_d·√(L)`** as the textbook reference — but rename it "standard (textbook) policy", not "classical", and treat it as the benchmark you outperform.
- **The "quantile-based safety stock" does not exist yet.** Case study 02's title claims it; the code computes Gaussian quantiles only for baselines and the ML config has `quantiles: None`. On Day 4 wire real q10/q90 model output and define `SS_quantile = Σ(q90 − point) over L+R`. Keep both RMSE- and quantile-based: they answer different questions (RMSE = "error is X, hold z·X"; quantile = "to cover 90% of outcomes, hold this surplus").
- **The simulation is the actual problem.** An order-up-to level without a day-by-day run over realized demand tells you nothing about stockouts or service level. Replace the analytical estimate with a deterministic periodic-review simulation (order every `R`, lead time `L`, track on-hand/on-order, fulfill-or-lose, count stockout days, compute fill rate, holding cost, and total cost = holding + stockout). This is Day 5 and it is the single biggest credibility upgrade available.
- **The better business metric:** total relevant cost (holding + lost-sales/stockout) **per SKU over the horizon, at a stated achieved fill rate**, plus a "units of inventory kept per unit of demand protected" ratio. Present policy A vs B as a trade-off curve (higher fill rate ↔ more holding cost), never as a single "winning policy."
- **Missing and important:** SKU-segment × policy × cost cross-tab, and a coverage check (did the 80% interval contain reality ~80% of the time?). Both are cheap.

Terminology note: **do not call the holding-cost estimate "monthly"** — it is computed over a 7-day review cycle × daily rate.

---

## 3. Architecture assessment

**Already good, don't touch:**
- `config.py` single entry for run mode / model / forecast type / backtest / paths → the workflow fluidity you wanted works.
- `src/` separation (features, models, recursive/direct, backtest, inventory, metrics).
- Deployment produces a self-contained artifact dir + `manifest.json` — a legitimate lightweight model registry for this scale.
- Walk-forward engine that carries OOS errors across windows for safety-stock estimation.

**Weak:**
- `src/inventory_policy.py` is currently **broken and will not import** (see #1). It also contains a half-finished `InventoryPolicy.daily_simulation` class that is dead code.
- `run_pipeline.py` re-implements `_json_safe`/`log_experiment_results` that already live in `src/logs.py`.
- The FVA pivot in `backtest_engine.run_window` is fragile (relies on `level_3` label; resets on metric-name changes).
- Legacy artifacts (`forecast_comparison.parquet`) have all-NaN `real_sales` for lgbm and schemas that no longer match current code — reproducibility of past "results" is weak. Don't publish these raw.

**Misleading if you call it an "industry forecasting engine" right now:**
- The FastAPI/PostgreSQL part. It reads tables no process ever writes to; the ML run doesn't produce `q10/q90`; `main.py` has a `not forecast_rows or policy_row` bug. This is the least-portfolio-relevant 10% of the repo this week. Don't claim "production forecasting service."
- README claims "production-grade … MLflow … ~0.8178" that the reproducible path doesn't currently produce.

**Minimum refactoring (and nothing more):**
1. Fix the import bug; remove dead `InventoryPolicy` class; reconcile safety-stock columns so `generate_inventory_policy` and `calculate_inventory_costs` agree (Day 1).
2. One shared JSON-log helper in `src/logs.py`; delete the duplicate (Day 1).
3. A `scripts/publish_artifacts.py`-style writer only if it takes <1 hour — otherwise skip Postgres entirely (Day 1–2).
4. Keep config-driven experiment/deploy. Add `quantiles` + static-descriptor flags to config (Days 3–4).

**Do NOT add:** model registry abstraction, feature-pipeline abstraction, forecasting interface classes, prediction-storage layer, DB schema work, API improvements.

---

## 4. Terminology — what you can truthfully say

| Term | Means | Yours after 7 days |
|---|---|---|
| Global ML forecasting model | One model across many series | ✔ Yes (already) |
| Forecasting pipeline | Repeatable raw→features→model→forecast→metrics path | ✔ Yes (once import fixed) |
| Forecasting engine | Pipeline + backtesting + model comparison + config-driven reuse | ✔ Yes |
| Automated forecasting engine | Engine that selects/scores/evaluates from config with reproducible artifacts, no per-run notebook steps | ~ Yes after Days 1–2 |
| Production forecasting service | Scheduled/live forecast generation served via API | ✖ **No** — do not claim |
| End-to-end forecast-to-decision system | Forecasts → uncertainty → inventory policy → business cost, presented as analysis | ✔ Yes after Day 5 |

**Truthful public phrasing:** *"A global ML demand-forecasting and inventory-decision engine built on 300 heterogeneous retail SKUs: leak-free feature engineering, walk-forward validation against the benchmarks planners already use, quantile-derived uncertainty, and inventory-policy evaluation as holding-cost vs. stockout/service-level trade-offs — driven by a single configuration and explored through an interactive dashboard."* No "production API", no MLOps claims.

Minimum characteristics of an *automated* global forecasting engine, and where you stand:
1. Config-driven run of train→validate→deploy → **have**
2. Single global model across many series → **have**
3. Leak-free, reproducible features → **have**
4. Walk-forward/rolling evaluation → **have**
5. Benchmark comparison (incl. intermittent) → **after Day 2**
6. Quantified uncertainty (quantiles) from the ML model → **after Day 4**
7. Inventory decision evaluation with stockout/service-level/total cost → **after Day 5**
8. Stable artifact set + manifest each run → **have, but fix schema drift (Day 1)**

---

## 5. The 7-day plan

Constraint: one primary task per day, 2–3 hours, no rabbit holes.

---

### Day 1 — Restore an end-to-end running pipeline

**Objective:** `run_pipeline.py` executes a small experiment head-to-toe again, with consistent output columns and logged artifacts.

**Why it matters:** Nothing in the portfolio is buildable on a repo that crashes at `import`. For a freelance client and a recruiter, "it runs" is the non-negotiable floor; everything else this week sits on top of it.

**What to do (in order):**
1. Fix `src/inventory_policy.py:50` (`type:str='rmse'|'quantile'` → remove/rewrite as a string param). Verify `uv run python -c "import run_pipeline"` succeeds.
2. Reconcile the safety-stock columns: make `generate_inventory_policy` produce exactly the columns `calculate_inventory_costs` reads (`safety_stock_rmse`, `safety_stock_mae`, `safety_stock_classical`, `order_up_to_*`, `raw_demand_std`), or delete the unused ones. Fix the "monthly_holding_cost_*" names to per-review-period.
3. Delete the duplicate `_json_safe`/`log_experiment_results` in `run_pipeline.py`; import from `src/logs.py`.
4. Run `uv run python run_pipeline.py` with `max_windows: 2` and a shorter `training_window`; confirm a metrics JSON + FVA table + non-null inventory-cost columns for ml/naive/moving_average.
5. Re-run the 2.0yr direct experiment so `results/Experiments/` reflects the current code (FVA pivot should not throw).

**Definition of done (yes/no):** `import run_pipeline` succeeds; a 2-window experiment completes and logs an experiment record; `inventory_policy_comparison` has non-null `safety_stock_*` and holding-cost values for every model; the FVA table prints without error.

**Do NOT do today:** Deploy mode, PostgreSQL, quantiles, new baselines, new features, model tuning, Docker.

---

### Day 2 — Add the benchmark that fits the data: Croston/SBA (or TSB)

**Objective:** a third, intermittent-demand-aware baseline joins seasonal naive and moving average in the same metrics table.

**Why it matters:** Your data is 69% >50%-zero SKUs. A Croston-type method is the one benchmark a demand-planning client would instantly respect, and its presence is the clearest signal to a recruiter that this isn't a tutorial rehash. It is also an afternoon's work on pure pandas/numpy.

**What to do (in order):**
1. Implement Croston-SBA (or TSB) in `src/baselines.py` returning the same schema as the other baselines (`sales_pred`, `real_sales`, Gaussian quantiles).
2. Wire it into `backtest_engine.run_window`: add to `models_preds`, the metric rows, the FVA columns, and `oos_errors` / inventory evaluation.
3. Run a 1.0yr experiment over ~10 windows; write `results/model_ranking.md` with one table per metric (WRMSSE, MAE, Bias%) and a one-line commentary per model.
4. Note the honest finding in that file: on weighted revenue accuracy vs. plain unit accuracy, where the global model wins and where it doesn't (already visible in the Deployment test: lgbm WRMSSE 0.787 vs MA 0.829, but lgbm MAE 2.52 vs MA 1.35 — the ML model wins where it is weighted to be valuable).

**Definition of done (yes/no):** Croston appears as a row in the metrics table of a full run; `results/model_ranking.md` exists with all four models ranked per metric.

**Do NOT do today:** ETS, ARIMA, hyperparameter tuning, quantiles, inventory simulation work.

---

### Day 3 — SKU descriptors + segmentation + per-segment accuracy

**Objective:** assign every SKU a deterministic demand-pattern segment; quantify where the model predicts well vs. poorly by segment.

**Why it matters:** This is the "Can you identify which products are predictable/unpredictable?" question — the one that makes a client stop and listen. It also gives the global model a defensible static-feature story ("the model knows this is an intermittent, low-volatility item").

**What to do (in order):**
1. Compute per-SKU static descriptors from train history: mean daily demand, zero-sales rate (intermittency/ADI), CV², revenue share, active-history days, simple trend sign.
2. Tag each SKU into 4 segments via a simple rule on the ADI–CV² grid (`smooth / intermittent / erratic / lumpy`) or a 2×2 `(intermittent × volatility)`.
3. Merge the segment descriptor block as features into one experiment; (optionally) add the segment tag as a categorical column.
4. Produce `results/segments/segment_descriptors.parquet` and a per-segment error table (WRMSSE/MAE per model per segment) + one chart: "WRMSSE relative to seasonal naive by segment" — the predict/unpredictable story.

**Definition of done (yes/no):** every SKU has a segment tag + descriptor row on disk; the per-segment model-vs-naive comparison exists as a table and a chart; you can name one segment the model clearly helps and one it doesn't.

**Do NOT do today:** k-means clustering, hierarchical modelling, tuning, quantiles, dashboard work.

---

### Day 4 — Quantile forecasts and quantile-derived safety stock

**Objective:** the ML model produces real q10/q90/q50 outputs; safety stock gains a third, quantile-based formulation.

**Why it matters:** "Translate uncertainty into inventory decisions" is the sentence from your own success criterion. Right now quantile models exist in code but are switched off (`quantiles: None`) and only the baselines emit Gaussian bands. This day makes the uncertainty claim true and gives the inventory comparison a decision-relevant input.

**What to do (in order):**
1. Set `config.py` `quantiles: [0.10, 0.50, 0.90]`; verify training/prediction of the three quantile models and that `recursive`/`direct` return `q10/q50/q90` columns.
2. On the validation horizon, compute **coverage** — the share of realized values inside [q10, q90] — and note whether the empirical band is too narrow/wide (this is the honest check journalists will notice if skipped).
3. Implement `SS_quantile = Σ(q90 − point forecast) over L+R` in `generate_inventory_policy`, alongside `safety_stock_rmse` (z·RMSE_τ) and `safety_stock_classical` (z·σ_d·√(L+R)).
4. Re-run the experiment with quantiles on; save an `inventory_policy` table containing all three SS variants per SKU.

**Definition of done (yes/no):** an experiment with quantiles enabled logs q10/q50/q90 outputs with a reported coverage figure; the policy table contains three distinct safety-stock columns; you can show one SKU where q90-based stock is meaningfully different from RMSE-based stock.

**Do NOT do today:** conformal prediction, quantile-model tuning, the day-by-day simulation, dashboard, Postgres.

---

### Day 5 — Credible inventory simulation (stockout, fill rate, total cost)

**Objective:** replace the analytical holding-cost estimate with a deterministic day-by-day order-up-to simulation that yields holding cost, stockout/lost-sales, achieved fill rate, and total cost per model × policy.

**Why it matters:** This is the difference between "we computed a target stock level" and "here is how much extra stock each extra service point costs, and what a stockout costs us." No misleading story: the simulation must *realize* the policies against actual demand and report achieved (not assumed) service levels.

**What to do (in order):**
1. Delete/retire the dead `InventoryPolicy.daily_simulation` class.
2. Implement a clean simulation: periodic review (order every `R=7`), order-up-to `S = forecast(L+R) + SS`, lead time `L=4`, track on-hand/on-order, fulfill-or-else-lose, count stockout days and lost units, accumulate holding cost daily, compute fill rate per SKU.
3. Run the simulation over the test horizon for every model (lgbm, croston, MA, naive) × every policy (rmse, quantile-q90, classical).
4. Produce `results/policies/policy_tradeoffs.parquet` (holding cost, stockout cost, fill rate, total cost per SKU) and one trade-off chart (fill rate vs. holding cost, one line per policy).
5. Sanity-check internal consistency: higher SS ⇒ higher fill rate and higher holding cost; absurd values investigated, not papered over.

**Definition of done (yes/no):** the simulation outputs per-SKU holding cost, stockout units, fill rate, and total cost, internally consistent across policies; a deterministic reproduction run passes end-to-end.

**Do NOT do today:** multi-echelon, stochastic/Monte-Carlo demand, lost-sales optimization theory, dynamic programming, dashboard.

---

### Day 6 — Interactive dashboard (SKU explorer)

**Objective:** a runnable, client-grade interactive view: pick a SKU (or segment), see actual vs. multi-model forecast with the uncertainty band, the chosen model's inventory policy S, and a cost comparison.

**Why it matters:** This is the single most publishable artifact and the thing that makes recruiters click. You already have the Plotly primitives (`utils_visuals.py`) — this is assembly, not research.

**What to do (in order):**
1. Add a small Streamlit app `dashboard.py` (add `streamlit` to `pyproject.toml`): sidebar (SKU select, segment filter, model toggle), main plot via `plot_item_forecast_and_inventory`, an inventory-policy bar chart via `plot_inventory_policy_bars`, and a compact KPIs row (WRMSSE, MAE, fill rate, total cost).
2. Load the deployment/experiment parquet artifacts directly (no DB).
3. Capture 3 curated screenshots (a smooth SKU, an intermittent SKU, the segment overview) into `case_studies/`.

**Definition of done (yes/no):** `streamlit run dashboard.py` works locally on saved artifacts; a non-technical viewer can understand each chart without explanation; 3 screenshots saved.

**Do NOT do today:** auth, multi-user, deployment/hosting, CSS polish rabbit holes, new analyses.

---

### Day 7 — Client-style business report + packaging

**Objective:** one polished report that reads like a real engagement deliverable, and a README/positioning pass so the repo and post are honest and reproducible.

**Why it matters:** The report is what a client actually reads; the README + framing is what a recruiter reads. Both must be supported by what the system now demonstrably does.

**What to do (in order):**
1. Write `REPORT.md` (client voice): What we analysed → what demand patterns exist (segments) → how accurate each method is, per segment → where the global model helps and doesn't → what uncertainty implies for stock → which policy produces which trade-off (with the Day-5 chart) → recommended next steps.
2. Rewrite `README.md`: honest positioning (per §4 of this doc), how to install and run (`uv sync`, `config.py`, `run_pipeline.py`), what's reproducible. Remove the MLflow and "production-grade API" claims.
3. Note in the README that `data/` and `models/` are gitignored; add the one-line script that regenerates the 300-SKU processed set from `data/raw/` (already exists as `scripts/split_and_save.py` + the utils sampler) so a cloned repo can reproduce.
4. Final end-to-end check: clean clone → data prep → experiment run → dashboard opens.

**Definition of done (yes/no):** `REPORT.md` exists and can be read unaided; the README no longer overclaims; a fresh clone of the repo can regenerate data and run one experiment.

**Do NOT do today:** any new model feature, the API/DB, CI/CD, uploading to the cloud.

---

## 6. What to publish

1. **Portfolio case study (repo).** Fixed, runnable repo + the 3 case-study notebooks (error accumulation, safety-stock/cost, cold start) + `REPORT.md` as the landing narrative.
2. **Interactive dashboard.** Streamlit demo (link/screenshots/GIF) — the visual anchor.
3. **The client-style business report  (`REPORT.md`).** The strongest single artifact for freelance outreach.
4. **LinkedIn technical post.** "I built a global demand-forecasting engine on 300 retail SKUs: leak-free features, walk-forward evaluation against the benchmarks planners actually use, quantile-based uncertainty, and what a stockout actually costs." Include the trade-off chart + one SKU plot.
5. **LinkedIn business-oriented post.** Result-framed, no model names: "69% of SKUs had a half-empty shelf half the time and no one could forecast them. Here is how segment-level demand analysis changed how much stock a retailer should hold."

Publish in this order: report + dashboard → technical post → business post, with the repo linked everywhere.

## 7. What NOT to publish

1. **The FastAPI/PostgreSQL layer.** It reads from a database nothing fills, and the README/API currently overclaim. Publishing it invites one question that sinks credibility ("can we see it live?"). Leave it out this week.
2. **MLflow references and the "production-grade / 0.8178 WRMSSE" claims from the current README.** No experiment is logged to MLflow, and the 0.8178 figure is not reproducible from the committed code. Publish claims a fresh clone can verify.
3. **Raw/legacy artifacts with schema drift** (e.g., `forecast_comparison.parquet` with all-NaN `real_sales` on the ML rows). Only publish artifacts regenerated by the fixed pipeline, or they become evidence the numbers are untrustworthy.

## 8. Final positioning (truthful, supported)

**To a freelance retail client:** *"I built an end-to-end forecasting and inventory-decision system for 300 SKUs with very different demand patterns — from best-sellers to items that sell a few units a month. It benchmarks a global ML model against the forecasting methods planning teams already use, converts forecast uncertainty into concrete safety-stock levels, and shows the holding-cost/stockout trade-off of each policy in euros. I can do this for your data."*

**To a recruiter/hiring manager:** *"An end-to-end global demand-forecasting and inventory-decision engine over 300 heterogeneous retail SKUs: single global model, leak-free feature engineering, walk-forward validation, benchmark comparison including intermittent-demand methods, ML-generated quantile uncertainty, and a realized inventory simulation with service level and total cost — all driven by one config and explored through an interactive dashboard."*

**One-line LinkedIn bio pitch:** *"I build global demand-forecasting and inventory-decision systems that translate sales data into defensible stock and cost decisions."*

---

## 9. Known reproducible numbers to use in the report (from committed artifacts)

- Current filtered set: 300 SKUs, CA_1, 1,096 train days, 138 test days. (Config comment says 500 — fix the comment.)
- 69% of SKUs have >50% zero-sales days; median per-SKU daily sales ≈ 0.78. Data is intermittency-heavy.
- Logged experiments (window-averaged): direct 1yr WRMSSE ≈ 0.864 / MAE ≈ 1.15; direct 2yr ≈ 0.868 / MAE ≈ 1.17; recursive 1yr ≈ 0.865 / MAE ≈ 2.04; recursive 2yr ≈ 0.895 / MAE ≈ 2.05.
- Deployment test metrics (lgbm / MA / SNaive): WRMSSE 0.787 / 0.829 / 1.149; MAE 2.52 / 1.35 / 1.57; Bias% +10.3 / +8.2 / −6.2.
- Calibration of that deployment: lgbm WRMSSE ≈ 0.825, bias +5.3%.

Treat these as *indicative* and regenerate them after Day 1 — the committed numbers are from runs whose code path is currently broken.