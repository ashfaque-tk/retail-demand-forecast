## Role

Act as a senior **demand forecasting / supply-chain analytics / operations-research practitioner** who has also built production-oriented forecasting systems and worked with freelance clients.

I am not asking you to write code yet. I want you to inspect my existing project structure and master CV, understand what is already implemented, and produce a **7-day execution plan** that turns my existing work into a credible portfolio/demo that can help me:

1. **Land my first freelance forecasting/optimization client within ~2 weeks**, and
2. Simultaneously demonstrate to recruiters that I can build an end-to-end forecasting system rather than only train an isolated ML model.

I have approximately **2–3 hours per day** available for the next 7 days.

Be brutally practical. Do not recommend rebuilding the project from scratch. Do not turn this into an MLOps project. Do not add technologies merely because they are common in industry.

---

# 1. First: understand my current project

I already have a substantial demand forecasting + inventory project.

The project is based on the **Walmart M5 dataset**, currently working with approximately **500 SKUs** covering different demand/revenue behaviours.

The current system includes, to varying degrees:

- Global LightGBM forecasting
- Recursive forecasting
- Direct / multi-horizon forecasting work
- Time-series lag, rolling and calendar features
- Rolling / walk-forward backtesting
- Multiple forecast evaluation metrics
- Quantile forecasts
- Safety-stock estimation
- Inventory simulation
- Holding/stockout cost analysis
- Different training-window experiments
- A configuration-driven experiment/deployment flow
- FastAPI
- PostgreSQL integration
- A project structure intended to separate experimentation from deployment

My current architecture has a `config.py` where I can select something like:

- `experiment` → run backtesting, generate forecast accuracy metrics and inventory-cost results across windows
- `deploy` → prepare/deploy the selected model

This was intentionally designed for **workflow fluidity**: I want to be able to change features/models and rerun the experiment without rewriting the pipeline.

FastAPI currently reads through a layer that connects to PostgreSQL. PostgreSQL is not yet populated with a deployed model/data system.

I am unsure whether this architecture is already reasonably industry-compatible or whether it needs some targeted refactoring.

**Do not assume that it needs a major rewrite. Audit it first.**

---

# 2. My actual business objective

My immediate goal is **freelancing in forecasting and optimization**, not merely getting a Data Scientist job.

I want a potential client to look at my portfolio and immediately understand:

> “This person can take demand data, analyse demand behaviour, build and compare forecasting approaches, translate uncertainty into inventory decisions, and communicate the financial/operational consequences.”

Therefore the portfolio should look like something I could plausibly show a real client after completing a forecasting engagement.

At the same time, recruiters should be able to see:

> “This person understands global forecasting, backtesting, model comparison, inventory implications, and how to structure a reusable forecasting system.”

The deliverable therefore needs to sit between:

**client-facing business analytics**

and

**credible forecasting engineering.**

---

# 3. The central problem I need you to solve

I currently have a lot of pieces, but I don't feel that the system has enough **fluidity and coherence**.

I don't know whether what I currently have should be considered:

- a forecasting project,
- a global ML forecasting model,
- a forecasting pipeline,
- a forecasting engine,
- or an automated forecasting system.

I also don't know what needs to be added/refactored before I can credibly describe it as a **global ML forecasting system/engine**.

Do NOT simply tell me to add MLflow, Airflow, Kubernetes, cloud infrastructure, CI/CD, etc.

The central questions are:

### A. Forecasting methodology

If I call this a **global ML forecasting model**, what does that actually require?

I understand that a global model learns across many SKUs rather than training one independent model per SKU.

But I am confused about the feature engineering required to make that claim credible.

In particular:

- Should the model explicitly contain features representing different SKU demand patterns?
- Should I create SKU-level statistical descriptors?
- Should I represent intermittency, volatility, demand frequency, trend, seasonality, revenue/volume, etc.?
- Should these be static features, rolling features, or both?
- Do I need explicit SKU-pattern clustering/grouping?
- Is clustering actually necessary?
- How should hierarchical information such as item/category/department/store be represented?
- Which features would materially improve the credibility of the global model?
- Which additions are unnecessary overengineering for this portfolio?
- How should I compare the global ML model against stronger baselines and/or specialised methods such as seasonal naive, moving averages, Croston-type methods, etc.?

I want you to distinguish between:

**“technically useful”**

and

**“necessary to demonstrate competence.”**

Do not make me implement everything possible.

---

# 4. Current forecasting comparison

I originally compared the global LightGBM model mainly against simpler baselines such as moving average and seasonal naive.

I now understand that if I want to present this seriously as a **global forecasting engine**, I may need a stronger benchmark framework.

Evaluate whether I should add methods such as:

- seasonal naive
- moving average
- exponential smoothing / ETS
- ARIMA or another statistical benchmark
- Croston / SBA / TSB for intermittent demand
- another global model if justified

Do not automatically recommend all of them.

Tell me:

1. Which baselines are actually important for my particular portfolio?
2. Which ones matter for freelance credibility?
3. Which ones matter for recruiter credibility?
4. Which ones are unnecessary within the 7-day constraint?
5. How should the comparison be presented so that a non-technical client can understand it?

---

# 5. Inventory / business decision layer

One of the things I want to demonstrate is that forecasting is not the final output.

My current idea is to show:

### Forecasting

Actual vs forecasted demand for selected SKUs, potentially comparing:

- global LightGBM
- direct forecast
- recursive forecast
- appropriate statistical baseline(s)

### Inventory

Then show two safety-stock approaches:

1. **Error-based / RMSE-based safety stock**
2. **Quantile-based safety stock**

Then simulate inventory decisions over the forecast period and calculate things such as:

- holding cost
- stockout/lost-sales cost
- total inventory-related cost
- service-level behaviour

The goal is to demonstrate:

> Forecast → uncertainty → safety stock → inventory decision → business cost

I want you to critically evaluate this design.

Tell me whether:

- both safety-stock policies should remain,
- one should be removed,
- the terminology should change,
- the simulation design is credible,
- the cost comparison is useful,
- there is a better business metric,
- or something important is missing.

Do not let me present a technically impressive but misleading inventory simulation.

---

# 6. Portfolio deliverables

My current proposed final portfolio consists of:

### A. Interactive Plotly forecasting dashboard

For example:

- actual vs forecast
- model comparison
- forecast horizon
- SKU selection
- different demand patterns
- potentially uncertainty intervals

I want the viewer to be able to select a SKU and understand what happened.

### B. Inventory decision visualisation

Show how different safety-stock policies affect:

- inventory
- stockouts
- holding cost
- stockout/lost-sales cost
- total cost

Ideally the viewer can clearly understand **which forecasting/model/policy combination is being used**.

### C. Business report

Create a concise report as if I had actually completed a forecasting engagement for a client.

It should answer things such as:

- What did we analyse?
- What demand patterns exist?
- How accurate are the forecasting methods?
- Where does the global model work well/poorly?
- What are the major sources of forecast error?
- What does uncertainty imply for inventory?
- What policy produces what business trade-off?
- What should the business do next?

The report should not read like a university thesis.

It should read like a **client deliverable**.

### D. Public portfolio / LinkedIn presentation

I want to turn the project into something I can publish publicly and use to attract:

- freelance clients
- hiring managers
- recruiters
- forecasting/supply-chain professionals

Evaluate whether these deliverables are the right ones.

If they are too complex, reduce them.

If something important is missing, replace/add it.

Do NOT assume that “more features = better portfolio.”

---

# 7. Architecture question

I currently have an experiment/deployment architecture roughly along the lines of:

`config.py`
→ experiment mode
→ feature/model experiments
→ rolling backtesting
→ forecast metrics
→ inventory cost metrics

and:

`config.py`
→ deployment mode
→ selected model
→ FastAPI
→ PostgreSQL/data layer

I deliberately built this for experimentation fluidity.

I want you to assess:

### What is already good?

### What is weak?

### What is misleading if I call it an industry-style forecasting engine?

### What is the minimum refactoring required?

### What should NOT be touched?

For example, determine whether I actually need:

- a cleaner configuration system
- model registry abstraction
- feature pipeline abstraction
- forecasting interface
- model selection layer
- prediction storage
- database schema
- API improvements
- separation of training/backtesting/inference
- reproducibility improvements

versus unnecessary additions such as:

- Airflow
- MLflow
- Kubernetes
- cloud deployment
- elaborate CI/CD
- orchestration platforms

The goal is:

> **minimum architectural changes that make the existing system coherent, reusable and credible.**

Not:

> rebuild the entire project as a production enterprise platform.

---

# 8. What does “automated global forecasting engine” actually mean?

I specifically want you to resolve this terminology.

Explain whether the following concepts are different:

1. Global ML forecasting model
2. Forecasting pipeline
3. Forecasting engine
4. Automated forecasting engine
5. Production forecasting service
6. End-to-end forecast-to-decision system

Then tell me which terminology I can **truthfully use for my current project after the 7-day improvements**.

Do not let me use impressive terminology that my implementation does not support.

Also explain what the minimum characteristics of an automated global forecasting engine would be.

---

# 9. Client perspective

Assume I am a freelance forecasting consultant approaching a small/medium company with historical sales data.

The client does NOT care that I used LightGBM.

They care about questions such as:

- Can you understand our demand?
- Can you forecast future demand?
- Can you identify which products are predictable/unpredictable?
- Can you quantify uncertainty?
- Can you improve inventory decisions?
- Can you identify potential overstock/stockout risk?
- Can you quantify financial impact?
- Can you explain the result?
- Can you give us something reusable rather than a one-off notebook?

Design the 7-day plan around proving those capabilities.

---

# 10. Recruiter perspective

At the same time, assume a recruiter/hiring manager sees the project on LinkedIn.

They should be able to infer:

- global forecasting
- multi-SKU modelling
- feature engineering
- time-series validation
- model benchmarking
- forecast uncertainty
- inventory/safety-stock reasoning
- optimisation/decision thinking
- reusable Python architecture
- API/deployment awareness
- business communication

But I do NOT want the portfolio to become an MLOps showcase.

The forecasting methodology and business reasoning should remain the centre.

---

# 11. The 7-day constraint

Produce exactly a **7-day action plan**.

Each day must contain **ONE primary task** that can realistically be completed in **2–3 hours**.

For each day give:

### Day X — [single task]

**Objective**

What this accomplishes.

**Why it matters**

Why it matters specifically for:
- freelance clients
- recruiters
- forecasting credibility

**What I should do**

Concrete actions, in order.

**Definition of done**

A very clear objective test.

I should be able to answer:

> “Did I actually finish this?”

with yes/no.

**Do NOT do today**

Explicitly list tempting work that I should postpone.

This is important because I tend to go down technical rabbit holes.

---

# 12. Prioritisation rules

Use these rules when designing the plan:

### Priority 1 — Client credibility

Would this help a potential client understand that I can solve a real forecasting/inventory problem?

### Priority 2 — Forecasting credibility

Would this make the methodology more defensible?

### Priority 3 — System credibility

Would this make the project look like a reusable forecasting system rather than a notebook?

### Priority 4 — Recruiter visibility

Would this create something visually compelling enough to publish on LinkedIn/GitHub/portfolio?

### Priority 5 — MLOps

Only include MLOps work if it directly improves the above.

---

# 13. Very important: do not overengineer

I have only 7 days.

If something would take 1–2 days but provides only marginal portfolio value, explicitly tell me:

> **DO NOT DO THIS NOW.**

I would rather have:

- excellent evaluation
- strong SKU segmentation
- clear visualisations
- credible inventory simulation
- a polished business report
- coherent architecture

than:

- MLflow
- Airflow
- Kubernetes
- cloud deployment
- elaborate CI/CD

with weak forecasting analysis.

---

# 14. Final deliverable of your response

Before giving me the 7-day plan, first provide a short **diagnosis of my current project** based on the project directory and files I provide.

Classify the current state into:

| Area | Current state | Gap | Priority |
|---|---|---|---|
| Forecasting methodology | | | |
| Global model design | | | |
| Feature engineering | | | |
| Baselines | | | |
| Backtesting | | | |
| Inventory simulation | | | |
| Business metrics | | | |
| Architecture | | | |
| Deployment/API | | | |
| Visualisation | | | |
| Client deliverable | | | |
| Recruiter signal | | | |

Then explicitly answer:

### “What is the single biggest weakness of the current project?”

### “What is the single biggest opportunity?”

### “What should I absolutely NOT spend these 7 days doing?”

Then produce the 7-day plan.

---

# 15. Final success criterion

At the end of Day 7, I should have something that can honestly be described approximately as:

> **An end-to-end global demand forecasting and inventory decision system that evaluates multiple forecasting approaches across heterogeneous SKUs, quantifies forecast uncertainty, translates forecasts into safety-stock/inventory decisions, evaluates downstream business costs, and exposes the results through an interactive client-facing analysis.**

If that statement is too ambitious given my existing implementation, **change the statement** rather than forcing the project to meet it.

The final output should tell me exactly what I need to do to reach the strongest truthful version of the project within the 7-day limit.

Finally, give me:

## The 5 things I should publish

For example:
1. Portfolio case study
2. Interactive dashboard
3. Business report
4. LinkedIn technical post
5. LinkedIn business-oriented post

But choose the actual five based on the project rather than blindly following these examples.

## The 3 things I should NOT publish

Identify anything that would expose weaknesses, create misleading claims, or distract from the core value.

## The final positioning

Give me 2–3 truthful ways I could describe the project publicly to a potential freelance client and to a recruiter.

Do not write marketing hype. The positioning must be supported by what the system actually does.


pool your analysis and report into seperate file named 'action_plan_report.md'
