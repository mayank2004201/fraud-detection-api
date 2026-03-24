# Free Fraud Detection API — Full Implementation Plan

> **How to use this document**
> Every phase has a `STATUS` field. When you stop working, update it to `IN PROGRESS` or `DONE`. When you return, `Ctrl+F STATUS: IN PROGRESS` to find exactly where you left off. Every section also lists a clear **resume point** so you never lose context.

---

## Project Summary

A production-style fraud detection microservice that combines:
- **ML** — XGBoost classifier + Isolation Forest anomaly detector + SHAP explainability
- **LLM** — Three distinct roles: fraud investigator, risk override decision-maker, and natural language query engine
- **FastAPI** — Async REST API with API key auth, rate limiting, SQLite logging, and a `/stats` endpoint
- **Docker** — Single container for packaging and deployment (dev uses plain uvicorn)
- **Render** — Free-tier hosting, deployed via GitHub push

**Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (free, 284k transactions, 492 fraud cases)
**LLM**: Groq API free tier — Llama 3.3 70B (fastest free inference available)
**Total cost**: $0

---

## Architecture Overview

```
Client
  └── POST /predict
  └── GET  /stats
  └── GET  /explain/{id}
  └── POST /query          ← natural language query endpoint

FastAPI (inside Docker, deployed on Render)
  ├── API key middleware
  ├── Pydantic validation
  ├── Rate limiter (slowapi)
  │
  ├── ML Engine
  │   ├── XGBoost  →  fraud_probability
  │   ├── Isolation Forest  →  anomaly_score
  │   └── SHAP  →  top_factors[]
  │
  ├── LLM Layer (Groq)
  │   ├── Role 1: Fraud Investigator  →  case_summary
  │   ├── Role 2: Risk Override  →  final_risk + override_reason
  │   └── Role 3: NL Query Engine  →  sql + result
  │
  ├── Risk Engine
  │   └── merges ML + LLM → final_verdict
  │
  └── SQLite
      ├── predictions table
      ├── llm_decisions table
      └── drift_buffer table
```

---

## Final API Response Shape

Understanding this upfront keeps every component focused on what it needs to produce.

```
POST /predict → {
  transaction_id       string
  fraud_probability    float     ← from XGBoost
  anomaly_score        float     ← from Isolation Forest
  ml_risk_level        string    ← LOW / MEDIUM / HIGH  (raw ML verdict)
  final_risk_level     string    ← LOW / MEDIUM / HIGH  (after LLM override)
  was_overridden       bool      ← true if LLM changed ML verdict
  override_reason      string    ← LLM's stated reason for changing verdict
  top_shap_factors     list      ← [{feature, impact}, ...]
  llm_case_summary     string    ← full investigation note from LLM
  drift_alert          bool      ← true if feature distribution has shifted
  latency_ms           float
}

GET /stats → {
  total_predictions    int
  high_risk_count      int
  override_rate        float     ← % of transactions LLM changed
  avg_fraud_prob       float
  drift_alerts_today   int
}

POST /query → {
  question             string    ← user's plain-English question
  generated_sql        string    ← what the LLM wrote
  result               list      ← query results
  interpretation       string    ← LLM's plain-English answer
}
```

---

## Phase 0 — Environment Setup

**STATUS: NOT STARTED**
**Estimated time: 2–3 hours**
**Resume point: check which step number you completed last and continue from the next one**

### What you need installed locally
- Python 3.11+
- Git
- A code editor (VS Code recommended)
- A Kaggle account (for dataset download)
- A Groq account (for free API key) → https://console.groq.com
- A Render account → https://render.com
- A GitHub account

#### [Files to Implement]
- **`app/main.py`**: Initial entry point (stub).
- **`.gitignore`**: Exclude `venv/`, `.env`, and `__pycache__`.
- **`.env.example`**: Template for Groq and API keys.
- **`requirements.txt`**: List all core dependencies (FastAPI, XGBoost, etc.).
- **`scripts/setup_structure.py`**: The script (formerly `template.py`) to generate the folder tree.

### Step-by-step

**Step 1 — Create the project folder and Git repo**
- Create a folder called `fraud-detection-api`
- Run `git init` inside it
- Create a GitHub repo and connect it
- This is your deployment pipeline — every `git push` to `main` will trigger a Render redeploy

**Step 2 — Create a Python virtual environment**
- Use `python -m venv venv` and activate it
- All packages should be installed here, never globally
- Add `venv/` to `.gitignore`



**Step 3 — Create the folder structure**
```
fraud-detection-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── logger.py            # central logging setup
│   │   ├── logging_config.py    # format + handlers
│   │   └── security.py          # API key and security logic
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── dependencies.py
│   │   └── middleware.py        # rate limiting and auth middleware
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── risk_engine.py
│   │   ├── analysis_service.py
│   │   └── llm_query_service.py # NL -> SQL execution logic
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── inference.py         # main prediction entry point
│   │   ├── preprocessing.py     # scaling and transformation
│   │   └── utils.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py            # Groq client initialization
│   │   ├── llm_investigator.py
│   │   ├── llm_override.py
│   │   ├── llm_query.py
│   │   └── prompts.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── storage_handler.py
│
├── monitoring/
│   ├── __init__.py
│   ├── drift.py
│   └── logging_monitor.py       # optional: log-based monitoring
│
├── logs/                        # all runtime logs go here
│   ├── app.log
│   └── error.log
│
├── model/
│   └── .gitkeep
│
├── training/
│   ├── train.ipynb
│   └── pipeline.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_ml.py
│   ├── test_llm.py
│   └── test_storage.py
│
├── scripts/
│   └── setup_structure.py
│
├── Dockerfile
├── render.yaml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

**Step 4 — Create `.gitignore`**
- Add: `venv/`, `.env`, `*.db`, `__pycache__/`, `.ipynb_checkpoints/`
- Do NOT gitignore `model/model.pkl` — it must be committed so Docker can include it

**Step 5 — Create `.env` and `.env.example`**
- `.env` (never committed): `GROQ_API_KEY=your_key_here`, `API_SECRET_KEY=choose_any_string`
- `.env.example` (committed): same keys but with empty values — this documents what others need to set

**Step 6 — Create `requirements.txt`**
Core packages to include:
- `fastapi` — web framework
- `uvicorn[standard]` — ASGI server for local development
- `pydantic` — request/response validation
- `python-dotenv` — loads `.env` file
- `xgboost` — gradient boosted tree classifier
- `scikit-learn` — Isolation Forest, preprocessing, metrics
- `shap` — model explainability
- `groq` — official Groq Python SDK
- `scipy` — KS test for drift detection
- `pandas` — data manipulation
- `numpy` — numerical operations
- `slowapi` — rate limiting middleware for FastAPI
- `pytest` and `httpx` — testing

**Step 7 — Verify your Groq API key works**
- Open a Python shell, import the Groq SDK, make a simple test call
- Confirm you get a response before spending hours building around it

---

## Phase 1 — ML Training (Google Colab)

**STATUS: NOT STARTED**
**Estimated time: 1–2 days**
**Resume point: open `training/train.ipynb` — the last completed cell group is your resume point. Add a markdown cell saying "STOPPED HERE" when you pause.**

### Why Google Colab, not locally
Colab gives you free CPU (and sometimes GPU) with no setup. Training XGBoost on 284k rows is fast on CPU — under 5 minutes. Colab is only for training. Everything else runs locally and on Render.

#### [Files to Implement]
- **`training/train.ipynb`**: Complete training walkthrough (Kaggle dataset -> Serialized model).
- **`training/pipeline.py`**: Reusable script for model retraining.
- **`model/model.pkl`**: The combined dictionary of model, scaler, and baseline stats (output of this phase).

### Concepts to learn and apply (in order)

**Concept 1 — Understanding the dataset**
- The dataset has 30 features: `Time`, `Amount`, and `V1–V28`
- `V1–V28` are already PCA-transformed (anonymised) — you cannot reverse them
- The `Class` column is the label: 0 = legitimate, 1 = fraud
- Critical insight: only ~0.17% of transactions are fraud — this is a severely imbalanced dataset. If you train naively, the model learns to predict "not fraud" for everything and gets 99.8% accuracy while being useless
- Learn about: precision-recall tradeoff, why accuracy is a bad metric here, what ROC-AUC and F1 actually mean

**Concept 2 — Preprocessing pipeline**
- Scale `Amount` and `Time` using `StandardScaler` — the PCA features are already scaled but these two aren't
- Build the scaler as a `scikit-learn` pipeline so the same scaler can be applied at inference time
- The scaler must be saved alongside the model — if you scale differently at training vs inference, your predictions will be wrong

**Concept 3 — Handling class imbalance**
Three approaches to consider (pick one or combine):
- `scale_pos_weight` in XGBoost: tells the model to penalise missing fraud cases more heavily — simplest approach
- SMOTE (Synthetic Minority Over-sampling): generates synthetic fraud examples — available in `imbalanced-learn`
- Threshold tuning: train normally, then adjust the decision threshold from 0.5 to something lower (e.g. 0.3) to catch more fraud at the cost of more false positives
- Recommended: start with `scale_pos_weight`, then try threshold tuning, evaluate both

**Concept 4 — XGBoost Classifier**
- Understand the key hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`
- Use `GridSearchCV` or manual tuning — no Optuna needed (too heavy for Colab free tier)
- Evaluate using: Precision-Recall AUC, F1-score, Confusion Matrix — NOT accuracy
- Target: F1 score > 0.80 on the test set is solid for a portfolio project

**Concept 5 — Isolation Forest**
- A completely separate unsupervised model — it has never seen the fraud labels
- It flags transactions that are structurally unusual regardless of whether they match the fraud pattern
- Key parameter: `contamination` — set to the expected fraud rate (~0.002)
- Its output is a score (negative = more anomalous) — you'll normalise this to 0–1
- This is your `anomaly_score` field — independent signal from `fraud_probability`

**Concept 6 — SHAP Values**
- SHAP explains WHY the model made a specific prediction for a specific transaction
- For XGBoost, use `TreeExplainer` — fast and exact
- Output: a SHAP value per feature per transaction — positive means "pushed toward fraud", negative means "pushed toward legit"
- At inference time you only need the top 3–5 features by absolute SHAP value
- This is what feeds the LLM investigator — the LLM needs to know *which features mattered*, not just the score

**Concept 7 — Drift baseline**
- After training, compute the mean and standard deviation of each feature on your training data
- Save these statistics alongside your model
- At inference time, you'll compare incoming data distributions against this baseline using the KS test

**Concept 8 — Serialisation**
- Save everything as a single dictionary using `joblib.dump`: `{"model": xgb_model, "scaler": scaler, "iso_forest": iso_model, "shap_explainer": explainer, "feature_baseline": baseline_stats}`
- Load this once at FastAPI startup — not on every request
- Commit `model/model.pkl` to Git so Docker can copy it into the image

### Evaluation checklist before moving to Phase 2
- [ ] F1 score on test set > 0.80
- [ ] Precision-Recall curve looks reasonable (AUC > 0.85)
- [ ] SHAP values produce non-trivial explanations for a few sample transactions
- [ ] `model.pkl` loads correctly in a plain Python script outside Colab
- [ ] Scaler transforms correctly on a single sample input

---

## Phase 2 — ML Engine (`app/ml/inference.py`)

**STATUS: NOT STARTED**
**Estimated time: 1 day**
**Resume point: check which function below is not yet returning correct output**

### What this file is responsible for
Loading `model.pkl` at startup and exposing a single function that takes a transaction dict and returns `fraud_probability`, `anomaly_score`, and `top_shap_factors`. Nothing else. No FastAPI, no LLM, no storage.

#### [Files to Implement]
- **`app/ml/inference.py`**: Model loading (lifespan) and prediction logic.
- **`app/ml/preprocessing.py`**: Scaling of `Amount` and `Time` features.
- **`app/ml/utils.py`**: Helper functions for SHAP impact extraction.

### Concepts to implement

**Concept 1 — Startup loading pattern**
- FastAPI has a `lifespan` context manager for startup/shutdown events
- Load `model.pkl` once at startup, store it in a module-level variable
- Never load it inside the prediction function — that would reload it on every request
- Understand: why startup loading matters for latency

**Concept 2 — Input preprocessing at inference**
- Incoming JSON must be transformed the same way training data was transformed
- Apply the saved `StandardScaler` to `Amount` and `Time` before passing to the model
- Feature order matters — the model expects features in the exact same column order as training
- Use a fixed list of feature names to enforce order

**Concept 3 — Probability output**
- XGBoost's `predict_proba()` returns `[prob_legit, prob_fraud]`
- You want index `[1]` — the fraud probability
- This is a float between 0 and 1

**Concept 4 — Anomaly score normalisation**
- Isolation Forest's `decision_function()` returns raw scores (negative = more anomalous)
- Normalise to 0–1 so it's comparable to `fraud_probability`
- Simple approach: clip to a reasonable range, then apply min-max normalisation
- Higher score = more anomalous after normalisation

**Concept 5 — SHAP at inference**
- Run `TreeExplainer` on the single transaction
- Extract the top N features by absolute SHAP value
- Return as a list of `{"feature": name, "impact": value}` dicts
- The sign of impact matters — negative means pushed toward legit, positive toward fraud

### Output this module must produce
```
{
  "fraud_probability": 0.91,
  "anomaly_score": 0.78,
  "top_shap_factors": [
    {"feature": "V17", "impact": -0.31},
    {"feature": "Amount", "impact": 0.22},
    {"feature": "V14", "impact": -0.18}
  ]
}
```

---

## Phase 3 — LLM Layer

**STATUS: NOT STARTED**
**Estimated time: 2 days**
**Resume point: check which of the three LLM roles below has a working function, continue from the next**

This is the most important phase for differentiation. The LLM plays three distinct, non-interchangeable roles. Understand each role's purpose before writing any code.

---

### Role 1 — Fraud Investigator (`app/llm/llm_investigator.py`)

**What it does**: Takes the ML output (scores + SHAP factors) and produces a structured case note that a human fraud analyst would find actionable. This runs on every single `/predict` call.

**Why the ML model cannot do this**: The ML model outputs numbers. Numbers alone cannot explain context, cannot reference time-of-day patterns, cannot write "this matches card-testing behaviour" — that requires language and reasoning.

#### [Files to Implement]
- **`app/llm/client.py`**: Shared Groq client initialization (Singleton).
- **`app/llm/prompts.py`**: Central storage for formatting and system prompts for all 3 roles.
- **`app/llm/llm_investigator.py`**: Logic for converting ML scores into structured case notes.

**Concepts to learn**

*Prompt engineering for structured output*
- The prompt must provide all context the LLM needs: fraud probability, anomaly score, SHAP factor names and impacts, transaction amount, time of day
- Structure the prompt to request a specific output format: RISK LEVEL, PRIMARY SIGNALS, CONTEXT, RECOMMENDED ACTION
- The LLM should not be asked to compute the probability — the ML model already did that. The LLM's job is interpretation, not calculation
- Use a system prompt that sets the persona: "You are a senior fraud analyst at a financial institution..."

*Temperature and determinism*
- Set temperature low (0.1–0.2) for this role — you want consistent, factual analysis, not creative variation
- High temperature is for creative tasks; fraud analysis should be reproducible

*Output validation*
- Parse the LLM response to ensure the expected sections are present
- If the LLM returns something unparseable, fall back to a template-based summary rather than crashing the API

**What the output should look like**
```
RISK LEVEL: HIGH

PRIMARY SIGNALS:
V17 showed a strong negative deviation (-0.31 SHAP impact), consistent
with unusual merchant category sequencing seen in card-testing attacks.
Transaction amount is 4.1x above the 90-day account median.

CONTEXT:
Anomaly score of 0.78 confirms this transaction is structurally
different from baseline — not a one-off statistical outlier. Occurred
at 02:14 local time, outside normal activity window.

RECOMMENDED ACTION:
Hold transaction. Trigger 2FA challenge to cardholder before processing.
```

---

### Role 2 — Risk Override (`app/llm/llm_override.py`)

**What it does**: Only activates when `fraud_probability` is between 0.4 and 0.7 (the uncertain zone). The LLM evaluates additional contextual signals and either escalates to HIGH or clears to LOW, with a recorded reason. This *changes the final verdict* — not just annotates it.

**Why this role is architecturally significant**: The LLM is in the decision path, not just the output path. The final risk level that gets returned to the client may differ from what the ML model said. `was_overridden: true` becomes a first-class field in your response.

**Concepts to learn**

#### [Files to Implement]
- **`app/llm/llm_override.py`**: Logic for Eskalate/Clear decisions in the uncertain ML score range (0.4 - 0.7).

*Conditional LLM invocation*
- Only call this when `0.4 <= fraud_probability <= 0.7`
- For clear LOW (< 0.4) and clear HIGH (> 0.7) cases, skip this call entirely to save latency and API quota
- This is an important architectural decision: not every prediction needs all components

*Context for override decisions*
- The LLM needs signals the ML model doesn't have direct access to: time of day, whether this is the first international transaction, number of transactions in the last hour, account age
- These extra fields can be optional inputs in your request schema — users who provide them get richer override decisions
- The prompt should ask for a binary decision (ESCALATE or CLEAR) plus a one-sentence reason

*Logging override decisions*
- Every override must be logged to the `llm_decisions` SQLite table with the original ML score, the LLM verdict, and the reason
- This creates an audit trail — a real production requirement for fraud systems
- This log also powers your `/stats` endpoint's `override_rate` field

**What the output should produce**
```
{
  "decision": "ESCALATE",
  "reason": "First international transaction in 180 days combined with 
             2AM timestamp and anomaly score of 0.74 suggests account 
             compromise despite borderline ML probability."
}
```

---

### Role 3 — Natural Language Query Engine (`app/llm/llm_query.py`)

**What it does**: A separate endpoint (`POST /query`) that accepts a plain-English question about past predictions, converts it to SQL, runs it against the SQLite database, and returns both the raw results and a plain-English interpretation.

**Why this role showcases a different LLM skill**: Roles 1 and 2 are about reasoning over a single transaction. Role 3 is text-to-SQL — converting natural language to structured queries. This is a distinct, industry-relevant LLM capability.

**Concepts to learn**

#### [Files to Implement]
- **`app/llm/llm_query.py`**: Logic for converting Natural Language to SQL (and interpreting results).
- **`app/services/llm_query_service.py`**: Orchestration of SQL execution and interpretation flow.

*Text-to-SQL prompting*
- The LLM needs to know your database schema to write correct SQL — include the full `CREATE TABLE` statement in the system prompt
- Ask the LLM to return only the SQL query, nothing else, in the first response
- Then execute the SQL using Python's `sqlite3` module
- Pass the results back to the LLM in a second call asking for a plain-English interpretation
- This two-step approach (generate SQL → run → interpret) is more reliable than asking for everything at once

*SQL injection safety*
- You are generating SQL from LLM output — never run raw LLM output against a production database without validation
- For a portfolio project: validate that the generated SQL contains only SELECT statements, no INSERT/UPDATE/DELETE/DROP
- Reject and return an error if the SQL contains write operations

*Handling failed queries*
- The LLM will sometimes generate syntactically incorrect SQL
- Wrap the SQL execution in a try/except, and on failure ask the LLM to fix the query given the error message
- One retry attempt is sufficient — if it fails twice, return a friendly error

**Example queries the endpoint should handle**
```
"How many HIGH risk transactions were flagged today?"
"What are the top 5 most common fraud-triggering features this week?"
"Show me all transactions that were overridden by the LLM yesterday"
"What's the average fraud probability over the last 7 days?"
```

---

## Phase 4 — Risk Engine (`app/services/risk_engine.py`)

**STATUS: NOT STARTED**
**Estimated time: half a day**
**Resume point: re-read the "Merging logic" concept below and check which case in the logic tree is not yet implemented**

### What this file is responsible for
Combining outputs from the ML engine and LLM layer into a single final verdict. This is the only file that touches both — `ml_engine.py` and `llm_*.py` should never import each other.

#### [Files to Implement]
- **`app/services/risk_engine.py`**: The "brain" that merges ML scores, LLM overrides, and anomaly signals into the final verdict.

### Concepts to implement

**Merging logic**
The decision tree for `final_risk_level`:
```
fraud_probability >= 0.7  →  final = HIGH   (skip LLM override)
fraud_probability <= 0.3  →  final = LOW    (skip LLM override)
0.3 < fraud_probability < 0.7  →  call LLM override
    LLM says ESCALATE  →  final = HIGH,  was_overridden = True
    LLM says CLEAR     →  final = LOW,   was_overridden = True
    LLM call fails     →  final = MEDIUM, was_overridden = False (fallback)
```

**Latency measurement**
- Record `time.perf_counter()` at the start of the full prediction flow and at the end
- Subtract to get `latency_ms`
- This is a real production metric — showing it in the response demonstrates you think about performance

**Combining anomaly score**
- If `fraud_probability < 0.7` but `anomaly_score > 0.85`, consider escalating to HIGH regardless
- This represents cases where the supervised model (which learned from labelled fraud) is uncertain, but the unsupervised model (which detects structural outliers) is very confident
- Add this as a separate condition in your merging logic

---

## Phase 5 — FastAPI Layer (`app/main.py`, `app/schemas/schemas.py`)

**STATUS: NOT STARTED**
**Estimated time: 1 day**
**Resume point: check which endpoint below returns a correct response when tested with curl or Postman**

### `app/schemas.py` — Pydantic models

#### [Files to Implement]
- **`app/main.py`**: FastAPI app setup, exception handlers, and lifespan management.
- **`app/schemas/schemas.py`**: Pydantic models for `/predict` and `/query`.
- **`app/api/routes.py`**: Definition of all REST endpoints.
- **`app/api/dependencies.py`**: Shared logic for DB sessions and security.
- **`app/api/middleware.py`**: Custom implementation of API key auth and rate limiting.

**Concepts to implement**

*Input schema for `/predict`*
- Required fields: `transaction_id` (string), `amount` (float), `V1` through `V28` (28 floats)
- Optional fields: `hour_of_day` (int), `is_international` (bool), `transactions_last_hour` (int) — used by LLM override for richer context
- Use Pydantic's `Field` with `ge` (greater than or equal) and `le` constraints on `amount` — shows you understand validation

*Output schema*
- Mirror the response shape defined in the Architecture section above
- Make `override_reason` and `llm_case_summary` `Optional[str]` — they may be absent if the LLM call failed

*Input schema for `/query`*
- Single field: `question` (string), with a max length constraint

### `app/main.py` — Routes and middleware

**Concepts to implement**

*API key authentication middleware*
- Read the API key from the `X-API-Key` request header
- Compare against `API_SECRET_KEY` from your `.env` file
- Return HTTP 401 if missing or wrong
- Apply to all routes except `/health`
- Concept: why API key auth is appropriate here vs JWT (simpler, stateless, no token refresh needed for a machine-to-machine API)

*Rate limiting with slowapi*
- Attach to the `/predict` endpoint: max 10 requests per minute per IP
- Return HTTP 429 with a clear message when exceeded
- Concept: why rate limiting matters for ML APIs specifically (inference is compute-expensive)

*The `/health` endpoint*
- Returns `{"status": "ok", "model_loaded": bool}`
- Check that `model.pkl` was loaded successfully at startup
- Render uses this to verify the service is alive

*The `/predict` endpoint*
- Call `ml_engine.predict()` → call `risk_engine.merge()` → call `storage.log_prediction()` as a background task
- Background task: logging should not block the response. FastAPI's `BackgroundTasks` lets you log to SQLite after the response is sent
- Concept: why background tasks matter for latency

*The `/stats` endpoint*
- Query SQLite for aggregate stats: total predictions, high risk count, override rate, average fraud probability
- This replaces Prometheus+Grafana — same data, zero infrastructure
- Return as a clean JSON response

*The `/query` endpoint*
- Accept the question, call `llm_query.run()`, return the result
- Add a note in the response if the SQL had to be retried

*CORS middleware*
- Add `CORSMiddleware` allowing all origins for development
- In a real production system you'd restrict this — mention it in your README

---

## Phase 6 — Storage Layer (`app/storage/storage_handler.py`)

**STATUS: NOT STARTED**
**Estimated time: half a day**
**Resume point: test each function independently with a sample dict before wiring to FastAPI**

### What this file is responsible for
All SQLite reads and writes. No ML, no LLM, no FastAPI logic here.

#### [Files to Implement]
- **`app/storage/storage_handler.py`**: Centralized SQLite operations (logging predictions, fetching stats).

### Database tables to create

**`predictions` table**
- `id` (auto-increment primary key)
- `timestamp` (datetime, default current)
- `transaction_id` (text)
- `fraud_probability` (real)
- `anomaly_score` (real)
- `ml_risk_level` (text)
- `final_risk_level` (text)
- `was_overridden` (integer — SQLite has no boolean)
- `top_features` (text — store as JSON string)
- `latency_ms` (real)

**`llm_decisions` table**
- `id`, `timestamp`, `transaction_id`
- `original_ml_prob` (real)
- `llm_verdict` (text — ESCALATE or CLEAR)
- `reason` (text)

**`drift_buffer` table**
- `id`, `timestamp`
- `feature_name` (text)
- `feature_value` (real)
- Keep only the last 500 rows per feature (delete older rows on insert)

### Concepts to implement

*Create tables on startup*
- Use `CREATE TABLE IF NOT EXISTS` so running the app multiple times doesn't error
- Call this from FastAPI's startup lifespan event alongside model loading

*Thread safety*
- SQLite with Python's `sqlite3` module is not fully thread-safe by default
- Use `check_same_thread=False` in the connection, or use a connection-per-request pattern
- For a portfolio project, connection-per-request is simplest and safe

---

## Phase 7 — Drift Detection (`monitoring/drift.py`)

**STATUS: NOT STARTED**
**Estimated time: half a day**
**Resume point: the KS test function is the core — get that working first, then wire to storage**

### What this module is responsible for
Detecting when incoming transaction data has shifted significantly from the training distribution. This answers: "Is the model still seeing the same kind of data it was trained on?"

#### [Files to Implement]
- **`monitoring/drift.py`**: Implementation of the KS test and drift baseline comparison logic.

### Concepts to implement

**Kolmogorov-Smirnov test**
- The KS test compares two distributions and tells you whether they are likely from the same underlying distribution
- You have: a reference distribution (mean/std from training data) and a sample of recent incoming values (from `drift_buffer`)
- `scipy.stats.ks_2samp(reference_sample, recent_sample)` returns a statistic and a p-value
- If `p_value < 0.05`, the distributions differ significantly — flag drift
- Concept: understand what p-value means here (probability that the difference occurred by chance)

**When to run the check**
- Don't run on every prediction — that's wasteful
- Run every 100 predictions (or on a schedule if you add a background worker)
- Simplest: call `check_drift()` from the `/stats` endpoint — it checks the buffer and returns `drift_alert: true/false`

**What to do when drift is detected**
- Log it to stderr (simple, visible in Render logs)
- Set `drift_alert: true` in the `/stats` response
- In a real system you'd trigger model retraining — for a portfolio project, document this as "next step" in your README

---

## Phase 8 — Docker (`Dockerfile`)

**STATUS: NOT STARTED**
**Estimated time: 2–3 hours**
**Resume point: run `docker build` and `docker run` locally — if the container starts and `/health` returns ok, this phase is done**

### Philosophy
Docker is the packaging layer, not the development environment. You develop with `uvicorn` locally. Docker exists so that Render gets a container with every dependency baked in — no "works on my machine" issues.

### Concepts to understand

**Why a single-stage build is fine here**
- Multi-stage builds (used in the original plan) are for minimising image size by separating build tools from runtime
- For a portfolio project, a single `python:3.11-slim` base image is clean and sufficient
- `python:3.11-slim` gives you a minimal Debian image with Python — no extras

**Layer caching**
- Docker builds images in layers. Each instruction (`FROM`, `COPY`, `RUN`) is a layer
- Copy `requirements.txt` and run `pip install` before copying your source code
- This way, if you only change source code, Docker reuses the cached `pip install` layer — faster rebuilds
- Understand: if you copy everything first and then `pip install`, every code change triggers a full reinstall

**What the Dockerfile needs to do**
1. Start from `python:3.11-slim`
2. Set `/app` as the working directory
3. Copy `requirements.txt` and install dependencies
4. Copy the rest of the project (including `model/model.pkl`)
5. Expose port 8000
6. Set the `CMD` to run uvicorn

**Environment variables in Docker**
- Your `.env` file is NOT copied into the image (it's in `.gitignore`)
- Environment variables are injected at runtime — by Render's dashboard in production, or by `docker run -e` locally
- This is the correct approach: secrets never live inside the image

**Testing Docker locally**
- `docker build -t fraud-api .` — builds the image
- `docker run -p 8000:8000 -e GROQ_API_KEY=xxx -e API_SECRET_KEY=xxx fraud-api` — runs it
- Hit `localhost:8000/health` — should return `{"status": "ok"}`
- If this works, Render will work

---

## Phase 9 — Render Deployment (`render.yaml`)

**STATUS: NOT STARTED**
**Estimated time: 1–2 hours**
**Resume point: check Render dashboard → your service → logs tab for the specific error**

### Concepts to understand

**How Render uses `render.yaml`**
- `render.yaml` in your repo root tells Render how to build and run your service
- It specifies: service type (web), runtime (docker), plan (free), which port to expose, and which environment variables are required
- On every `git push` to `main`, Render detects the change, builds the Docker image, and deploys it
- Zero manual steps after initial setup

**Free tier constraints**
- Render's free tier spins down after 15 minutes of inactivity and takes ~30 seconds to spin back up on the next request
- This is a known limitation — mention it in your README and tell viewers to hit `/health` first to wake the service
- The service gets 512MB RAM — your model.pkl must fit comfortably within this, XGBoost models for this dataset typically are 5–20MB

**Environment variables on Render**
- After connecting your GitHub repo, go to Render dashboard → Environment tab
- Add `GROQ_API_KEY` and `API_SECRET_KEY` manually (never in `render.yaml` — that file is committed to Git)
- `render.yaml` should reference them with `sync: false` to indicate they must be set externally

**What `render.yaml` should contain**
- `type: web`
- `name: fraud-detection-api`
- `runtime: docker`
- `plan: free`
- `envVars`: list of required variable names with `sync: false`

**Verifying deployment**
- Render gives you a URL like `https://fraud-detection-api.onrender.com`
- Hit `/health` — if it returns `{"status": "ok", "model_loaded": true}`, everything is working
- Check the Logs tab if anything fails — it shows stdout/stderr from your container

---

## Phase 10 — Testing (`tests/test_predict.py`)

**STATUS: NOT STARTED**
**Estimated time: half a day**
**Resume point: run `pytest` and check which test in the list below is failing**

### Why tests matter for a portfolio project
Tests are how you show you write production code, not just scripts. A reviewer who sees a `tests/` folder with passing tests immediately categorises you differently.

### Tests to write

**Unit tests**
- `test_risk_engine_high_risk`: give a probability of 0.9, verify `final_risk_level` is HIGH and `was_overridden` is False
- `test_risk_engine_low_risk`: probability 0.1, verify LOW
- `test_risk_engine_medium_escalated`: probability 0.55, mock the LLM override to return ESCALATE, verify final is HIGH and `was_overridden` is True
- `test_shap_output_format`: verify SHAP output is a list of dicts with `feature` and `impact` keys
- `test_anomaly_normalisation`: verify the normalised anomaly score is between 0 and 1

**Integration tests (using FastAPI's `TestClient`)**
- `test_predict_endpoint_valid`: send a valid transaction, verify the response has all required fields
- `test_predict_endpoint_missing_field`: send a request missing `V1`, verify a 422 validation error
- `test_predict_endpoint_no_api_key`: send without the `X-API-Key` header, verify 401
- `test_health_endpoint`: verify `/health` returns 200 with `model_loaded: true`
- `test_stats_endpoint`: verify `/stats` returns the expected shape

**Mocking LLM calls in tests**
- Never call Groq in tests — it's slow, costs quota, and makes tests non-deterministic
- Use `unittest.mock.patch` to replace the Groq client with a mock that returns a fixed response
- Understand: why testing with real external APIs is bad practice

---

## Phase 11 — README

**STATUS: NOT STARTED**
**Estimated time: 2–3 hours**
**Resume point: check which section below is missing from the current README**

A great README is part of the project. A recruiter or reviewer should understand what the project does, why it's interesting, and how to try it — in under 2 minutes.

### Sections to include

**Project overview** (3–5 sentences)
What the project is, what problem it solves, what makes it interesting technically. Mention all three technologies by name.

**Architecture diagram**
Paste or link the architecture diagram. Explain in one paragraph the role of each component.

**The three LLM roles** (this is your differentiator — explain it clearly)
Explain why each role is genuinely necessary — not just a wrapper around the ML output. Show the example response with `was_overridden: true`.

**Live demo**
Your Render URL. Explain that free tier spins down — hit `/health` first. Include a sample curl command for `/predict`.

**API reference** (short table)
| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run full fraud detection pipeline |
| `/stats` | GET | Aggregate stats from prediction log |
| `/query` | POST | Natural language query over prediction history |
| `/explain/{id}` | GET | Retrieve stored explanation for a past prediction |
| `/health` | GET | Service health check |

**Tech stack table**
List every tool with one-line description of its role.

**Local development** (step by step)
Clone → create venv → install requirements → add `.env` → `uvicorn app.main:app --reload`

**Known limitations / next steps**
- Render free tier cold start
- Model retraining not yet automated on drift detection
- JWT auth would replace API key for multi-user scenarios
- Prometheus + Grafana would replace `/stats` in a higher-traffic deployment

---

## Checkpoint Map (for resuming after a break)

Use this table as your single source of truth on project state.

| Phase | Description | Status | Notes |
|---|---|---|---|
| 0 | Environment setup | NOT STARTED | |
| 1 | ML training in Colab | NOT STARTED | |
| 2 | ML engine (`ml_engine.py`) | NOT STARTED | |
| 3a | LLM Role 1 — Investigator | NOT STARTED | |
| 3b | LLM Role 2 — Risk override | NOT STARTED | |
| 3c | LLM Role 3 — NL query | NOT STARTED | |
| 4 | Risk engine | NOT STARTED | |
| 5 | FastAPI routes + schemas | NOT STARTED | |
| 6 | Storage (SQLite) | NOT STARTED | |
| 7 | Drift detection | NOT STARTED | |
| 8 | Docker | NOT STARTED | |
| 9 | Render deployment | NOT STARTED | |
| 10 | Tests | NOT STARTED | |
| 11 | README | NOT STARTED | |

**How to update**: when you stop working on a phase, change its status to `IN PROGRESS` and add a note like "completed concepts 1–3, stopped before implementing SHAP". When done, mark `DONE`.

**Quick resume**: `Ctrl+F` → search `IN PROGRESS` → that section is where you are.

---

## Dependency Order (what must be done before what)

```
Phase 0 (setup)
    └── Phase 1 (train model)
            └── Phase 2 (ml_engine.py)
                    ├── Phase 3a (llm_investigator.py)   ← needs SHAP output
                    ├── Phase 3b (llm_override.py)       ← needs fraud_probability
                    └── Phase 3c (llm_query.py)          ← independent, can be done anytime after Phase 0
                            └── Phase 4 (risk_engine.py) ← needs Phase 2 + 3a + 3b
                                    └── Phase 5 (FastAPI) ← needs Phase 4 + 6
                                            └── Phase 6 (Storage)   ← can start after Phase 0
                                                    └── Phase 7 (Drift) ← needs Phase 6
                                                            └── Phase 10 (Tests) ← needs Phase 5
                                                                    └── Phase 8 (Docker) ← needs all above
                                                                            └── Phase 9 (Render)
                                                                                    └── Phase 11 (README)
```

> Phases 3c, 6, and 7 can be developed in parallel with other phases as soon as Phase 0 is done. They have no hard dependencies on each other.

---

## Common Mistakes to Avoid

**ML**
- Training the scaler on the test set — only fit the scaler on training data, then transform both train and test
- Evaluating with accuracy — use F1 and Precision-Recall AUC only
- Forgetting to save the scaler inside `model.pkl` — you will get wrong predictions at inference

**LLM**
- Asking the LLM to compute the fraud probability itself — the ML model does that, the LLM interprets it
- Not setting temperature low for Roles 1 and 2 — high temperature gives inconsistent analysis
- Running LLM override on every request — only run it in the 0.4–0.7 band

**FastAPI**
- Loading `model.pkl` inside the prediction function — load it once at startup
- Not using background tasks for logging — synchronous SQLite writes add ~5–20ms to every request

**Docker**
- Copying source before installing requirements — breaks layer caching
- Putting secrets in the Dockerfile or `render.yaml` — use environment variables only

**General**
- Committing `.env` to Git — double-check `.gitignore` before every push
- Not testing the Docker container locally before pushing to Render — always run it locally first

---

*Document version: 1.0 — Generated for the Free Fraud Detection API project*
*Last updated: Phase 0 not yet started*