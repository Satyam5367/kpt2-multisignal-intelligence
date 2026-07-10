# KPT2 — Multi-Signal Kitchen Prep Time Intelligence

A causally-simulated system for correcting noisy, merchant-reported
"Food Order Ready" (FOR) signals in a food-delivery pipeline. It fuses
four independent signals — bias-corrected merchant timestamps, a
graph-resolved kitchen workflow estimate, a real-time kitchen load
score, and per-merchant historical patterns — into a single **Composite
Readiness Score (CRS)** that materially outperforms the raw signal, and
a **Shadow Model** that estimates prep time without trusting the
merchant's timestamp at all.

> This is a synthetic, causally-modeled simulation (not production
> data), built to demonstrate the modeling approach end-to-end —
> including where a naive version of this idea goes wrong. See
> [Design notes](#design-notes--what-i-fixed-from-v1) below.

## Results

Evaluated on a held-out, time-ordered test split (never seen during
model fitting):

| Model | MAE (min) | P50 error | P90 error |
|---|---|---|---|
| Baseline (raw merchant FOR) | 1.97 | 1.82 | 3.83 |
| **CRS (composite)** | **0.97** | **0.84** | **1.96** |
| Shadow (FOR-independent) | 1.60 | 1.38 | 3.14 |

CRS roughly **halves** the mean absolute error versus the raw
merchant-marked signal. The Shadow Model — which is given **no access
to the merchant's FOR timestamp at all** — still beats the naive
baseline, which is the whole point of having it: it's an independent
cross-check that can't be fooled by a merchant who is deliberately or
accidentally gaming their own FOR marking.

![Error distributions](assets/error_distributions.png)

Run `python main.py` yourself to regenerate these numbers (they'll
differ slightly run-to-run only if you change `--seed`).

## Why the raw signal is noisy in the first place

Merchants mark an order "ready" based on their own internal process,
which is systematically biased in two ways this project models
explicitly:

1. **Fixed per-merchant bias** — some merchants habitually mark ready
   early or late. This part is straightforward to fix: average the
   historical offset per merchant and subtract it out (`merchant_bias.py`).
2. **Rush-hour distortion** — during high kitchen congestion, staff
   mark orders ready earlier than they actually are, to keep the queue
   moving. This part **cannot** be fixed by a single per-merchant
   average, because it varies order-to-order with how busy the kitchen
   is at that moment. This is exactly the residual error that the
   Kitchen Load Score and workflow-graph signal are designed to claw
   back — and it's why CRS beats simple bias correction alone.

![Merchant Bias Index](assets/merchant_bias_index.png)

## Architecture

```
                     ┌─────────────────────┐
                     │   Order Simulator    │  heap-based congestion sweep
                     │   (simulation.py)     │  sliding-window rider clustering
                     └──────────┬───────────┘
                                │
                ┌───────────────┼───────────────┬──────────────────┐
                ▼               ▼               ▼                  ▼
      ┌─────────────────┐ ┌───────────┐ ┌───────────────┐ ┌──────────────────┐
      │ Merchant Bias    │ │ Kitchen   │ │ Workflow Graph │ │ Historical        │
      │ correction       │ │ Load Score│ │ (DAG + CPM)    │ │ Pattern (deque)   │
      └────────┬─────────┘ └─────┬─────┘ └───────┬────────┘ └─────────┬─────────┘
               │                 │               │                    │
               └────────┬────────┴───────┬───────┴──────────┬─────────┘
                         ▼                ▼                   ▼
                ┌──────────────────┐          ┌───────────────────────┐
                │ Composite         │          │ Shadow Model          │
                │ Readiness Score   │          │ (FOR-independent)     │
                │ (fitted, w/       │          └───────────┬───────────┘
                │ congestion        │                      │
                │ interaction term) │                      │
                └────────┬──────────┘                      │
                         └───────────────┬───────────────────┘
                                         ▼
                              ┌───────────────────┐
                              │ Evaluation +        │
                              │ Drift Monitoring     │
                              └──────────┬───────────┘
                                         ▼
                              ┌───────────────────┐
                              │ MySQL / SQLAlchemy  │
                              │ (persistence layer)  │
                              └───────────────────┘
```

### Kitchen workflow dependency graph

A kitchen order isn't one number — it's a small DAG of prep steps, some
of which can run in parallel and some of which must wait on others. The
true minimum prep time is the **longest path** through that graph (the
critical path), not the sum of every step. `workflow_graph.py`
implements this from scratch:

- **Kahn's algorithm** (BFS with an in-degree-counted queue, via
  `collections.deque`) for topological sorting.
- A single **dynamic-programming pass** over the topological order to
  compute the critical path length (the standard Critical Path Method
  used in project scheduling).

![Workflow graph](assets/workflow_graph.png)

### Data structures used deliberately, not decoratively

| Technique | Where | Why |
|---|---|---|
| Adjacency-list graph + topological sort + DP longest-path | `workflow_graph.py` | Resolve step dependencies, compute critical path |
| Min-heap (`heapq`) interval-overlap sweep | `simulation.py` | O(n log n) estimate of concurrent in-kitchen orders ("meeting rooms" pattern) |
| Sliding window (two-pointer) | `simulation.py` | O(n) rider-arrival clustering density per merchant |
| Bounded deque (`collections.deque(maxlen=...)`) | `historical_tracker.py` | O(1)-amortized rolling per-merchant historical average |
| Hash map (`dict`) | `merchant_bias.py`, `db_manager.py` | O(1) merchant_id → bias / row lookups |

## Project structure

```
kpt2-multisignal-intelligence/
├── main.py                       # CLI entry point
├── pyproject.toml
├── requirements.txt
├── src/kpt2/
│   ├── config.py                 # every tunable constant, in one place
│   ├── simulation.py              # causal order-lifecycle simulator
│   ├── workflow_graph.py          # graph DSA: topo sort + critical path
│   ├── merchant_bias.py           # Merchant Bias Index + correction
│   ├── kitchen_load.py            # Kitchen Load Score
│   ├── historical_tracker.py      # rolling per-merchant pattern (deque)
│   ├── composite_scoring.py       # CRS: fitted, congestion-adaptive
│   ├── shadow_model.py            # FOR-independent shadow estimate
│   ├── drift_monitor.py           # merchant drift / instability flags
│   ├── evaluation.py              # MAE / P50 / P90 comparisons
│   ├── visualization.py           # headless plot generation
│   └── db/
│       ├── schema.sql             # canonical MySQL DDL (InnoDB, indexed FKs)
│       └── db_manager.py          # SQLAlchemy Core read/write layer
├── tests/                         # pytest suite (17 tests)
└── assets/                        # committed sample plots for this README
```

## Setup

```bash
git clone <this-repo>
cd kpt2-multisignal-intelligence
pip install -r requirements.txt
```

## Usage

```bash
# Run the full pipeline with defaults (3,000 orders, 60 merchants)
python main.py

# Larger simulation
python main.py --num-orders 8000 --num-merchants 120

# Persist results to MySQL
python main.py --store-db --db-url "mysql+pymysql://user:password@localhost:3306/kpt2"

# Persist results to a local SQLite file (no MySQL server needed)
python main.py --store-db --db-url "sqlite:///kpt2_local.db"
```

Plots are written to `output/` (git-ignored; regenerate anytime).

### MySQL setup

```bash
mysql -u root -p < src/kpt2/db/schema.sql
python main.py --store-db --db-url "mysql+pymysql://root:<password>@localhost:3306/kpt2"
```

`schema.sql` is the canonical, hand-authored DDL (InnoDB, `ENUM`
columns, composite indexes on `(merchant_id, order_time)` and
`model_name` matching the pipeline's actual query patterns, foreign
keys with `ON DELETE CASCADE`). `db_manager.py` is a dialect-portable
SQLAlchemy Core layer that targets the same logical schema — pointed at
MySQL in production, or at SQLite for local development/tests without
needing a server running.

## Tests

```bash
pip install -r requirements.txt
pytest -v
```

17 tests covering: topological-sort correctness, critical-path-not-sum
correctness, cycle detection, bias-index arithmetic, KLS normalization,
leakage-free rolling history, and — most importantly — an end-to-end
regression test asserting **CRS and Shadow both beat the raw baseline
on the held-out split**, so this project can't silently regress back to
the mistake described below.

## Design notes — what I fixed from v1

An earlier version of this project generated most "signals" as
independent random noise with no causal link to the ground-truth prep
time, and blended them with fixed, hand-picked weights. Empirically,
mixing uncorrelated noise into a signal via a weighted sum can only
dilute it — and that version's own evaluation showed the "improved"
composite score performing *worse* than the naive baseline it was
supposed to beat, plus a crash in the evaluation code that meant it
never actually printed final results.

This version fixes both problems at the root:

- Every input signal (`active_orders`, `wait_cluster`,
  `workflow_duration_estimate`, `historical_pattern`) is now generated
  by, or estimated from, a real causal process — kitchen congestion is
  simulated with a min-heap interval sweep, rider clustering with a
  sliding window, and prep time itself is the critical path through a
  dependency graph rather than an unrelated random draw.
- CRS is a **fitted** linear model (scikit-learn) evaluated on a
  held-out, time-ordered split, not a static hand-picked blend — so its
  reported improvement is an honest out-of-sample number.
- The predictive workflow-duration estimator is deliberately given
  **imperfect calibration and independent per-order noise**, rather
  than direct access to the same formula used to generate ground
  truth, so the reported improvement margin is a defensible one, not an
  artifact of the simulation leaking its own answer key.

## Tech stack

Python · pandas · NumPy · scikit-learn · SQLAlchemy · MySQL · pytest ·
matplotlib

Core CS concepts applied: graphs (DAG, topological sort, longest-path
DP), heaps/priority queues, sliding windows, hash maps, deques, and a
relational schema with indexed, foreign-keyed tables.

## License

MIT — see `LICENSE`.
