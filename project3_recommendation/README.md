# Project 3 — Personalised Product Recommendation

Recommend products from past interactions, purchases and product similarity — evaluated
honestly on a chronological split, with an explicit answer to the question most
recommender projects skip: **what do you serve a product that nobody has interacted with
yet?**

---

## Results

6,000 users · 1,500 items · 190,975 interactions · 98.63% sparse · chronological split at
**2026-06-12** (last 20% of the timeline held out) · 4,604 evaluable users.

| Model | P@10 | R@10 | NDCG@10 | MAP@10 | Hit@10 | Coverage | Novelty | Cold recall@20 |
|---|---|---|---|---|---|---|---|---|
| Popularity | 0.0261 | 0.0782 | 0.0631 | 0.0373 | 0.2148 | 0.015 | 6.85 | 0.000 |
| Popularity (21-day decay) | 0.0263 | 0.0785 | 0.0631 | 0.0372 | 0.2168 | 0.015 | 6.86 | 0.000 |
| **Item-item CF** | **0.0484** | **0.1454** | **0.1128** | **0.0672** | **0.3545** | 0.177 | 7.68 | 0.000 |
| ALS matrix factorisation | 0.0403 | 0.1236 | 0.0929 | 0.0542 | 0.3104 | 0.512 | 8.88 | 0.000 |
| Content-based (TF-IDF) | 0.0072 | 0.0216 | 0.0141 | 0.0068 | 0.0678 | **0.957** | **12.64** | 0.0446 |
| Hybrid (CF + content fallback) | 0.0465 | 0.1399 | 0.1073 | 0.0635 | 0.3434 | 0.235 | 9.47 | **0.0475** |

**Headline:** item-item CF reaches NDCG@10 = 0.1128, **+78.7% over the popularity
baseline**, and triples hit rate (0.21 → 0.35).

### Four findings worth defending

**1. The popularity baseline is not a straw man.** The catalogue follows a power law —
the top 15 items account for 34.6% of all events — so "recommend what's hot" gets Hit@10
= 0.21 on its own. Any recommender project that does not report this number is hiding
how much of its performance is just popularity.

**2. Item-item CF beats matrix factorisation here (0.1128 vs 0.0929).** That is the
opposite of the usual expectation and worth being straight about: with a median of 14
training items per user, the neighbourhood method has enough co-occurrence signal to work
with, while 40 latent factors per user is a lot of parameters to fit from 14 observations.
ALS would be expected to overtake CF with denser histories or with more tuning — its
hyperparameters were left at sensible defaults rather than searched, which is the clearest
remaining improvement in this project.

**3. ALS is nonetheless the more *useful* model on one axis.** It covers 51% of the
catalogue against CF's 18%, with higher novelty (8.88 vs 7.68). A recommender that only
ever surfaces 18% of the catalogue has limited merchandising value regardless of its
NDCG, so the choice between these two is a product decision, not purely a metric one.

**4. The hybrid trades accuracy for reach, deliberately.** It gives up 4.9% of NDCG@10
(0.1128 → 0.1073) and in exchange goes from **zero** cold-start recall to 0.0475, surfacing
81 distinct never-before-seen items in users' top-10. Every purely collaborative model
scores exactly 0.0000 on cold items — not "poorly", but *structurally zero*, because an
item with no interactions has no row in the similarity matrix and no learned factor. If
new products matter to the business, that 4.9% is cheap.

### Performance by user history

NDCG@10, item-item CF: light users (1–5 items) **0.1504** · medium (6–20) **0.1169** ·
heavy (21+) **0.0973**.

Light users scoring *highest* is a measurement artefact, not a genuine quality ordering,
and it is worth stating plainly: heavy users have more held-out items, so the ideal DCG
they are normalised against is larger and harder to approach. The ranking is real within a
segment, not across them — which is exactly why the evaluation reports segments rather
than one average.

---

## How to run

```bash
pip install -r ../requirements.txt

python -m src.generate_data     # 191k interactions with latent-factor preferences
python -m src.data_prep         # chronological split, sparsity report
python -m src.evaluate          # fit all 5 models, tune hybrid, full metrics

streamlit run app.py                          # interactive explorer
uvicorn src.api:app --reload --port 8000      # recommendation API + /docs
```

`src.evaluate` must run before the app or the API, since both load its saved models.

---

## Approach

### 1. Interaction data (`generate_data.py`)

Users and items each get an 8-dimensional latent factor vector; preference is their dot
product. On top of that sit the things a real catalogue has: power-law popularity,
category affinity, price sensitivity, and a rating effect.

Four properties are built in on purpose:

- **Recoverable latent structure**, so matrix factorisation has something real to find.
- **Realistic funnel rates** — 112,862 views, 38,484 clicks, 26,398 carts, 13,231
  purchases. *(The first version of the generator produced almost as many purchases as
  views, which would have made the implicit-feedback weighting meaningless; the
  conversion rates were corrected to e-commerce levels.)*
- **Cold-start by construction** — 12% of items are released inside the final fifth of
  the window, so 88 of them appear in the test period with **zero** training interactions.
- **Staggered sign-ups**, giving 274 users with no training history at all.

### 2. Split and implicit feedback (`data_prep.py`)

**The split is chronological, never random.** A random split would let the model train on
a user's March purchase and be tested on their February one — predicting the past from the
future. The last 20% of the timeline is held out; everything before it trains.

Feedback is implicit, so each user-item pair gets a confidence weight
`1 + α·log1p(Σ event_weight)` with α = 12 (Hu, Koren & Volinsky 2008), where a purchase
counts 8× a view. The binary value says *whether* they engaged; the confidence says *how
much to trust it*.

**Relevance** = items interacted with in the test period that the user had **not** already
seen in training. Re-recommending last month's purchase is not a win, so those items are
excluded from both the ground truth and the candidate list.

### 3. Models (`recommenders.py`)

| Model | Mechanism |
|---|---|
| `PopularityRecommender` | weighted interaction counts, with an optional 21-day half-life |
| `ItemItemCF` | cosine similarity between item columns, shrunk by co-occurrence count, truncated to each item's top 120 neighbours |
| `ALSMatrixFactorization` | implicit-feedback ALS, implemented from scratch |
| `ContentBasedRecommender` | TF-IDF over category, subcategory, brand, tags and discretised price/rating bands |
| `HybridRecommender` | rank-level blend, with content substituted for cold items |

**ALS is written directly rather than imported** so the mechanics are explicit. The
identity that makes it tractable is

```
Yᵀ C_u Y  =  YᵀY  +  Y_uᵀ (C_u − I) Y_u
```

so the full `n_items × n_items` weighted Gram matrix is never formed — `YᵀY` is computed
once per iteration and only the user's own items contribute a correction. Convergence is
logged and plotted (`reports/figures/als_convergence.png`).

Two implementation details that change the results materially:

- **Neighbour truncation in CF is not an optimisation.** Keeping all 1,500 neighbours lets
  a long tail of weak, noisy similarities dominate the weighted sum and measurably hurts
  precision. 120 is the operative choice.
- **The hybrid blends percentile *ranks*, not raw scores**, because ALS dot products,
  cosine similarities and decayed event counts live on incomparable scales — whichever has
  the widest numeric range would otherwise dominate. Ties share a rank
  (`rankdata(method="average")`) rather than being broken by array position.

### 4. Cold start

An item released after the training cut-off cannot be ranked by any collaborative model.
The hybrid detects items with fewer than 3 training interactions and scores them by
content alone, since the CF and MF scores for such an item are not *low-confidence*, they
are *meaningless*.

### 5. Evaluation (`evaluate.py`)

Precision@K, Recall@K, NDCG@K, MAP@K and HitRate@K at K = 5, 10, 20, plus catalogue
coverage, novelty (mean self-information), popularity bias, per-segment accuracy, and
cold-start recall measured only over users who actually have a cold item held out.

**Hybrid weights are tuned on an inner validation split carved out of the training
period** — the test set plays no part in any modelling decision. The search includes
single-model references so the blend has to earn its place, and in fact it did not win on
accuracy: the search correctly selected pure CF, and the hybrid's remaining value is the
cold-item substitution.

### 6. API (`api.py`)

FastAPI, loading the saved models so startup is instant.

| Endpoint | Purpose |
|---|---|
| `GET /recommend/{user_id}?k=10&model=hybrid` | personalised list |
| `GET /users/{user_id}/history` | what the model knows about that user |
| `GET /similar/{item_id}?model=als_mf` | item-to-item neighbours |
| `GET /segments` | recommendation behaviour by user segment |
| `GET /models`, `GET /health` | available models and service state |

Cold users are detected and served time-decayed popularity, with `fallback_used: true` in
the response rather than silently returning something arbitrary.

---

## Files

```
src/generate_data.py   latent-factor interaction log with cold-start items
src/data_prep.py       chronological split, confidence weighting, segments
src/recommenders.py    5 models behind one batched interface
src/evaluate.py        ranking + coverage + cold-start metrics, weight tuning
src/api.py             FastAPI serving layer
app.py                 Streamlit explorer
reports/               evaluation.json, figures/
```

---

## Honest limitations

- **ALS hyperparameters were not searched** (40 factors, reg 0.08, 15 iterations). Given
  it lost to item-item CF, this is the most likely place to find an easy improvement.
- **Offline ranking metrics are a proxy.** They can only reward recommending what the user
  found *anyway*, and penalise a genuinely good suggestion they never saw. Only an A/B
  test measures a recommender properly.
- **No sequence modelling.** Interactions are treated as a set per user. Session-based
  ordering carries real signal that a GRU4Rec- or SASRec-style model would exploit.
- **Popularity bias is measured but not corrected.** Inverse-propensity weighting or an
  explicit diversity re-ranking step would be the next addition.
- **The data is synthetic**, so these numbers describe recovery of a known generative
  process. The pipeline and the evaluation protocol transfer; the metrics do not.
