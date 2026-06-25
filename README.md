# Multi-Modal Restaurant Recommendation System

A restaurant recommender that fuses three modalities — **review text** (DistilBERT
embeddings), **food imagery** (ResNet18 features), and **structured attributes**
(cuisine, price band, location) — into a single hybrid ranking model. It combines
**collaborative** and **content-based** filtering via **late fusion**, and handles
**cold-start** for both new users and new restaurants. The design mirrors what a
discovery feed like Zomato's or Yelp's does under the hood.

Everything runs locally on CPU/MPS in a few minutes: the heavy encoders (DistilBERT,
ResNet18) are used **frozen** as feature extractors, and only the lightweight fusion +
ranking layers are trained.

---

## Architecture

```
                 ┌──────────────┐   ┌──────────────┐   ┌────────────────┐
 reviews ───────▶│  DistilBERT  │   │   ResNet18   │   │   Structured   │
 (text)          │  (frozen)    │   │   (frozen)   │   │   encoder      │
                 └──────┬───────┘   └──────┬───────┘   └───────┬────────┘
                    768-d                512-d              cuisine/price/
                        │                    │              location (20-d)
                        └──────────┬─────────┴──────────┬──────────┘
                                   ▼                    ▼
                        LATE FUSION  (z-score → PCA → L2-norm per modality → concat)
                                   │
                                   ▼  item content vector
        ┌──────────────────────────────────────────────────────────┐
        │                  UNIFIED MLP RANKER                        │
        │   [ user_CF | item_CF | item_content | user_profile |      │
        │     cross-features: cuisine match / price fit / proximity ]│
        └──────────────────────────────────────────────────────────┘
              ▲                    ▲                      ▲
        matrix-factorization   content profiles     cold-start regressors
        (collaborative)        (rating-weighted)    (content → CF vector)
```

* **Late fusion (feature level):** each modality is standardized, PCA-reduced to a
  common size, L2-normalized, then concatenated — so cosine similarity in this space
  is a clean weighted average of per-modality similarities.
* **Late fusion (model level):** the ranker MLP fuses the collaborative signal
  (matrix-factorization user/item vectors) with the content signal and explicit
  structured cross-features.
* **Cold-start:** small regressors predict a CF vector from content, so a brand-new
  restaurant (no interactions) still slots into the hybrid ranker; new users are
  bootstrapped from onboarding preferences.

## Project layout

```
src/restaurant_rec/
  config.py              # all paths, dims, seeds, hyperparameters
  data/
    schema.py            # table/column definitions
    generate.py          # synthetic multi-modal world (the coherent dataset)
    images.py            # procedural per-cuisine food images
  encoders/              # text (DistilBERT) · image (ResNet18) · structured  [frozen]
  fusion/late_fusion.py  # standardize → PCA → L2 → concat
  models/
    collaborative.py     # matrix factorization (PyTorch)
    content.py           # user taste profiles + pairwise cross features
    ranker.py            # unified hybrid MLP ranker
    coldstart.py         # content → CF regressors
  pipeline.py            # shared assembly of inference state
  train.py · evaluate.py · recommend.py · cli.py
notebooks/walkthrough.ipynb
tests/test_smoke.py
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10. First run downloads DistilBERT (~270 MB) and ResNet18 (~45 MB)
once; all embeddings are then cached under `artifacts/`.

## Usage

Run the whole pipeline end to end:

```bash
python -m restaurant_rec.cli all      # generate → embeddings → train → evaluate
```

…or step by step:

```bash
python -m restaurant_rec.cli generate-data       # synthetic CSVs + food images → data/
python -m restaurant_rec.cli build-embeddings     # cache DistilBERT + ResNet features
python -m restaurant_rec.cli train                # CF, ranker, cold-start models
python -m restaurant_rec.cli evaluate             # metrics + ablations + cold-start
```

Get recommendations:

```bash
# existing user
python -m restaurant_rec.cli recommend --user-id 5 --k 8

# brand-new (cold) user, described only by preferences
python -m restaurant_rec.cli recommend --cold --cuisines Italian,Japanese --price 2 --city Bayport
```

The annotated walkthrough lives in [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)
(`pip install jupyter matplotlib` to run it).

## Results (default config: 300 restaurants, 1.5k users, 20k interactions)

**Model comparison — fusion wins** (warm users, held-out interactions):

| Model            | NDCG@10 | MAP@10 | P@10  |
|------------------|--------:|-------:|------:|
| **Hybrid (fusion)** | **0.142** | **0.090** | **0.046** |
| Content only     | 0.099   | 0.057  | 0.037 |
| CF only          | 0.063   | 0.037  | 0.022 |

**Modality ablations — each modality contributes** (NDCG@10, ranker retrained per subset):
`all 0.142 > text-only 0.140 > image-only 0.137 > structured-only 0.134`.

**Cold-start:** new users are served well from onboarding preferences (NDCG@10 ≈ 0.26);
new restaurants land mid-pack (median rank ~150/300) rather than being dropped — the
classic cold-item exposure problem, handled gracefully via the content→CF regressor.

(Exact numbers vary slightly with the random seed; reproduce with `cli evaluate`.)

## How the synthetic data carries real signal

The hard part of "multi-modal" is getting all three modalities for the *same* item.
Instead of stapling together unrelated public datasets, we generate a **coherent world**
where hidden factors drive the ratings and each modality surfaces a different slice:

| Hidden factor | Affects rating | Recoverable from |
|---------------|:--------------:|------------------|
| cuisine affinity | ✓ | text, image, structured |
| food quality | ✓ | text sentiment + image brightness |
| **service** | ✓ | **text only** (review wording) |
| **presentation** | ✓ | **image only** (plating/vibrancy) |
| **price / location** | ✓ | **structured only** |
| latent taste residual | ✓ | interactions only → collaborative filtering |

This is exactly why content-based filtering beats any single modality, and the hybrid
(adding the CF-only latent residual) beats content alone.

## Swapping in real data

The synthetic generator is a drop-in stand-in. To use real data, replace the CSVs in
`data/` (matching the schema in `data/schema.py`) and the images at
`data/images/rest_XXXX.png` (paths are stored in `restaurants.csv`), then rerun
`build-embeddings` onward. No model code changes are needed.

## Tests

```bash
pytest -q          # fast smoke tests (no model downloads needed)
```
