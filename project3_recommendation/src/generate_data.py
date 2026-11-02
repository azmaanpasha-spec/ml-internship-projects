"""Generate a synthetic e-commerce interaction log for recommender training.

Step 1 of the approach needs user-item interactions from views, clicks, carts and
purchases. The generator builds them from a latent-factor preference model, which gives
the data the structure a recommender is supposed to exploit:

* **Latent taste.** Each user and item has an 8-dimensional factor vector; the dot
  product drives preference. This is recoverable by matrix factorisation and is the
  reason MF should beat popularity.
* **Power-law popularity.** A small number of items take most of the traffic, so a
  popularity baseline is genuinely hard to beat -- as it is in production.
* **Category affinity and price sensitivity.** Signals that live in item *metadata*,
  which is what gives the content-based model something to work with.
* **Cold-start by construction.** Roughly 12% of items are released inside the final
  fifth of the window, so after a time-based split they have no training interactions at
  all. Any model without a content fallback simply cannot recommend them.
* **Funnel structure.** A view is weak evidence, a purchase is strong. The event types
  are drawn conditional on true affinity, so the implicit-feedback weighting in
  `data_prep.py` has a real signal to weight.

Outputs: users.csv, items.csv, interactions.csv

Run:  python -m src.generate_data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SEED = 2024
DEFAULT_USERS = 6000
DEFAULT_ITEMS = 1500
DEFAULT_DAYS = 180
LATENT_DIM = 8
TIME_BLOCKS = 9              # availability is refreshed per block when sampling
COLD_START_ITEM_SHARE = 0.12
START_TIME = pd.Timestamp("2026-01-06 00:00:00")

EVENT_TYPES = ["view", "click", "add_to_cart", "purchase"]
# Implicit-feedback confidence weights: a purchase is worth many views.
EVENT_WEIGHTS = {"view": 1.0, "click": 2.0, "add_to_cart": 4.0, "purchase": 8.0}

CATEGORIES: Dict[str, Dict] = {
    "Electronics": {
        "subcategories": ["Headphones", "Laptops", "Cameras", "Smart Home", "Wearables"],
        "price_range": (45, 1800),
        "tags": ["wireless", "bluetooth", "4k", "noise-cancelling", "usb-c", "fast-charge",
                 "portable", "hd", "gaming", "compact"],
    },
    "Home & Kitchen": {
        "subcategories": ["Cookware", "Appliances", "Storage", "Bedding", "Lighting"],
        "price_range": (15, 650),
        "tags": ["stainless-steel", "non-stick", "space-saving", "dishwasher-safe",
                 "cotton", "energy-efficient", "ceramic", "modular", "bamboo", "dimmable"],
    },
    "Sports & Outdoors": {
        "subcategories": ["Fitness", "Camping", "Cycling", "Running", "Water Sports"],
        "price_range": (12, 900),
        "tags": ["waterproof", "lightweight", "breathable", "adjustable", "insulated",
                 "quick-dry", "shock-absorbing", "high-visibility", "packable", "reinforced"],
    },
    "Books & Media": {
        "subcategories": ["Fiction", "Non-fiction", "Technical", "Children", "Audio"],
        "price_range": (6, 95),
        "tags": ["bestseller", "illustrated", "hardcover", "award-winning", "series",
                 "beginner-friendly", "reference", "unabridged", "translated", "annotated"],
    },
    "Beauty & Personal Care": {
        "subcategories": ["Skincare", "Haircare", "Fragrance", "Grooming", "Cosmetics"],
        "price_range": (8, 260),
        "tags": ["fragrance-free", "vegan", "dermatologist-tested", "spf", "organic",
                 "sensitive-skin", "long-lasting", "refillable", "paraben-free", "travel-size"],
    },
}

BRANDS = [f"Brand{chr(65 + i)}" for i in range(18)]


def _build_items(rng: np.random.Generator, n_items: int, n_days: int) -> pd.DataFrame:
    """Catalogue with latent factors, metadata and staggered release dates."""
    category_names = list(CATEGORIES)
    categories = rng.choice(category_names, size=n_items, p=[0.26, 0.22, 0.20, 0.16, 0.16])

    rows: List[dict] = []
    # Popularity follows a power law: rank 1 gets orders of magnitude more traffic.
    popularity = rng.pareto(1.3, n_items) + 1.0
    popularity = popularity / popularity.max()

    cold_start_count = int(n_items * COLD_START_ITEM_SHARE)
    cold_start_items = set(rng.choice(n_items, size=cold_start_count, replace=False))

    for index in range(n_items):
        category = categories[index]
        spec = CATEGORIES[category]
        low, high = spec["price_range"]

        # Cold-start items appear only in the final fifth of the window.
        if index in cold_start_items:
            release_day = float(rng.uniform(n_days * 0.80, n_days * 0.98))
        else:
            release_day = float(rng.uniform(-240, n_days * 0.55))

        n_tags = int(rng.integers(3, 6))
        tags = rng.choice(spec["tags"], size=n_tags, replace=False)

        rows.append(
            {
                "item_id": index,
                "title": f"{rng.choice(BRANDS)} {rng.choice(spec['subcategories'])} {index:04d}",
                "category": category,
                "subcategory": str(rng.choice(spec["subcategories"])),
                "brand": str(rng.choice(BRANDS)),
                "price": round(float(np.exp(rng.uniform(np.log(low), np.log(high)))), 2),
                "avg_rating": round(float(np.clip(rng.normal(4.1, 0.55), 1.0, 5.0)), 2),
                "tags": " ".join(sorted(tags)),
                "release_date": START_TIME + pd.Timedelta(days=release_day),
                "base_popularity": round(float(popularity[index]), 6),
                "is_cold_start": index in cold_start_items,
            }
        )

    items = pd.DataFrame(rows)

    # Latent factors correlated with category, so metadata carries real signal about
    # taste -- this is what makes the content-based fallback more than a guess.
    category_centres = {
        name: rng.normal(0, 0.8, LATENT_DIM) for name in category_names
    }
    factors = np.vstack(
        [category_centres[category] + rng.normal(0, 0.55, LATENT_DIM) for category in items["category"]]
    )
    for dimension in range(LATENT_DIM):
        items[f"_factor_{dimension}"] = np.round(factors[:, dimension], 4)

    return items


def _build_users(rng: np.random.Generator, n_users: int, n_days: int) -> pd.DataFrame:
    """Users with latent taste, activity level, price sensitivity and signup date."""
    category_names = list(CATEGORIES)
    factors = rng.normal(0, 0.9, (n_users, LATENT_DIM))

    # Activity is heavy-tailed: a few users generate a large share of all events.
    activity = np.clip(rng.lognormal(mean=3.2, sigma=0.75, size=n_users), 3, 300)

    rows = {
        "user_id": np.arange(n_users),
        "signup_date": [
            START_TIME + pd.Timedelta(days=float(d))
            for d in np.where(
                rng.random(n_users) < 0.10,
                rng.uniform(n_days * 0.75, n_days * 0.95, n_users),   # recent sign-ups
                rng.uniform(-400, n_days * 0.4, n_users),
            )
        ],
        "activity_level": np.round(activity, 1),
        "price_sensitivity": np.round(np.clip(rng.normal(0.55, 0.28, n_users), 0.0, 1.4), 3),
        "preferred_category": rng.choice(category_names, size=n_users),
        "device": rng.choice(["mobile", "desktop", "tablet"], size=n_users, p=[0.58, 0.33, 0.09]),
        "country": rng.choice(["IN", "US", "GB", "DE", "AU"], size=n_users,
                              p=[0.38, 0.27, 0.14, 0.12, 0.09]),
    }
    users = pd.DataFrame(rows)
    for dimension in range(LATENT_DIM):
        users[f"_factor_{dimension}"] = np.round(factors[:, dimension], 4)
    return users


def _preference_scores(
    users: pd.DataFrame, items: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latent affinity, category bonus and price penalty for every user-item pair.

    Returned separately so the sampler can combine them per time block without
    recomputing the expensive dot product.
    """
    user_factors = users[[f"_factor_{d}" for d in range(LATENT_DIM)]].to_numpy()
    item_factors = items[[f"_factor_{d}" for d in range(LATENT_DIM)]].to_numpy()

    affinity = user_factors @ item_factors.T / np.sqrt(LATENT_DIM)

    category_bonus = (
        users["preferred_category"].to_numpy()[:, None] == items["category"].to_numpy()[None, :]
    ).astype(float)

    log_price = np.log1p(items["price"].to_numpy())
    price_penalty = users["price_sensitivity"].to_numpy()[:, None] * (log_price[None, :] / log_price.max())

    return affinity, category_bonus, price_penalty


def _sample_interactions(
    rng: np.random.Generator, users: pd.DataFrame, items: pd.DataFrame, n_days: int
) -> pd.DataFrame:
    """Draw each user's interaction stream, respecting item release dates."""
    affinity, category_bonus, price_penalty = _preference_scores(users, items)
    log_popularity = np.log1p(items["base_popularity"].to_numpy() * 100.0)
    release_days = (
        (items["release_date"] - START_TIME).dt.total_seconds().to_numpy() / 86400.0
    )

    # Utility, before availability masking.
    utility = (
        2.30 * affinity
        + 1.15 * log_popularity[None, :]
        + 0.95 * category_bonus
        - 1.10 * price_penalty
        + 0.45 * (items["avg_rating"].to_numpy()[None, :] - 4.0)
    )

    block_edges = np.linspace(0, n_days, TIME_BLOCKS + 1)
    availability = [release_days <= edge for edge in block_edges[1:]]

    signup_days = (users["signup_date"] - START_TIME).dt.total_seconds().to_numpy() / 86400.0

    user_ids, item_ids, timestamps, event_types = [], [], [], []

    for user_index in range(len(users)):
        n_events = int(rng.poisson(users["activity_level"].iloc[user_index] * 0.95)) + 1
        active_from = max(signup_days[user_index], 0.0)
        if active_from >= n_days:
            continue

        # Spread this user's events across the blocks they were active for.
        user_utility = utility[user_index]
        for block_index in range(TIME_BLOCKS):
            block_start, block_end = block_edges[block_index], block_edges[block_index + 1]
            if block_end <= active_from:
                continue

            share = (block_end - max(block_start, active_from)) / max(n_days - active_from, 1e-6)
            block_events = int(rng.poisson(n_events * share))
            if block_events == 0:
                continue

            available = availability[block_index]
            if not available.any():
                continue

            scores = np.where(available, user_utility, -np.inf)
            scores = scores - scores[available].max()
            probabilities = np.exp(scores / 0.85)
            total = probabilities.sum()
            if not np.isfinite(total) or total <= 0:
                continue
            probabilities = probabilities / total

            chosen = rng.choice(len(items), size=block_events, p=probabilities)
            event_days = rng.uniform(max(block_start, active_from), block_end, block_events)

            # Funnel: stronger true affinity converts further down the funnel. The
            # absolute rates are kept at realistic e-commerce levels -- most events are
            # views, a few percent are purchases -- so the implicit-feedback weighting
            # faces the same signal-to-noise problem it faces in production.
            strength = affinity[user_index, chosen]
            p_purchase = 0.015 + 0.085 / (1.0 + np.exp(-(strength * 1.6 - 1.2)))
            p_cart = p_purchase * 2.0
            p_click = 0.20

            draw = rng.random(block_events)
            events = np.where(
                draw < p_purchase, "purchase",
                np.where(
                    draw < p_purchase + p_cart, "add_to_cart",
                    np.where(draw < p_purchase + p_cart + p_click, "click", "view"),
                ),
            )

            user_ids.append(np.full(block_events, users["user_id"].iloc[user_index]))
            item_ids.append(chosen)
            timestamps.append(event_days)
            event_types.append(events)

    interactions = pd.DataFrame(
        {
            "user_id": np.concatenate(user_ids),
            "item_id": np.concatenate(item_ids),
            "event_type": np.concatenate(event_types),
            "timestamp": [
                START_TIME + pd.Timedelta(days=float(d)) for d in np.concatenate(timestamps)
            ],
        }
    )
    interactions["event_weight"] = interactions["event_type"].map(EVENT_WEIGHTS)
    return interactions.sort_values("timestamp").reset_index(drop=True)


def generate(
    n_users: int = DEFAULT_USERS,
    n_items: int = DEFAULT_ITEMS,
    n_days: int = DEFAULT_DAYS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    items = _build_items(rng, n_items, n_days)
    users = _build_users(rng, n_users, n_days)
    interactions = _sample_interactions(rng, users, items, n_days)
    return users, items, interactions


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Generate e-commerce interaction data.")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=here / "data" / "raw")
    args = parser.parse_args()

    users, items, interactions = generate(args.users, args.items, args.days, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    users.to_csv(args.out / "users.csv", index=False)
    items.to_csv(args.out / "items.csv", index=False)
    interactions.to_csv(args.out / "interactions.csv", index=False)

    sparsity = 1.0 - len(interactions.drop_duplicates(["user_id", "item_id"])) / (
        len(users) * len(items)
    )
    print(f"users        : {len(users):,}")
    print(f"items        : {len(items):,} ({int(items['is_cold_start'].sum())} cold-start)")
    print(f"interactions : {len(interactions):,}")
    print(f"  by type    : {interactions['event_type'].value_counts().to_dict()}")
    print(f"sparsity     : {sparsity:.4%} empty")
    print(f"per user     : median {interactions.groupby('user_id').size().median():.0f}, "
          f"max {interactions.groupby('user_id').size().max():,}")
    print(f"top-1% items : {interactions['item_id'].value_counts().head(15).sum() / len(interactions):.1%} "
          f"of all events")
    print(f"date range   : {interactions['timestamp'].min().date()} .. "
          f"{interactions['timestamp'].max().date()}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
