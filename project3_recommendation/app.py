"""Streamlit explorer for the recommendation models.

    streamlit run app.py

Lets you pick a user, see what the models know about them, and compare every model's
recommendations side by side -- which is far more convincing than a metrics table, because
you can see *why* popularity looks respectable on paper and feels useless in practice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

REPORTS = HERE / "reports"
FIGURES = REPORTS / "figures"
MODELS = HERE / "models"

st.set_page_config(page_title="Recommendation Explorer", page_icon="🛍️", layout="wide")


@st.cache_resource
def load_artefacts():
    from common.io_utils import load_model

    artefact = load_model(MODELS / "recommenders.joblib")
    return artefact["models"], artefact["dataset"]


@st.cache_data
def load_evaluation() -> dict:
    return json.loads((REPORTS / "evaluation.json").read_text())


if not (MODELS / "recommenders.joblib").exists():
    st.error("Fitted models not found.")
    st.markdown(
        "Run the pipeline first:\n\n"
        "```bash\n"
        "python -m src.generate_data\n"
        "python -m src.evaluate\n"
        "```"
    )
    st.stop()

models, dataset = load_artefacts()
evaluation = load_evaluation()
items_indexed = dataset.items.set_index("item_id", drop=False)

st.title("Personalised Product Recommendations")
st.caption(
    f"{dataset.n_users:,} users · {dataset.n_items:,} items · "
    f"training cut-off **{dataset.cutoff:%Y-%m-%d}** · "
    f"{len(dataset.eval_users):,} evaluable users"
)

explorer, metrics, cold_start, similarity = st.tabs(
    ["User explorer", "Model comparison", "Cold start", "Item similarity"]
)

# --------------------------------------------------------------------------------------
with explorer:
    from src.data_prep import user_activity_segment

    controls = st.columns([2, 1, 1])
    user_id = controls[0].selectbox(
        "User", dataset.eval_users[:3000],
        help="Users with training history and held-out test interactions",
    )
    k = controls[1].slider("Recommendations", 5, 20, 10)
    show_models = controls[2].multiselect(
        "Models", list(models), default=["hybrid", "item_item_cf", "als_mf", "popularity"]
    )

    seen = dataset.seen_by_user.get(user_id, set())
    relevant = dataset.relevant_by_user.get(user_id, set())

    summary = st.columns(4)
    summary[0].metric("Items in history", len(seen))
    summary[1].metric("Segment", user_activity_segment(dataset, user_id))
    summary[2].metric("Held-out items", len(relevant))
    history_rows = items_indexed.loc[list(seen)] if seen else pd.DataFrame()
    summary[3].metric(
        "Favourite category",
        history_rows["category"].mode().iloc[0] if len(history_rows) else "n/a",
    )

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Training history")
        if len(history_rows):
            st.dataframe(
                history_rows[["title", "category", "price"]].head(25),
                use_container_width=True, hide_index=True,
            )
        st.subheader("Held out (the answer)")
        if relevant:
            st.dataframe(
                items_indexed.loc[list(relevant)][["title", "category", "price"]],
                use_container_width=True, hide_index=True,
            )

    with right:
        st.subheader("Recommendations by model")
        st.caption("✅ marks an item the user actually interacted with in the test period.")
        for name in show_models:
            recommended = models[name].recommend(user_id, k=k)
            frame = items_indexed.loc[recommended][
                ["title", "category", "price", "avg_rating", "is_cold_start"]
            ].copy()
            frame.insert(0, "hit", ["✅" if item in relevant else "" for item in recommended])
            frame.insert(1, "rank", range(1, len(frame) + 1))

            hits = sum(1 for item in recommended if item in relevant)
            st.markdown(
                f"**{name}** — {hits}/{min(k, len(recommended))} hits "
                f"(precision@{k} = {hits / max(k, 1):.2f})"
            )
            st.dataframe(frame, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
with metrics:
    accuracy = pd.DataFrame(evaluation["accuracy"]).T
    catalogue = pd.DataFrame(evaluation["catalogue"]).T
    combined = accuracy.join(catalogue)

    st.subheader("Ranking quality and catalogue behaviour")
    st.dataframe(
        combined[
            ["precision@10", "recall@10", "ndcg@10", "map@10", "hitrate@10",
             "coverage", "novelty", "mean_popularity_rank"]
        ].style.format("{:.4f}").background_gradient(subset=["ndcg@10"], cmap="Greens"),
        use_container_width=True,
    )
    st.caption(
        "Accuracy and coverage pull in opposite directions. `popularity` covers 1.5% of "
        "the catalogue; `content_based` covers 96% but ranks poorly. The useful models "
        "sit in between, and `hybrid` buys cold-start reach for a small accuracy cost."
    )

    st.subheader("Hybrid weight search")
    trials = evaluation.get("hybrid_weight_trials", [])
    if trials:
        rows = []
        for trial in trials:
            row = dict(trial["weights"])
            row["validation_ndcg@10"] = trial["ndcg@10"]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Searched on a validation split carved out of the training period — the test "
            "set plays no part in choosing these weights."
        )

    for name, caption in [
        ("model_comparison.png", "Models at K = 10"),
        ("ndcg_vs_k.png", "NDCG@K as the list grows"),
        ("coverage_vs_accuracy.png", "The accuracy / coverage trade-off"),
        ("segment_performance.png", "Performance by user history length"),
        ("als_convergence.png", "ALS convergence"),
    ]:
        if (FIGURES / name).exists():
            st.image(str(FIGURES / name), caption=caption, use_container_width=True)

# --------------------------------------------------------------------------------------
with cold_start:
    st.subheader("Items with no training interactions")
    st.caption(
        "These products were released inside the final fifth of the window, so after a "
        "chronological split they have zero interaction history. No collaborative model "
        "can rank them at all — this is what the content-based fallback is for."
    )

    cold = pd.DataFrame(evaluation["cold_start"]).T
    st.dataframe(cold, use_container_width=True)

    catalogue = pd.DataFrame(evaluation["catalogue"]).T
    st.bar_chart(catalogue["cold_items_shown"])
    st.caption("Distinct cold-start items appearing in any user's top-10.")

    cold_items = sorted(dataset.cold_start_items)
    if cold_items:
        st.subheader("The cold-start catalogue")
        st.dataframe(
            items_indexed.loc[cold_items][
                ["title", "category", "subcategory", "brand", "price", "avg_rating", "tags"]
            ],
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------------------------------------
with similarity:
    st.subheader("Item-to-item similarity")
    columns = st.columns([2, 1, 1])
    item_id = columns[0].selectbox(
        "Item", items_indexed["item_id"].tolist(),
        format_func=lambda i: f"{i} — {items_indexed.loc[i, 'title']}",
    )
    neighbours = columns[1].slider("Neighbours", 5, 20, 10)
    model_name = columns[2].selectbox("Model", ["als_mf", "content_based"])

    anchor = items_indexed.loc[item_id]
    st.markdown(
        f"**{anchor['title']}** · {anchor['category']} / {anchor['subcategory']} · "
        f"{anchor['brand']} · {anchor['price']:.2f} · ★{anchor['avg_rating']:.1f}"
    )
    st.caption(f"tags: {anchor['tags']}")

    results = models[model_name].similar_items(int(item_id), k=neighbours)
    frame = items_indexed.loc[[i for i, _ in results]][
        ["title", "category", "subcategory", "brand", "price", "tags"]
    ].copy()
    frame.insert(0, "similarity", [round(score, 4) for _, score in results])
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption(
        "`als_mf` finds items bought by similar people — it can link across categories. "
        "`content_based` finds items that *describe* similarly, which is more literal but "
        "works for brand-new products."
    )
