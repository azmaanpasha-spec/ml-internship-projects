"""Recommendation explorer.

    streamlit run app.py

Metrics tables are a poor way to argue that a recommender works. This app lets you pick
a real user, look at what the models know about them, and compare five models' output
side by side with the items they actually went on to buy marked — which makes the
difference between "popularity" and "personalisation" obvious in a way NDCG cannot.

  Overview     how the five models compare, and the accuracy/coverage trade-off
  User         one user, five models, with the held-out answers revealed
  Models       ranking quality by K, by segment, and the hybrid weight search
  Cold start   the items no collaborative model can touch
  Similarity   item-to-item neighbours, latent vs. metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import ui  # noqa: E402

REPORTS = HERE / "reports"
MODELS = HERE / "models"

MODEL_NOTES = {
    "popularity": "Global interaction counts — the baseline every model must beat",
    "popularity_decay_21d": "Popularity with a 21-day half-life",
    "item_item_cf": "Item-item cosine collaborative filtering — best accuracy",
    "als_mf": "Implicit-feedback ALS, written from scratch — best catalogue coverage",
    "content_based": "TF-IDF over item metadata — the only model that can serve new items",
    "hybrid": "Collaborative filtering, with content substituted for cold items",
}

ui.setup_page(
    title="Recommendation Explorer",
    icon="🛍️",
    heading="Personalised Product Recommendations",
    subtitle="Five models on a chronological split — popularity, CF, ALS, content, hybrid",
    tagline="Pick a user in the User tab and the held-out purchases are revealed next to "
            "each model's guesses. That comparison is the argument; the metrics just "
            "confirm it.",
)

if not (MODELS / "recommenders.joblib").exists():
    ui.missing_artefacts(["python -m src.generate_data", "python -m src.evaluate"])


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading fitted models…")
def load_artefacts():
    from common.io_utils import load_model

    artefact = load_model(MODELS / "recommenders.joblib")
    return artefact["models"], artefact["dataset"]


@st.cache_data(show_spinner=False)
def load_evaluation() -> dict:
    return json.loads((REPORTS / "evaluation.json").read_text())


models, dataset = load_artefacts()
evaluation = load_evaluation()
items_indexed = dataset.items.set_index("item_id", drop=False)
accuracy = pd.DataFrame(evaluation["accuracy"]).T
catalogue = pd.DataFrame(evaluation["catalogue"]).T
best_model = accuracy["ndcg@10"].idxmax()

overview_tab, user_tab, models_tab, cold_tab, similar_tab = st.tabs(
    ["📊 Overview", "👤 User", "🔬 Models", "❄️ Cold start", "🔗 Similarity"]
)


# ======================================================================================
# OVERVIEW
# ======================================================================================
with overview_tab:
    info = evaluation["dataset"]
    baseline = accuracy.loc["popularity", "ndcg@10"]
    uplift = (accuracy.loc[best_model, "ndcg@10"] / baseline - 1) * 100

    ui.kpi_row([
        {"label": "Users", "value": f"{info['n_users']:,}", "tone": "primary"},
        {"label": "Items", "value": f"{info['n_items']:,}", "tone": "primary"},
        {"label": "Interactions", "value": f"{info['train_interactions']:,}",
         "delta": f"{info['sparsity']:.2%} sparse", "tone": "teal"},
        {"label": "Best NDCG@10", "value": f"{accuracy.loc[best_model, 'ndcg@10']:.4f}",
         "delta": f"{best_model}", "tone": "success"},
        {"label": "Uplift vs. popularity", "value": f"+{uplift:.0f}%", "tone": "purple"},
        {"label": "Cold-start items", "value": info["cold_start_items_in_test"],
         "delta": "zero training interactions", "tone": "warning"},
    ])
    st.caption(f"Chronological split at {info['cutoff'][:10]} · "
               f"{info['eval_users']:,} evaluable users · "
               f"{info['cold_start_users']} users with no history")

    ui.section("Five models, one hold-out period",
               "Ranking quality at K = 10. The popularity baseline is not a straw man — "
               "the top 15 items take a third of all traffic.")

    metrics = ["precision@10", "recall@10", "ndcg@10", "map@10", "hitrate@10"]
    figure = go.Figure()
    for index, metric in enumerate(metrics):
        figure.add_trace(go.Bar(
            x=accuracy.index, y=accuracy[metric], name=metric.replace("@10", ""),
            marker_color=ui.SERIES[index % len(ui.SERIES)],
            hovertemplate="%{x}<br>" + metric + " %{y:.4f}<extra></extra>"))
    figure.update_layout(barmode="group", title="Ranking metrics at K = 10")
    ui.chart(figure, height=400)

    ui.section("Accuracy is not the only axis",
               "A recommender that only ever shows the same 50 best-sellers scores "
               "respectably and adds nothing. Bubble size is novelty — how obscure the "
               "recommended items are.")

    scatter = go.Figure()
    for index, model in enumerate(accuracy.index):
        scatter.add_trace(go.Scatter(
            x=[catalogue.loc[model, "coverage"]], y=[accuracy.loc[model, "ndcg@10"]],
            mode="markers+text", name=model, text=[model], textposition="top center",
            textfont=dict(size=10, color=ui.MUTED),
            marker=dict(size=catalogue.loc[model, "novelty"] * 4.5,
                        color=ui.SERIES[index % len(ui.SERIES)], opacity=0.82,
                        line=dict(width=1, color="rgba(255,255,255,0.25)")),
            hovertemplate=(f"<b>{model}</b><br>coverage %{{x:.1%}}<br>NDCG@10 %{{y:.4f}}"
                           f"<br>novelty {catalogue.loc[model, 'novelty']:.2f}<extra></extra>")))
    scatter.update_layout(title="Accuracy vs. catalogue coverage",
                          xaxis=dict(title="Share of catalogue ever recommended",
                                     tickformat=".0%"),
                          yaxis_title="NDCG@10", showlegend=False)
    ui.chart(scatter, height=430)

    ui.insight(
        f"<b>popularity</b> covers {catalogue.loc['popularity', 'coverage']:.1%} of the "
        f"catalogue. <b>content_based</b> covers "
        f"{catalogue.loc['content_based', 'coverage']:.1%} but ranks poorly. "
        f"<b>als_mf</b> gives up some accuracy against item-item CF in exchange for "
        f"{catalogue.loc['als_mf', 'coverage']:.0%} coverage — on a real catalogue that "
        "trade is a merchandising decision, not a modelling one.")


# ======================================================================================
# USER
# ======================================================================================
with user_tab:
    from src.data_prep import user_activity_segment

    ui.section("One user, five models, answers revealed",
               "✅ marks an item the user genuinely interacted with after the training "
               "cut-off. This is the comparison that makes the metrics believable.")

    controls = st.columns([2, 1, 1.6])
    user_id = controls[0].selectbox(
        "User", dataset.eval_users[:3000],
        format_func=lambda u: (f"user {u} · {len(dataset.seen_by_user.get(u, set()))} items "
                               f"· {len(dataset.relevant_by_user.get(u, set()))} held out"))
    top_k = controls[1].slider("Top-K", 5, 20, 10)
    shown = controls[2].multiselect(
        "Models to compare", list(models),
        default=[m for m in ("hybrid", "item_item_cf", "als_mf", "popularity")
                 if m in models])

    seen = dataset.seen_by_user.get(user_id, set())
    relevant = dataset.relevant_by_user.get(user_id, set())
    history_rows = items_indexed.loc[list(seen)] if seen else pd.DataFrame()

    ui.kpi_row([
        {"label": "History", "value": len(seen), "delta": "items before cut-off",
         "tone": "primary"},
        {"label": "Segment", "value": user_activity_segment(dataset, user_id),
         "tone": "teal"},
        {"label": "Held out", "value": len(relevant), "delta": "the answers",
         "tone": "success"},
        {"label": "Top category",
         "value": (history_rows["category"].mode().iloc[0] if len(history_rows) else "—"),
         "tone": "purple"},
        {"label": "Mean price seen",
         "value": (f"{history_rows['price'].mean():.0f}" if len(history_rows) else "—"),
         "tone": "warning"},
    ])

    if len(history_rows):
        taste = history_rows["category"].value_counts()
        donut = go.Figure(go.Pie(
            labels=taste.index, values=taste.to_numpy(), hole=0.62,
            marker=dict(colors=ui.SERIES[: len(taste)]),
            hovertemplate="%{label}<br>%{value} items (%{percent})<extra></extra>"))
        donut.update_layout(title="What this user has engaged with",
                            annotations=[dict(text=f"{len(seen)}<br>items", showarrow=False,
                                              font=dict(size=16, color=ui.MUTED))])
        donut_col, held_col = st.columns([1, 1.3])
        with donut_col:
            ui.chart(donut, height=320)
        with held_col:
            ui.section("The held-out answers")
            if relevant:
                st.dataframe(
                    items_indexed.loc[list(relevant)][
                        ["title", "category", "price", "avg_rating"]],
                    width="stretch", hide_index=True, height=240)
            else:
                st.info("No held-out items for this user.")

    ui.section(f"Recommendations — top {top_k}")
    if not shown:
        st.info("Pick at least one model.")
    else:
        columns = st.columns(len(shown))
        for column, name in zip(columns, shown):
            recommended = models[name].recommend(user_id, k=top_k)
            hits = sum(1 for item in recommended if item in relevant)
            precision = hits / max(top_k, 1)
            tone = "success" if hits else "muted"

            with column:
                st.markdown(
                    f"**{name}**<br>{ui.badge(f'{hits}/{top_k} hits', tone)} "
                    f"{ui.badge(f'P@{top_k} {precision:.2f}', 'primary')}",
                    unsafe_allow_html=True)
                table = items_indexed.loc[recommended][
                    ["title", "category", "price", "is_cold_start"]].copy()
                table.insert(0, "✓", ["✅" if i in relevant else "" for i in recommended])
                table.insert(1, "#", range(1, len(table) + 1))
                st.dataframe(table, width="stretch", hide_index=True, height=min(
                    420, 38 * len(table) + 40),
                    column_config={
                        "price": st.column_config.NumberColumn("Price", format="%.0f"),
                        "is_cold_start": st.column_config.CheckboxColumn("New"),
                    })

        scores = {name: sum(1 for i in models[name].recommend(user_id, k=top_k)
                            if i in relevant) for name in shown}
        winner = max(scores, key=scores.get)
        if scores[winner]:
            ui.insight(
                f"For this user <b>{winner}</b> found <b>{scores[winner]}</b> of their "
                f"{len(relevant)} held-out items in {top_k} guesses, out of a catalogue of "
                f"{dataset.n_items:,}.", "good")
        else:
            ui.insight(
                "No model found a held-out item for this user in the top "
                f"{top_k}. With {len(relevant)} relevant items among {dataset.n_items:,}, "
                "a random guess would hit about "
                f"{len(relevant) * top_k / dataset.n_items:.3f} times — misses like this "
                "are the normal case, which is why metrics are averaged over thousands "
                "of users.", "warn")


# ======================================================================================
# MODELS
# ======================================================================================
with models_tab:
    ui.section("How quality changes with list length",
               "Longer lists catch more but rank worse. Toggle metrics to compare.")

    metric_base = st.radio("Metric", ["ndcg", "precision", "recall", "map", "hitrate"],
                           horizontal=True)
    k_values = evaluation["k_values"]

    figure = go.Figure()
    for index, model in enumerate(accuracy.index):
        values = [accuracy.loc[model, f"{metric_base}@{k}"] for k in k_values]
        figure.add_trace(go.Scatter(
            x=k_values, y=values, mode="lines+markers", name=model,
            line=dict(color=ui.SERIES[index % len(ui.SERIES)], width=2.5),
            marker=dict(size=9),
            hovertemplate=f"{model}<br>K=%{{x}}<br>%{{y:.4f}}<extra></extra>"))
    figure.update_layout(title=f"{metric_base.upper()}@K", xaxis_title="K",
                         yaxis_title=f"{metric_base.upper()}@K",
                         xaxis=dict(tickvals=k_values))
    ui.chart(figure, height=400)

    ui.section("Model profile",
               "Each axis is scaled to the best model on that metric, so the shape shows "
               "what each model is *for* rather than which is biggest.")

    radar_metrics = {
        "ndcg@10": "Ranking", "recall@10": "Recall", "hitrate@10": "Hit rate",
        "coverage": "Coverage", "novelty": "Novelty",
    }
    combined = accuracy.join(catalogue)
    normalised = combined[list(radar_metrics)].div(combined[list(radar_metrics)].max())

    radar = go.Figure()
    for index, model in enumerate(normalised.index):
        values = normalised.loc[model].tolist()
        radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=list(radar_metrics.values()) + [list(radar_metrics.values())[0]],
            name=model, fill="toself", opacity=0.42,
            line=dict(color=ui.SERIES[index % len(ui.SERIES)], width=2),
            hovertemplate=f"{model}<br>%{{theta}} %{{r:.2f}} of best<extra></extra>"))
    radar.update_layout(title="Relative strengths",
                        polar=dict(bgcolor="rgba(0,0,0,0)",
                                   radialaxis=dict(visible=True, range=[0, 1],
                                                   gridcolor="rgba(138,148,166,0.2)"),
                                   angularaxis=dict(gridcolor="rgba(138,148,166,0.2)")))
    ui.chart(radar, height=470)

    ui.section("By how much history the user has",
               "One average hides everything. Light users are most of the base.")
    segments = evaluation["segments"]
    segment_names = sorted(next(iter(segments.values())))
    seg_fig = go.Figure()
    for index, model in enumerate(segments):
        seg_fig.add_trace(go.Bar(
            x=segment_names, y=[segments[model][s]["ndcg@10"] for s in segment_names],
            name=model, marker_color=ui.SERIES[index % len(ui.SERIES)],
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    seg_fig.update_layout(barmode="group", title="NDCG@10 by user history length")
    ui.chart(seg_fig, height=400)

    ui.insight(
        "Light users scoring highest is a measurement artefact worth naming: they have "
        "fewer held-out items, so the ideal DCG they are normalised against is smaller "
        "and easier to approach. The ranking is meaningful <i>within</i> a segment, not "
        "across them — which is exactly why segments are reported at all.", "warn")

    trials = evaluation.get("hybrid_weight_trials", [])
    if trials:
        ui.section("Hybrid weight search",
                   "Searched on a validation split carved from inside the training "
                   "period — the test set plays no part in choosing these weights.")
        rows = []
        for trial in trials:
            row = dict(trial["weights"])
            row["validation ndcg@10"] = trial["ndcg@10"]
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("validation ndcg@10", ascending=False)
        st.dataframe(table, width="stretch", hide_index=True,
                     column_config={"validation ndcg@10": st.column_config.ProgressColumn(
                         "Validation NDCG@10", min_value=0.0,
                         max_value=float(table["validation ndcg@10"].max() * 1.1),
                         format="%.4f")})
        ui.insight(
            "The search picked pure collaborative filtering — the blend did not win on "
            "accuracy, and saying so is more useful than quietly shipping a hybrid that "
            "looks sophisticated. Its value shows up in the next tab instead.")


# ======================================================================================
# COLD START
# ======================================================================================
with cold_tab:
    cold = pd.DataFrame(evaluation["cold_start"]).T
    cold_items = sorted(dataset.cold_start_items)

    ui.kpi_row([
        {"label": "Cold-start items", "value": len(cold_items),
         "delta": "released after the cut-off", "tone": "warning"},
        {"label": "Opportunities",
         "value": int(cold.iloc[0]["cold_item_opportunities"]), "tone": "primary"},
        {"label": "Best cold recall@20",
         "value": f"{cold['cold_item_recall@20'].max():.4f}",
         "delta": cold["cold_item_recall@20"].idxmax(), "tone": "success"},
        {"label": "Collaborative models", "value": "0.0000",
         "delta": "structurally, not poorly", "tone": "danger"},
    ])

    ui.section("The problem no amount of collaborative filtering solves",
               "An item released after the training cut-off has no row in the similarity "
               "matrix and no learned factor. Its score is not low-confidence — it is "
               "undefined.")

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=cold.index, y=cold["cold_item_recall@20"], name="Cold-item recall@20",
        marker_color=[ui.SUCCESS if v > 0 else ui.DANGER
                      for v in cold["cold_item_recall@20"]],
        text=[f"{v:.4f}" for v in cold["cold_item_recall@20"]], textposition="outside",
        hovertemplate="%{x}<br>recall@20 %{y:.4f}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=catalogue.index, y=catalogue["cold_items_shown"], name="Cold items surfaced",
        yaxis="y2", mode="markers", marker=dict(size=14, color=ui.PURPLE, symbol="diamond"),
        hovertemplate="%{x}<br>%{y} distinct cold items in a top-10<extra></extra>"))
    figure.update_layout(
        title="Only the content model — and the hybrid that uses it — can serve new items",
        yaxis=dict(title="Recall@20 on cold items"),
        yaxis2=dict(title="Distinct cold items shown", overlaying="y", side="right",
                    showgrid=False))
    ui.chart(figure, height=420)

    hybrid_ndcg = accuracy.loc["hybrid", "ndcg@10"]
    cf_ndcg = accuracy.loc["item_item_cf", "ndcg@10"]
    hybrid_cold = cold.loc["hybrid", "cold_item_recall@20"]
    ui.insight(
        f"The hybrid gives up <b>{(1 - hybrid_ndcg / cf_ndcg) * 100:.1f}%</b> of NDCG@10 "
        f"({cf_ndcg:.4f} → {hybrid_ndcg:.4f}) and in exchange goes from "
        f"<b>zero</b> cold-start recall to <b>{hybrid_cold:.4f}</b>, surfacing "
        f"{int(catalogue.loc['hybrid', 'cold_items_shown'])} never-before-seen items. "
        "If new products matter to the business, that is a cheap trade.", "good")

    if cold_items:
        ui.section("The cold catalogue",
                   "These products exist, have descriptions, and had never been bought "
                   "when the model was trained.")
        st.dataframe(
            items_indexed.loc[cold_items][
                ["title", "category", "subcategory", "brand", "price", "avg_rating", "tags"]],
            width="stretch", hide_index=True, height=320,
            column_config={"price": st.column_config.NumberColumn("Price", format="%.2f"),
                           "avg_rating": st.column_config.ProgressColumn(
                               "Rating", min_value=0.0, max_value=5.0, format="%.1f")})


# ======================================================================================
# SIMILARITY
# ======================================================================================
with similar_tab:
    ui.section("Item-to-item neighbours",
               "Two different notions of 'similar': bought-by-similar-people (latent) "
               "versus describes-similarly (metadata).")

    controls = st.columns([2.4, 1, 1])
    item_id = controls[0].selectbox(
        "Item", items_indexed["item_id"].tolist(),
        format_func=lambda i: f"{i} — {items_indexed.loc[i, 'title']}")
    neighbours = controls[1].slider("Neighbours", 5, 20, 10)
    model_name = controls[2].selectbox("Model", ["als_mf", "content_based"])

    anchor = items_indexed.loc[item_id]
    ui.entity_card(
        title=str(anchor["title"]),
        meta=(f"{anchor['category']} / {anchor['subcategory']} · {anchor['brand']} · "
              f"{anchor['price']:.2f} · ★{anchor['avg_rating']:.1f}"),
        bullets=[f"tags: {anchor['tags']}"],
        tone="primary")

    results = models[model_name].similar_items(int(item_id), k=neighbours)
    neighbour_ids = [i for i, _ in results]
    similarities = [s for _, s in results]

    frame = items_indexed.loc[neighbour_ids][
        ["title", "category", "subcategory", "brand", "price", "tags"]].copy()
    frame.insert(0, "similarity", [round(s, 4) for s in similarities])

    chart_col, table_col = st.columns([1, 1.5])
    with chart_col:
        same_category = [items_indexed.loc[i, "category"] == anchor["category"]
                         for i in neighbour_ids]
        bar = go.Figure(go.Bar(
            x=similarities[::-1],
            y=[items_indexed.loc[i, "title"][:30] for i in neighbour_ids][::-1],
            orientation="h",
            marker_color=[ui.SUCCESS if s else ui.PURPLE for s in same_category][::-1],
            hovertemplate="%{y}<br>similarity %{x:.4f}<extra></extra>"))
        bar.update_layout(title="Green = same category", xaxis_title="Similarity")
        ui.chart(bar, height=max(300, 28 * len(neighbour_ids) + 90))
    with table_col:
        st.dataframe(frame, width="stretch", hide_index=True, height=max(
            300, 28 * len(neighbour_ids) + 90),
            column_config={"price": st.column_config.NumberColumn("Price", format="%.2f")})

    cross = sum(1 for i in neighbour_ids
                if items_indexed.loc[i, "category"] != anchor["category"])
    if model_name == "als_mf":
        ui.insight(
            f"<b>{cross} of {len(neighbour_ids)}</b> neighbours come from a different "
            "category. ALS has no idea what a category is — it only knows who bought "
            "what together, so cross-category links are genuine co-purchase signal.")
    else:
        ui.insight(
            f"<b>{len(neighbour_ids) - cross} of {len(neighbour_ids)}</b> neighbours share "
            "the anchor's category, which is expected: the content model matches on "
            "description. More literal than ALS, but it works for an item nobody has "
            "bought yet.")
