"""Shared design system for the three Streamlit applications.

The three apps solve different problems but should read as one product: same
typography, same colour semantics, same chart grammar, same spacing. Everything that
decides how they *look* lives here, so a change to the palette or the card style
propagates to all three instead of being re-typed in each `app.py`.

Two rules the components follow:

* **Colour carries meaning, never decoration.** `DANGER` is only ever used for something
  genuinely bad (a critical machine, a lost customer), `SUCCESS` only for something good.
  A reader who learns the palette on one page can read the other two without relearning.
* **Charts are interactive by default.** Plotly, not static images, because the questions
  people actually ask of these dashboards -- "which machine is that spike?", "what is the
  precision at 0.3 recall?" -- are answered by hovering, not by squinting.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------------------
# Colour system
# --------------------------------------------------------------------------------------

PRIMARY = "#5B8DEF"
SUCCESS = "#2FB67C"
WARNING = "#E8A33D"
DANGER = "#E0556A"
PURPLE = "#A277E0"
TEAL = "#38B2C4"
MUTED = "#8A94A6"

# Ordered series colours for multi-line / multi-bar charts.
SERIES = [PRIMARY, SUCCESS, WARNING, PURPLE, TEAL, DANGER, "#D98EC1", "#7E8CA0"]

# Semantic tones used by badges, KPI accents and risk bands.
TONES: Mapping[str, str] = {
    "primary": PRIMARY,
    "success": SUCCESS,
    "warning": WARNING,
    "danger": DANGER,
    "purple": PURPLE,
    "teal": TEAL,
    "muted": MUTED,
}

# Risk bands are shared vocabulary across projects 1 and 2.
RISK_TONES: Mapping[str, str] = {
    "Critical": DANGER,
    "Alert": WARNING,
    "High": WARNING,
    "Watch": TEAL,
    "Medium": TEAL,
    "Normal": SUCCESS,
    "Low": SUCCESS,
}


def tone_colour(name: str, default: str = PRIMARY) -> str:
    return TONES.get(name, RISK_TONES.get(name, default))


# --------------------------------------------------------------------------------------
# Plotly theme
# --------------------------------------------------------------------------------------

_TEMPLATE_NAME = "mlprojects"

#: Chart toolbar config: keep the useful tools, drop the ones nobody uses.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
}


def _register_template() -> None:
    """Transparent-background template so charts inherit the Streamlit theme.

    Hard-coding a background would make every chart a pale rectangle on a dark page (or
    the reverse). Transparent paper plus a semi-transparent grid works in both themes.
    """
    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, Segoe UI, Roboto, sans-serif",
            size=13,
            color="#C9D1DB",
        ),
        title=dict(font=dict(size=15, color="#E6EAF0"), x=0.01, xanchor="left"),
        xaxis=dict(
            gridcolor="rgba(138,148,166,0.16)",
            zerolinecolor="rgba(138,148,166,0.28)",
            linecolor="rgba(138,148,166,0.28)",
            ticks="outside",
            tickcolor="rgba(138,148,166,0.28)",
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="rgba(138,148,166,0.16)",
            zerolinecolor="rgba(138,148,166,0.28)",
            linecolor="rgba(138,148,166,0.28)",
            ticks="outside",
            tickcolor="rgba(138,148,166,0.28)",
            automargin=True,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(138,148,166,0.20)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            bgcolor="#1C2230",
            bordercolor="rgba(138,148,166,0.35)",
            font=dict(size=12, color="#E6EAF0"),
        ),
        margin=dict(l=10, r=10, t=48, b=10),
    )
    pio.templates[_TEMPLATE_NAME] = template
    pio.templates.default = _TEMPLATE_NAME


def chart(figure: go.Figure, height: int | None = None, key: str | None = None) -> None:
    """Render a Plotly figure with the shared template and toolbar config."""
    figure.update_layout(template=_TEMPLATE_NAME)
    if height is not None:
        figure.update_layout(height=height)
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key=key)


# --------------------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }

/* Tighten Streamlit's very generous default top padding. */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }

/* ---------- hero ---------- */
.hero {
  padding: 1.1rem 1.4rem; margin-bottom: 1.1rem; border-radius: 14px;
  background: linear-gradient(135deg, rgba(91,141,239,0.16), rgba(162,119,224,0.08) 55%, rgba(0,0,0,0));
  border: 1px solid rgba(138,148,166,0.20);
}
.hero h1 { margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; }
.hero .sub { margin-top: .3rem; font-size: .9rem; color: #9AA5B5; }
.hero .tagline {
  margin-top: .55rem; font-size: .84rem; color: #7E8CA0;
  border-left: 3px solid rgba(91,141,239,0.55); padding-left: .6rem;
}

/* ---------- KPI cards ---------- */
.kpi-grid { display: flex; gap: .7rem; flex-wrap: wrap; margin: .2rem 0 1.1rem 0; }
.kpi {
  flex: 1 1 150px; min-width: 150px; padding: .8rem .9rem;
  border-radius: 12px; background: rgba(138,148,166,0.07);
  border: 1px solid rgba(138,148,166,0.16); border-top: 3px solid var(--accent, #5B8DEF);
  transition: transform .12s ease, background .12s ease;
}
.kpi:hover { transform: translateY(-2px); background: rgba(138,148,166,0.11); }
.kpi .label {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
  color: #8A94A6; font-weight: 600;
}
.kpi .value { font-size: 1.55rem; font-weight: 700; margin-top: .15rem; line-height: 1.15; }
.kpi .delta { font-size: .76rem; margin-top: .2rem; color: #8A94A6; }

/* ---------- section headers ---------- */
.section { margin: 1.4rem 0 .6rem 0; }
.section h3 {
  margin: 0; font-size: 1.04rem; font-weight: 650; letter-spacing: -0.01em;
  display: flex; align-items: center; gap: .5rem;
}
.section h3::before {
  content: ''; width: 3px; height: 1.05em; border-radius: 2px; background: #5B8DEF;
}
.section .caption { margin-top: .25rem; font-size: .83rem; color: #8A94A6; max-width: 90ch; }

/* ---------- badges ---------- */
.badge {
  display: inline-block; padding: .14rem .55rem; border-radius: 999px;
  font-size: .74rem; font-weight: 600; letter-spacing: .01em;
  border: 1px solid currentColor; white-space: nowrap;
}

/* ---------- insight callout ---------- */
.insight {
  padding: .75rem .95rem; border-radius: 10px; margin: .5rem 0 .9rem 0;
  background: rgba(91,141,239,0.09); border-left: 3px solid #5B8DEF;
  font-size: .87rem; line-height: 1.55; color: #C2CBD8;
}
.insight.warn { background: rgba(232,163,61,0.10); border-left-color: #E8A33D; }
.insight.good { background: rgba(47,182,124,0.10); border-left-color: #2FB67C; }
.insight b, .insight strong { color: #E6EAF0; }

/* ---------- entity card (machine / customer / item) ---------- */
.entity {
  padding: .8rem 1rem; border-radius: 12px; margin-bottom: .6rem;
  background: rgba(138,148,166,0.06); border: 1px solid rgba(138,148,166,0.18);
  border-left: 4px solid var(--accent, #5B8DEF);
}
.entity .title { font-weight: 650; font-size: .98rem; }
.entity .meta { font-size: .8rem; color: #8A94A6; margin-top: .1rem; }
.entity ul { margin: .45rem 0 .1rem 1.05rem; padding: 0; }
.entity li { font-size: .84rem; color: #C2CBD8; margin-bottom: .14rem; }
.entity .action {
  margin-top: .5rem; font-size: .82rem; font-weight: 600; color: #E6EAF0;
  background: rgba(138,148,166,0.12); padding: .32rem .6rem; border-radius: 7px;
  display: inline-block;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid rgba(138,148,166,0.18); }
.stTabs [data-baseweb="tab"] {
  height: 2.5rem; padding: 0 .95rem; border-radius: 9px 9px 0 0; font-weight: 550; font-size: .9rem;
}
.stTabs [aria-selected="true"] { background: rgba(91,141,239,0.14); }

/* ---------- dataframes ---------- */
[data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid rgba(138,148,166,0.16); }

/* ---------- misc ---------- */
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
hr { margin: 1.1rem 0; border-color: rgba(138,148,166,0.16); }
footer, #MainMenu { visibility: hidden; }
</style>
"""


def setup_page(
    title: str,
    icon: str,
    heading: str,
    subtitle: str = "",
    tagline: str = "",
) -> None:
    """Configure the page, inject the stylesheet, and render the hero banner."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide",
                       initial_sidebar_state="collapsed")
    _register_template()
    st.markdown(_CSS, unsafe_allow_html=True)

    parts = [f"<div class='hero'><h1>{icon}&nbsp; {heading}</h1>"]
    if subtitle:
        parts.append(f"<div class='sub'>{subtitle}</div>")
    if tagline:
        parts.append(f"<div class='tagline'>{tagline}</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------------------


def kpi_row(items: Sequence[Mapping[str, object]]) -> None:
    """A row of accent-topped metric cards.

    Each item: {label, value, delta (optional), tone (optional)}.
    """
    cards = []
    for item in items:
        accent = tone_colour(str(item.get("tone", "primary")))
        delta = item.get("delta")
        delta_html = f"<div class='delta'>{delta}</div>" if delta else ""
        cards.append(
            f"<div class='kpi' style='--accent:{accent}'>"
            f"<div class='label'>{item['label']}</div>"
            f"<div class='value' style='color:{accent}'>{item['value']}</div>"
            f"{delta_html}</div>"
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    caption_html = f"<div class='caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"<div class='section'><h3>{title}</h3>{caption_html}</div>", unsafe_allow_html=True
    )


def badge(text: str, tone: str = "primary") -> str:
    """Returns HTML -- compose several into one `st.markdown` call."""
    return f"<span class='badge' style='color:{tone_colour(tone)}'>{text}</span>"


def insight(text: str, kind: str = "info") -> None:
    """A callout for the interpretation a reader would otherwise have to infer."""
    css_class = {"info": "insight", "warn": "insight warn", "good": "insight good"}[kind]
    st.markdown(f"<div class='{css_class}'>{text}</div>", unsafe_allow_html=True)


def entity_card(
    title: str,
    meta: str,
    bullets: Iterable[str] = (),
    action: str = "",
    tone: str = "primary",
) -> None:
    """A machine, customer or item with its reasons and recommended action."""
    bullet_html = "".join(f"<li>{b}</li>" for b in bullets if b)
    list_html = f"<ul>{bullet_html}</ul>" if bullet_html else ""
    action_html = f"<div class='action'>{action}</div>" if action else ""
    st.markdown(
        f"<div class='entity' style='--accent:{tone_colour(tone)}'>"
        f"<div class='title'>{title}</div><div class='meta'>{meta}</div>"
        f"{list_html}{action_html}</div>",
        unsafe_allow_html=True,
    )


def missing_artefacts(commands: Sequence[str], note: str = "") -> None:
    """Consistent empty state when a pipeline has not been run yet."""
    st.error("Model outputs not found for this project.")
    if note:
        st.caption(note)
    st.code("\n".join(commands), language="bash")
    st.stop()


# --------------------------------------------------------------------------------------
# Chart builders shared by more than one app
# --------------------------------------------------------------------------------------


def gauge(value: float, title: str, threshold: float | None = None,
          tone: str = "primary", suffix: str = "%") -> go.Figure:
    """A single-value gauge, used for per-entity risk."""
    colour = tone_colour(tone)
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 30, "color": colour}},
            title={"text": title, "font": {"size": 13, "color": MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "rgba(138,148,166,0.4)"},
                "bar": {"color": colour, "thickness": 0.72},
                "bgcolor": "rgba(138,148,166,0.10)",
                "borderwidth": 0,
                "threshold": (
                    {"line": {"color": DANGER, "width": 3}, "thickness": 0.82,
                     "value": threshold}
                    if threshold is not None else None
                ),
            },
        )
    )
    figure.update_layout(height=200, margin=dict(l=18, r=18, t=38, b=8))
    return figure


def horizontal_bars(labels: Sequence[str], values: Sequence[float], title: str,
                    colour: str = PRIMARY, value_fmt: str = "{:.4f}") -> go.Figure:
    """Ranked horizontal bars -- feature importance, driver charts."""
    figure = go.Figure(
        go.Bar(
            x=list(values), y=list(labels), orientation="h",
            marker=dict(color=colour, line=dict(width=0)),
            hovertemplate="%{y}<br>%{x:.5f}<extra></extra>",
            text=[value_fmt.format(v) for v in values],
            textposition="outside", textfont=dict(size=10, color=MUTED),
            cliponaxis=False,
        )
    )
    figure.update_layout(
        title=title,
        height=max(260, 24 * len(labels) + 90),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True),
    )
    return figure
