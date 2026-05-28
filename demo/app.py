"""Streamlit demo — next-match forecast: pick two teams, get calibrated 1X2 probabilities.

A thin HTTP client of the inference service: all logic lives in :mod:`demo.service` and
:mod:`demo.summary`; this script only draws widgets and wires them together. The demo forecasts the
teams' *next* (unscheduled) meeting, so it asks for no date — :class:`demo.service.PredictClient`
sends the point-in-time cutoff the frozen service needs. Streamlit reruns the whole script on every
interaction, so the predict call is guarded behind the button press.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from demo.service import PredictClient, Probabilities, ServiceError, available_leagues, teams_for
from demo.summary import verbal_summary

_OUTCOME_LABELS = ("Home win", "Draw", "Away win")


def _render_prediction(probs: Probabilities, home: str, away: str) -> None:
    """Draw the metric trio, a probability bar chart, and the plain-English read."""
    st.subheader("Calibrated probabilities")
    home_col, draw_col, away_col = st.columns(3)
    home_col.metric(f"{home} win", f"{probs.home_win * 100:.0f}%")
    draw_col.metric("Draw", f"{probs.draw * 100:.0f}%")
    away_col.metric(f"{away} win", f"{probs.away_win * 100:.0f}%")

    # Altair (not st.bar_chart) so the categorical x-axis stays in H -> D -> A order: st.bar_chart
    # alpha-sorts its string index and would render Away | Draw | Home instead.
    chart_data = pd.DataFrame(
        {"outcome": list(_OUTCOME_LABELS), "probability": [probs.home_win, probs.draw, probs.away_win]}
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("outcome:N", sort=list(_OUTCOME_LABELS), title=None),
            y=alt.Y("probability:Q", title="probability"),
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.info(verbal_summary(probs))


st.set_page_config(page_title="match-odds — next-match forecast", page_icon="⚽")

# Streamlit ships no first-class toggle for the "Made with Streamlit vX.Y.Z" footer; a one-line CSS
# injection is the supported workaround. Keep this immediately after set_page_config so the style
# applies before any widget renders. Do not strip as "cruft" — without it the footer comes back.
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("⚽ match-odds — next-match forecast")
st.caption(
    "Calibrated home / draw / away probabilities for the next meeting of two teams, served by the "
    "match-odds inference service. Pick a league and the two teams, then hit Predict."
)

league = st.selectbox("League", available_leagues())
team_names = teams_for(league)
home_box, away_box = st.columns(2)
home = home_box.selectbox("Home team", team_names, index=0)
away = away_box.selectbox("Away team", team_names, index=1 if len(team_names) > 1 else 0)
st.caption(
    "🗓️ Forecast for the **next match** between these teams — it uses each side's latest form, so no "
    "fixture date is needed."
)

if st.button("Predict", type="primary"):
    if home == away:
        st.warning("Pick two different teams — a side can't play itself.")
    else:
        try:
            probabilities = PredictClient().predict(home=home, away=away, league=league)
        except ServiceError as exc:
            st.error(str(exc))
        else:
            _render_prediction(probabilities, home, away)
