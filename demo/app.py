"""Streamlit demo — "odds this weekend": pick a fixture, get calibrated 1X2 probabilities.

A thin HTTP client of the inference service (PLAN Epic 06): all logic lives in :mod:`demo.service`
and :mod:`demo.summary`; this script only draws widgets and wires them together. Streamlit reruns the
whole script on every interaction, so the predict call is guarded behind the button press.
"""

from __future__ import annotations

import datetime as dt

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

    chart = pd.DataFrame(
        {"probability": [probs.home_win, probs.draw, probs.away_win]},
        index=list(_OUTCOME_LABELS),
    )
    st.bar_chart(chart, y="probability")

    st.info(verbal_summary(probs))


st.set_page_config(page_title="match-odds — odds this weekend", page_icon="⚽")

st.title("⚽ match-odds — odds this weekend")
st.caption(
    "Calibrated home / draw / away probabilities for a fixture, served by the match-odds inference "
    "service. Pick a league, the two teams, and a match date, then hit Predict."
)

league = st.selectbox("League", available_leagues())
team_names = teams_for(league)
home_box, away_box = st.columns(2)
home = home_box.selectbox("Home team", team_names, index=0)
away = away_box.selectbox("Away team", team_names, index=1 if len(team_names) > 1 else 0)
match_date = st.date_input("Match date", value=dt.date.today())

if st.button("Predict", type="primary"):
    if home == away:
        st.warning("Pick two different teams — a side can't play itself.")
    else:
        try:
            probabilities = PredictClient().predict(home=home, away=away, league=league, match_date=match_date)
        except ServiceError as exc:
            st.error(str(exc))
        else:
            _render_prediction(probabilities, home, away)
