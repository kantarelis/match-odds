"""Headless AppTest smoke for the Streamlit demo — no browser, no network.

``streamlit.testing.v1.AppTest`` runs the script in-process. The predict path's HTTP call is the only
thing that would touch the network, so we monkeypatch ``PredictClient.predict`` (the predict/error
parsing itself is already covered by ``test_service.py``); everything else is exercised for real.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from demo.service import PredictClient, Probabilities, ServiceError

_APP = str(Path(__file__).resolve().parents[1] / "app.py")
_PROBS = Probabilities(home_win=0.6, draw=0.25, away_win=0.15)


def test_initial_render_populates_selectboxes():
    at = AppTest.from_file(_APP).run()

    assert not at.exception
    # League + home + away selectboxes, each populated from the canonical registry.
    assert len(at.selectbox) == 3
    # No date picker — the demo forecasts the next (unscheduled) meeting.
    assert len(at.date_input) == 0
    assert at.selectbox[0].options  # leagues
    assert at.selectbox[1].options  # home teams
    assert at.selectbox[0].value != at.selectbox[1].value or len(at.selectbox[1].options) == 1


def test_predict_path_renders_metrics_and_summary(monkeypatch):
    monkeypatch.setattr(PredictClient, "predict", lambda self, **kwargs: _PROBS)

    at = AppTest.from_file(_APP).run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.metric) == 3
    assert any("60%" in info.value and "favours" in info.value for info in at.info)


def test_service_error_shows_friendly_message(monkeypatch):
    def _raise(self, **kwargs):
        raise ServiceError("service down")

    monkeypatch.setattr(PredictClient, "predict", _raise)

    at = AppTest.from_file(_APP).run()
    at.button[0].click().run()

    assert not at.exception
    assert at.error
    assert "service down" in at.error[0].value
