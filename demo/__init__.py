"""Streamlit demo: a thin HTTP client of the inference service (Epic 06).

The demo never imports ``serving.app`` and never loads the model — it POSTs fixtures to the running
service and renders the response. Reusable logic (service client, option helpers, verbal summary)
lives in importable modules here; ``demo/app.py`` is the thin Streamlit glue (Task 2).
"""
