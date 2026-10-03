import os

import streamlit as st

from src.drift_monitor import DRIFT_SHARE_THRESHOLD, run_drift_check

st.set_page_config(page_title="Drift Monitor", layout="wide")
st.title("Data Drift Monitoring")
st.caption(
    "Compares real requests logged in data/logs/requests.jsonl against the "
    f"training reference distribution (alert threshold: {DRIFT_SHARE_THRESHOLD:.0%})."
)

if st.button("Generate new report"):
    with st.spinner("Running drift analysis..."):
        try:
            drift_share = run_drift_check()
            if drift_share >= DRIFT_SHARE_THRESHOLD:
                st.error(f"Drift above threshold: {drift_share:.0%}")
            else:
                st.success(f"Report updated. Drift: {drift_share:.0%}")
        except (FileNotFoundError, ValueError) as e:
            st.warning(str(e))

report_path = "drift_report.html"

if os.path.exists(report_path):
    st.components.v1.html(open(report_path, encoding="utf-8").read(), height=900, scrolling=True)
else:
    st.warning("Report not generated yet.")
