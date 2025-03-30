import os

import streamlit as st

from src.drift_monitor import run_drift_report

st.set_page_config(page_title="Drift Monitor", layout="wide")
st.title("Monitoramento de Drift de Dados")

if st.button("Gerar novo relatório"):
    with st.spinner("Executando análise de drift..."):
        run_drift_report()
    st.success("Relatório atualizado!")

report_path = "drift_report.html"

if os.path.exists(report_path):
    st.components.v1.html(
        open(report_path, "r", encoding="utf-8").read(), height=900, scrolling=True
    )
else:
    st.warning("Relatório ainda não gerado.")
