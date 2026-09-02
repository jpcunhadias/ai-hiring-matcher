import os

import streamlit as st

from src.drift_monitor import DRIFT_SHARE_THRESHOLD, run_drift_check

st.set_page_config(page_title="Drift Monitor", layout="wide")
st.title("Monitoramento de Drift de Dados")
st.caption(
    "Compara requisições reais registradas em data/logs/requests.jsonl contra a "
    f"distribuição de referência do treino (limite de alerta: {DRIFT_SHARE_THRESHOLD:.0%})."
)

if st.button("Gerar novo relatório"):
    with st.spinner("Executando análise de drift..."):
        try:
            drift_share = run_drift_check()
            if drift_share >= DRIFT_SHARE_THRESHOLD:
                st.error(f"Drift acima do limite: {drift_share:.0%}")
            else:
                st.success(f"Relatório atualizado. Drift: {drift_share:.0%}")
        except (FileNotFoundError, ValueError) as e:
            st.warning(str(e))

report_path = "drift_report.html"

if os.path.exists(report_path):
    st.components.v1.html(open(report_path, encoding="utf-8").read(), height=900, scrolling=True)
else:
    st.warning("Relatório ainda não gerado.")
