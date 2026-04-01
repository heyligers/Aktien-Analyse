# ============================================================================
# pages/3_Backtesting.py — Backtesting-Seite (Phase 3)
# ============================================================================

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Backtesting · Aktien Pro v8", layout="wide", page_icon="⚡")

from modules.bookmarks import display_watchlist
from modules.backtesting import display_backtesting
from modules.ui_components import display_portfolio, display_correlation_heatmap
from modules.data_api import calculate_risk_metrics, calculate_correlation_matrix
from modules.bookmarks import load_portfolio

# Sidebar
st.sidebar.header("Navigation")
if st.sidebar.button("⬅ Zur Analyse", use_container_width=True):
    st.switch_page("pages/1_Analyse.py")
if st.sidebar.button("🔍 Screener", use_container_width=True):
    st.switch_page("pages/2_Screener.py")

display_watchlist()

# Tabs
tab_bt, tab_portfolio = st.tabs(["⚡ Backtesting", "💼 Portfolio"])

with tab_bt:
    display_backtesting()

with tab_portfolio:
    display_portfolio()

    portfolio = load_portfolio()
    if portfolio:
        unique_tickers = list(set(pos.get('ticker', k) for k, pos in portfolio.items()))

        if len(unique_tickers) >= 1:
            st.markdown("---")
            st.markdown("### Risk-Metriken")
            if st.button("Risk-Analyse laden", type="primary", key="load_risk") or \
               st.session_state.get("risk_loaded"):
                st.session_state["risk_loaded"] = True
                with st.spinner("Risk-Metriken werden berechnet…"):
                    risk_data = calculate_risk_metrics(unique_tickers)
                if risk_data:
                    import pandas as pd
                    rows = []
                    for sym, m in risk_data.items():
                        rows.append({
                            "Ticker": sym,
                            "Rendite p.a.": f"{m['ann_return']:+.1f}%",
                            "Volatilität p.a.": f"{m['ann_vol']:.1f}%",
                            "Sharpe": f"{m['sharpe']:.3f}",
                            "Sortino": f"{m['sortino']:.3f}",
                            "Max Drawdown": f"{m['max_drawdown']:.1f}%",
                            "VaR (95%)": f"{m['var_95']:.2f}%",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption("Basierend auf 1 Jahr täglicher Returns · Risk-Free Rate: 4%")
                else:
                    st.info("Keine Risk-Daten verfügbar.")

        if len(unique_tickers) >= 2:
            st.markdown("### Korrelations-Matrix")
            if st.button("Korrelation laden", type="primary", key="load_corr") or \
               st.session_state.get("corr_loaded"):
                st.session_state["corr_loaded"] = True
                with st.spinner("Korrelations-Matrix wird berechnet…"):
                    corr_df = calculate_correlation_matrix(unique_tickers)
                display_correlation_heatmap(corr_df)
