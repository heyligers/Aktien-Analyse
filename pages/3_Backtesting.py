# ============================================================================
# pages/3_Backtesting.py — Backtesting-Seite (Phase 3)
# ============================================================================

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Backtesting · Aktien Pro v8", layout="wide", page_icon="")

from modules.bookmarks import display_watchlist
from modules.backtesting import display_backtesting
from modules.backtesting import compare_all_strategies
from modules.backtesting import run_backtest
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from modules.ui_components import display_portfolio, display_correlation_heatmap, display_stock_ticker
from modules.data_api import calculate_risk_metrics, calculate_correlation_matrix
from modules.bookmarks import load_portfolio

# Sidebar
st.sidebar.header("Navigation")
if st.sidebar.button(" Zur Analyse", use_container_width=True):
    st.switch_page("pages/1_Analyse.py")
if st.sidebar.button(" Screener", use_container_width=True):
    st.switch_page("pages/2_Screener.py")

display_stock_ticker()
display_watchlist()

# Tabs
tab_bt, tab_compare, tab_portfolio = st.tabs([
    " Backtesting",
    " Strategievergleich",
    " Portfolio"
])

with tab_bt:
    display_backtesting()

with tab_compare:
    st.markdown("## Alle Strategien vergleichen")
    st.caption("Simuliert alle verfügbaren Strategien auf demselben Ticker und Zeitraum.")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        cmp_ticker_raw = st.text_input("Ticker", value="AAPL", key="cmp_ticker")
        from modules.data_api import get_ticker
        cmp_ticker = get_ticker(cmp_ticker_raw)
        st.caption(f"Symbol: **{cmp_ticker}**")
    with cc2:
        cmp_period = st.selectbox("Zeitraum", ["1 Jahr", "2 Jahre", "3 Jahre", "5 Jahre"],
                                   index=1, key="cmp_period")
        cmp_days = {"1 Jahr": 365, "2 Jahre": 730, "3 Jahre": 1095, "5 Jahre": 1825}[cmp_period]
    with cc3:
        cmp_capital = st.number_input("Startkapital ($)", min_value=1000, value=10000,
                                       step=1000, key="cmp_capital")

    if st.button("Alle Strategien starten", type="primary", key="run_compare"):
        with st.spinner("Lade Kursdaten und berechne alle Strategien..."):
            try:
                end = datetime.today().date()
                start = end - timedelta(days=cmp_days)
                df_cmp = yf.download(cmp_ticker, start=start, end=end,
                                     interval="1d", auto_adjust=True, progress=False)
                if isinstance(df_cmp.columns, pd.MultiIndex):
                    df_cmp.columns = df_cmp.columns.get_level_values(0)
                df_cmp = df_cmp.loc[:, ~df_cmp.columns.duplicated()]

                if df_cmp.empty or len(df_cmp) < 60:
                    st.error("Zu wenige Daten.")
                else:
                    cmp_results = compare_all_strategies(df_cmp, float(cmp_capital))
                    st.session_state["cmp_results"] = cmp_results
                    st.session_state["cmp_ticker_name"] = cmp_ticker
                    st.success(f"{len(cmp_results)} Strategien berechnet!")
            except Exception as e:
                st.error(f"Fehler: {e}")

    if "cmp_results" in st.session_state:
        cmp_res = st.session_state["cmp_results"]
        ticker_name = st.session_state.get("cmp_ticker_name", "")

        # ─── Equity Curve Übersicht Chart ───
        colors = ["#26a69a", "#2196f3", "#ff9800", "#e91e63",
                  "#9c27b0", "#00bcd4", "#4caf50", "#ff5722"]
        fig_cmp = go.Figure()
        for i, (name, r) in enumerate(cmp_res.items()):
            eq = r["equity"]
            fig_cmp.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                mode="lines", name=name,
                line=dict(color=colors[i % len(colors)], width=1.8)
            ))
        fig_cmp.update_layout(
            template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
            height=420, margin=dict(l=0, r=10, t=30, b=0),
            legend=dict(orientation="h", y=-0.15),
            title=f"Equity Curve Vergleich — {ticker_name}"
        )
        fig_cmp.update_yaxes(gridcolor="#1e2230")
        fig_cmp.update_xaxes(gridcolor="#1e2230")
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ─── Metriken Tabelle ───
        rows_cmp = []
        for name, r in cmp_res.items():
            m = r["metrics"]
            rows_cmp.append({
                "Strategie": name,
                "Rendite %": f"{m['net_profit_pct']:+.1f}%",
                "CAGR %": f"{m.get('cagr', 0):+.1f}%",
                "Sharpe": f"{m['sharpe_ratio']:.3f}",
                "Sortino": f"{m.get('sortino_ratio', 0):.3f}",
                "Max DD": f"{m['max_drawdown']:.1f}%",
                "Win Rate": f"{m['win_rate']:.1f}%",
                "Trades": m["total_trades"],
                "B&H": f"{m.get('buy_hold_return', 0):+.1f}%"
            })
        df_cmp_rows = pd.DataFrame(rows_cmp)
        st.dataframe(df_cmp_rows, use_container_width=True, hide_index=True)

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
