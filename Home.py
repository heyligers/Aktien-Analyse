# ============================================================================
# Home.py — Aktien Analyse Pro v8 · Hauptdatei (Multi-Page-App)
# ============================================================================
# Starten mit: streamlit run Home.py
# ============================================================================

import streamlit as st
import os
import sys

# Stelle sicher, dass das Projektverzeichnis im Suchpfad liegt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ui_components import display_stock_ticker

st.set_page_config(
    page_title="Aktien Analyse",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ticker ──────────────────────────────────────────────────────────────────
display_stock_ticker()

# ── Willkommens-Seite ────────────────────────────────────────────────────────
st.title(" Aktien Analyse Pro v8")
st.markdown("**Professionelle Chart-, News- & Fundamentalanalyse**")

st.markdown("""
<style>
.feature-card {
    background: rgba(41, 98, 255, 0.08);
    border: 1px solid rgba(41, 98, 255, 0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.feature-card h4 { color: #2962FF; margin: 0 0 6px 0; font-size: 1rem; }
.feature-card p  { color: #d1d4dc; margin: 0; font-size: 0.875rem; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
<div class="feature-card">
  <h4> Analyse (Seite 1)</h4>
  <p>Interaktiver LWC-Chart mit 10+ Indikatoren, Zeichenwerkzeugen, S/R-Linien, Fibonacci.
  News, Fundamentaldaten, Optionen, Makro, KI-Analyse via Gemini.</p>
</div>
<div class="feature-card">
  <h4> Screener (Seite 2)</h4>
  <p>Aktien-Screener über DAX 40, S&P 500, Nasdaq 100 oder eigene Ticker.
  Vordefinierte Strategien, manuelle Filter und <b>Custom Formula Builder</b> (TradingView-Stil).</p>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="feature-card">
  <h4> Backtesting (Seite 3)</h4>
  <p>Strategietests auf historischen Daten: RSI, Golden Cross, MACD, Bollinger Bands.
  Equity Curve, Trade-Marker, Sharpe Ratio, Max Drawdown, Win Rate.</p>
</div>
<div class="feature-card">
  <h4> KI Forecast (Seite 4)</h4>
  <p>Maschinelles Lernen mit Scikit-Learn: Polynomiale Regression für Trendkanäle
  sowie Random-Forest Mustererkennung zur Wahrscheinlichkeitsschätzung.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Quick Navigation
st.markdown("### Schnellstart")
qc1, qc2, qc3, qc4 = st.columns(4)
with qc1:
    if st.button(" Zur Analyse", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Analyse.py")
with qc2:
    if st.button(" Zum Screener", use_container_width=True):
        st.switch_page("pages/2_Screener.py")
with qc3:
    if st.button(" Zum Backtesting", use_container_width=True):
        st.switch_page("pages/3_Backtesting.py")
with qc4:
    if st.button(" Zum Forecast", use_container_width=True):
        st.switch_page("pages/4_Forecast.py")

st.markdown("---")
st.markdown("""
<div style="font-size:0.8rem;color:#555;text-align:center;">
  Aktien Analyse Pro v8 · Powered by Yahoo Finance · Gemini AI · LightweightCharts<br>
  Nur zu Informationszwecken — keine Anlageberatung
</div>
""", unsafe_allow_html=True)
