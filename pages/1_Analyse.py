# ============================================================================
# pages/1_Analyse.py — Haupt-Analyse-Seite (Chart + News + Fundamentals + KI)
# ============================================================================

import streamlit as st
import os
import sys
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Analyse · Aktien Pro v8", layout="wide", page_icon="")

from modules.utils import get_api_key_value
from modules.data_api import (
    get_ticker, load_data, get_full_ticker_data, get_ticker_info,
    get_fundamentals, get_options_data, get_macro_context,
)
from modules.bookmarks import (
    load_bookmarks, add_bookmark, remove_bookmark, display_watchlist,
)
from modules.news_api import get_combined_news, get_combined_market_news, display_news_aesthetic
from modules.technical_analysis import (
    detect_candlestick_patterns, calculate_pivot_points,
    calculate_fibonacci, sr_fib_to_lwc_lines,
)
from modules.charting import build_lwc_html
from modules.ai_gemini import display_ai_news
from modules.ui_components import (
    display_fundamentals, display_options, display_macro,
    display_comparison, display_insider, display_social_sentiment,
    display_stock_ticker
)
from modules.report_generator import display_pdf_export

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Optionen")

_auto_key = get_api_key_value()
news_api_key = st.sidebar.text_input(
    "NewsAPI Key", value=_auto_key, type="password",
    help="Optional: Für historische News. Aus st.secrets oder NEWSAPI_KEY env."
)

st.sidebar.markdown("---")

display_watchlist()

# ── Hauptbereich ─────────────────────────────────────────────────────────────
display_stock_ticker()

st.title(" Analyse")

# Ticker/Suche & Zeitspanne oben
if "jump_ticker" in st.session_state:
    _default_input = st.session_state.pop("jump_ticker")
else:
    _default_input = "Apple"

col_header1, col_header2, col_header3, col_header4 = st.columns([2.2, 0.8, 1.8, 0.8])
with col_header1:
    user_input = st.text_input("Ticker", _default_input, label_visibility="collapsed", placeholder="Name oder Ticker...")
    ticker = get_ticker(user_input)

with col_header2:
    intervals = {"1 Min": "1m", "15 Min": "15m", "1 Std": "1h", "4 Std": "4h", "1 Tag": "1d", "1 Woche": "1wk"}
    sel_inter = st.selectbox("Intervall", list(intervals.keys()), index=4, label_visibility="collapsed")
    interval = intervals[sel_inter]

with col_header3:
    default_end = datetime.today().date()
    if interval == "1m":
        default_start = default_end - timedelta(days=6) # Max. 7 Tage für 1m
    elif interval == "15m":
        default_start = default_end - timedelta(days=59) # Max. 60 Tage für < 60m
    elif interval in ("1h", "4h"):
        default_start = default_end - timedelta(days=729) # Max. 730 Tage für 1h/4h
    else:
        default_start = default_end - timedelta(days=365)
        
    date_selection = st.date_input("Zeitraum", value=(default_start, default_end), max_value=default_end, label_visibility="collapsed")
    if len(date_selection) == 2:
        start_date, end_date = date_selection
    else:
        start_date, end_date = date_selection[0], date_selection[0]

with col_header4:
    _bm = load_bookmarks()
    _is_bm = ticker in _bm
    _bm_label = "★" if _is_bm else "☆"
    if st.button(_bm_label, key="bm_toggle", use_container_width=True, help="Zur Watchlist hinzufügen/entfernen"):
        if _is_bm:
            remove_bookmark(ticker)
            st.toast(f"{ticker} entfernt.")
        else:
            _info = get_ticker_info(ticker)
            add_bookmark(ticker, _info.get("name", ticker))
            st.toast(f"{ticker} hinzugefügt.")
        st.rerun()
col_main, col_side = st.columns([5, 3])

try:
    df = load_data(ticker, start_date, end_date, interval)

    if df is None or df.empty:
        st.error(f"Keine Daten für **{ticker}** im gewählten Zeitraum.")
    else:
        with col_main:
            # Indikatoren-Toolbar (Platzsparend mit Popover)
            with st.popover("Indikatoren & Einstellungen", use_container_width=True):
                pop_col1, pop_col2, pop_col3 = st.columns(3)
                with pop_col1:
                    st.markdown("**Trend/Overlay**")
                    sma_on = st.checkbox("SMA (20/50)", value=True)
                    ema_on = st.checkbox("EMA (20/50)", value=False)
                    bb_on = st.checkbox("Bollinger Bands", value=False)
                    vwap_on = st.checkbox("VWAP", value=False)
                with pop_col2:
                    st.markdown("**Struktur**")
                    cdl_on = st.checkbox("Candles (Patterns)", value=False)
                    sr_on = st.checkbox("Support/Resistance", value=False)
                    fib_on = st.checkbox("Fibonacci Levels", value=False)
                    ich_on = st.checkbox("Ichimoku Cloud", value=False)
                with pop_col3:
                    st.markdown("**Oszillatoren**")
                    rsi_on = st.checkbox("RSI (14)", value=False)
                    macd_on = st.checkbox("MACD", value=False)
                    stoch_on = st.checkbox("Stochastik", value=False)
                    atr_on = st.checkbox("ATR", value=False)
                    obv_on = st.checkbox("OBV", value=False)
                    willr_on = st.checkbox("Williams %R", value=False)

            # --- Indikatoren berechnen (NACH der Definition) ---
            if sma_on:
                df.ta.sma(length=20, append=True)
                df.ta.sma(length=50, append=True)
            if ema_on:
                df.ta.ema(length=20, append=True)
                df.ta.ema(length=50, append=True)
            if bb_on:
                df.ta.bbands(length=20, append=True)
            if rsi_on:
                df.ta.rsi(length=14, append=True)
            if macd_on:
                df.ta.macd(append=True)
            if stoch_on:
                df.ta.stoch(append=True)
            if vwap_on:
                typical = (df["High"] + df["Low"] + df["Close"]) / 3
                df["VWAP"] = (df["Volume"] * typical).cumsum() / df["Volume"].cumsum()
            if atr_on:
                df.ta.atr(length=14, append=True)
            if obv_on:
                df.ta.obv(append=True)
            if willr_on:
                df.ta.willr(length=14, append=True)
            if ich_on:
                ich = df.ta.ichimoku(lookahead=False)
                if ich is not None and len(ich) == 2:
                    span_df = ich[1] if hasattr(ich[1], 'columns') else None
                    base_df = ich[0] if hasattr(ich[0], 'columns') else None
                    if base_df is not None:
                        for col in base_df.columns:
                            df[col] = base_df[col]
                    if span_df is not None:
                        for col in span_df.columns:
                            df[col] = span_df[col]

            cdl_markers = detect_candlestick_patterns(df) if cdl_on else []
            pivot_data = calculate_pivot_points(df) if sr_on else None
            fib_data = calculate_fibonacci(df) if fib_on else None
            sr_fib_js = sr_fib_to_lwc_lines(pivot_data, fib_data, df) if (sr_on or fib_on) else ""

            chart_height = 520 + 100 + sum([
                120 if rsi_on else 0, 120 if macd_on else 0, 120 if stoch_on else 0,
                100 if atr_on else 0, 100 if obv_on else 0, 100 if willr_on else 0,
            ]) + 66

            html_chart = build_lwc_html(
                df, sma_on, ema_on, bb_on, vwap_on,
                rsi_on, macd_on, stoch_on,
                atr_on, obv_on, willr_on, ich_on, interval,
                cdl_markers=cdl_markers,
                sr_fib_js=sr_fib_js
            )
            st.components.v1.html(html_chart, height=chart_height, scrolling=False)

            with st.expander("Ticker-Vergleich"):
                display_comparison(ticker, start_date, end_date)

        with col_side:
            info = get_ticker_info(ticker)
            company_name = info["name"]
            country = info["country"]
            currency = info["currency"]

            if country == "Germany" or currency == "EUR" or ticker.endswith(".DE") or ticker.endswith(".F"):
                market_proxy = "^GDAXI"
                market_query = "DAX OR EZB OR Eurozone"
                n_lang = "de"; y_reg = "DE"; y_lang = "de-DE"
            elif country == "United Kingdom" or currency == "GBP" or ticker.endswith(".L"):
                market_proxy = "^FTSE"
                market_query = "FTSE OR Bank of England"
                n_lang = "en"; y_reg = "GB"; y_lang = "en-GB"
            elif country == "Japan" or currency == "JPY" or ticker.endswith(".T"):
                market_proxy = "^N225"
                market_query = "Nikkei OR Bank of Japan"
                n_lang = "en"; y_reg = "US"; y_lang = "en-US"
            else:
                market_proxy = "^GSPC"
                market_query = '"S&P 500" OR "Wall Street" OR "Federal Reserve"'
                n_lang = "en"; y_reg = "US"; y_lang = "en-US"

            tab_news, tab_market, tab_fund, tab_opt, tab_macro, tab_ai, tab_insider, tab_social = st.tabs([
                "News", "Markt", "Fund.", "Opt.", "Makro", "KI", "Insider", "Social"
            ])

            with tab_news:
                news_date = st.date_input("News-Datum", value=end_date,
                                          max_value=datetime.today().date(),
                                          key="news_date_input")
                with st.container(height=720):
                    with st.spinner("News werden geladen…"):
                        final_news = get_combined_news(
                            ticker, company_name, news_api_key,
                            y_reg, y_lang, n_lang, news_date
                        )
                    display_news_aesthetic(final_news, n_lang)

            with tab_market:
                with st.container(height=780):
                    with st.spinner("Markt-News werden geladen…"):
                        market_news = get_combined_market_news(
                            market_proxy, market_query, news_api_key,
                            y_reg, y_lang, n_lang, end_date
                        )
                    display_news_aesthetic(market_news, n_lang)

            with tab_fund:
                with st.container(height=780):
                    with st.spinner("Fundamentaldaten werden geladen…"):
                        fund_data = get_fundamentals(ticker)
                    display_fundamentals(fund_data, ticker)

            with tab_opt:
                with st.container(height=780):
                    if st.session_state.get(f"opt_loaded_{ticker}") or \
                       st.button("Optionsdaten laden", type="primary", key="load_opt"):
                        st.session_state[f"opt_loaded_{ticker}"] = True
                        with st.spinner("Optionsdaten werden geladen…"):
                            opt_data = get_options_data(ticker)
                        display_options(opt_data, ticker)
                    else:
                        st.info("Klicke zum Laden der Optionsdaten (nur US-Aktien).")

            with tab_macro:
                with st.container(height=780):
                    if st.session_state.get(f"macro_loaded_{ticker}") or \
                       st.button("Makrodaten laden", type="primary", key="load_macro"):
                        st.session_state[f"macro_loaded_{ticker}"] = True
                        with st.spinner("Makrodaten werden geladen…"):
                            macro_data = get_macro_context(ticker, market_proxy)
                        display_macro(macro_data, ticker, market_proxy)
                    else:
                        st.info("Klicke zum Laden der Makro/Sektordaten.")

            with tab_ai:
                with st.container(height=780):
                    display_ai_news(ticker, company_name, n_lang)

            with tab_insider:
                with st.container(height=780):
                    if st.session_state.get(f"insider_loaded_{ticker}") or \
                       st.button("Insider-Daten laden", type="primary", key="load_insider"):
                        st.session_state[f"insider_loaded_{ticker}"] = True
                        display_insider(ticker)
                    else:
                        st.info("Klicke zum Laden der Insider-Transaktionen.")

            with tab_social:
                with st.container(height=780):
                    if st.session_state.get(f"social_loaded_{ticker}") or \
                       st.button("Social Sentiment laden", type="primary", key="load_social"):
                        st.session_state[f"social_loaded_{ticker}"] = True
                        display_social_sentiment(ticker, company_name, n_lang)
                    else:
                        st.info("Klicke zum Laden von Reddit-Sentiment.")

        # ── PDF-Export (unterhalb Chart) ─────────────────────────────────────
        st.markdown("---")
        fund_data_pdf = get_fundamentals(ticker)
        ai_summary_pdf = st.session_state.get(f"ai_summary_{ticker}", "")
        bt_metrics_pdf = st.session_state.get("bt_result", {}).get("metrics") \
            if st.session_state.get("bt_run_ticker") == ticker else None

        # Erstelle einen einfachen Plotly-Chart speziell für den PDF-Export (LWC ist nicht als Bild exportierbar)
        import plotly.graph_objects as go
        pdf_chart_fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Preis"
        )])
        pdf_chart_fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            paper_bgcolor='rgba(19,23,34,1)',
            plot_bgcolor='rgba(19,23,34,1)'
        )

        display_pdf_export(
            ticker=ticker,
            company_name=company_name,
            fund_data=fund_data_pdf,
            ai_summary=ai_summary_pdf,
            chart_fig=pdf_chart_fig,
            bt_metrics=bt_metrics_pdf,
        )

except Exception as e:
    st.error(f"Kritischer Fehler: {e}")
    st.exception(e)
