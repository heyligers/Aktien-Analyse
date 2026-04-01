# ============================================================================
# modules/screener.py — Aktien-Screener-Logik inkl. Custom Formula Builder
# ============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.utils import fmt_number, logger
from modules.bookmarks import load_bookmarks


# ============================================================================
# UNIVERSUM & PRESET-STRATEGIEN
# ============================================================================

INDEX_UNIVERSES = {
    "DAX 40": [
        "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BEI.DE","BMW.DE","BNR.DE",
        "CON.DE","1COV.DE","DHER.DE","DB1.DE","DBK.DE","DHL.DE","DTE.DE","EOAN.DE",
        "FRE.DE","FME.DE","HNR1.DE","HEI.DE","HEN3.DE","IFX.DE","MBG.DE","MRK.DE",
        "MTX.DE","MUV2.DE","P911.DE","PAH3.DE","QIA.DE","RHM.DE","RWE.DE","SAP.DE",
        "SHL.DE","SIE.DE","SY1.DE","VNA.DE","VOW3.DE","VW.DE","ZAL.DE","HFG.DE"
    ],
    "S&P 500 (Top 30)": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","UNH",
        "V","XOM","MA","PG","JNJ","HD","COST","ABBV","MRK","CVX",
        "LLY","BAC","PEP","KO","AVGO","TMO","MCD","WMT","CRM","NFLX"
    ],
    "Nasdaq 100 (Top 20)": [
        "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AVGO","COST","NFLX",
        "ASML","AMD","PEP","ADBE","CSCO","INTC","TXN","QCOM","HON","INTU"
    ],
}

PRESET_STRATEGIES = {
    "▼ Überverkauft (RSI < 30)": {
        "desc": "Aktien, bei denen der RSI unter 30 liegt — mögliche Erholungskandidaten.",
        "filters": {"rsi_max": 30}
    },
    "▲ Überverkauft + über SMA50": {
        "desc": "RSI < 35 aber Kurs über SMA50 — Schwäche im Aufwärtstrend.",
        "filters": {"rsi_max": 35, "above_sma50": True}
    },
    "↑ Golden Cross (SMA20 > SMA50)": {
        "desc": "SMA20 hat SMA50 von unten gekreuzt — klassisches Kaufsignal.",
        "filters": {"golden_cross": True}
    },
    "↓ Death Cross (SMA20 < SMA50)": {
        "desc": "SMA20 hat SMA50 von oben gekreuzt — Warnsignal.",
        "filters": {"death_cross": True}
    },
    "» MACD Bullish Crossover": {
        "desc": "MACD-Linie hat die Signallinie von unten gekreuzt.",
        "filters": {"macd_bullish": True}
    },
    "◆ Günstige Bewertung (KGV < 15)": {
        "desc": "Niedrig bewertete Aktien mit KGV unter 15.",
        "filters": {"pe_max": 15}
    },
    "↑ Breakout (Nahe 52W-Hoch)": {
        "desc": "Kurs liegt innerhalb 5 % des 52-Wochen-Hochs.",
        "filters": {"near_52w_high": True}
    },
}

# Verfügbare Variablen für den Custom Formula Builder
CUSTOM_FORMULA_VARS = {
    "Close": "Aktueller Schlusskurs",
    "SMA20": "Simple Moving Average (20 Tage)",
    "SMA50": "Simple Moving Average (50 Tage)",
    "RSI": "Relative Strength Index (14 Tage)",
    "MACD": "MACD-Linie",
    "Signal": "MACD-Signallinie",
    "PE": "Kurs-Gewinn-Verhältnis (KGV)",
    "MarketCap": "Marktkapitalisierung (in €/$ absolut)",
    "Pct52wHigh": "Abstand vom 52-Wochen-Hoch (in %)",
    "Volume": "Handelsvolumen (letzter Tag)",
}


# ============================================================================
# SCREENING-KERN
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def screen_ticker(sym: str) -> dict | None:
    """Lädt Screening-Daten für einen einzelnen Ticker."""
    macd_cross_bull = False
    try:
        t = yf.Ticker(sym)
        fi = t.fast_info
        info = t.info

        price = getattr(fi, "last_price", None)
        sma20 = sma50 = rsi = macd_val = macd_sig = volume = None
        pct_from_high = None

        df = yf.download(sym, period="90d", interval="1d",
                         auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].squeeze()
            if len(close) >= 50:
                sma20 = float(close.rolling(20).mean().iloc[-1])
                sma50 = float(close.rolling(50).mean().iloc[-1])
            if len(close) >= 14:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, float("nan"))
                rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            if len(close) >= 26:
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd_line = ema12 - ema26
                sig_line = macd_line.ewm(span=9).mean()
                macd_val = float(macd_line.iloc[-1])
                macd_sig = float(sig_line.iloc[-1])
                if len(macd_line) >= 2:
                    macd_cross_bull = bool(
                        (macd_line.iloc[-2] < sig_line.iloc[-2]) and (macd_val > macd_sig))
            if "Volume" in df.columns:
                volume = float(df["Volume"].iloc[-1])

        w52_high = info.get("fiftyTwoWeekHigh")
        if price and w52_high and w52_high > 0:
            pct_from_high = (price / w52_high - 1) * 100

        return {
            "ticker": sym, "name": info.get("shortName", sym)[:25],
            "price": price, "sma20": sma20, "sma50": sma50, "rsi": rsi,
            "macd_val": macd_val, "macd_sig": macd_sig,
            "macd_cross_bull": macd_cross_bull,
            "pe": info.get("trailingPE"), "market_cap": info.get("marketCap"),
            "pct_from_52high": pct_from_high, "w52_high": w52_high,
            "volume": volume,
        }
    except (ValueError, KeyError, AttributeError, requests.RequestException) as e:
        logger.warning(f"Screener-Fehler für {sym}: {e}")
        return None


def apply_filters(results: list, filters: dict) -> list:
    out = []
    for r in results:
        if r is None:
            continue
        p = r.get("price")
        rsi = r.get("rsi")
        sma20 = r.get("sma20")
        sma50 = r.get("sma50")
        pe = r.get("pe")
        pct_h = r.get("pct_from_52high")
        if "rsi_max" in filters:
            if rsi is None or rsi > filters["rsi_max"]:
                continue
        if "rsi_min" in filters:
            if rsi is None or rsi < filters["rsi_min"]:
                continue
        if filters.get("above_sma50"):
            if not (p and sma50 and p > sma50):
                continue
        if filters.get("below_sma50"):
            if not (p and sma50 and p < sma50):
                continue
        if filters.get("golden_cross"):
            if not (sma20 and sma50 and sma20 > sma50):
                continue
        if filters.get("death_cross"):
            if not (sma20 and sma50 and sma20 < sma50):
                continue
        if filters.get("macd_bullish"):
            if not r.get("macd_cross_bull"):
                continue
        if "pe_max" in filters:
            if pe is None or pe <= 0 or pe > filters["pe_max"]:
                continue
        if "pe_min" in filters:
            if pe is None or pe < filters["pe_min"]:
                continue
        if filters.get("near_52w_high"):
            if pct_h is None or pct_h < -5:
                continue
        if "mcap_min" in filters:
            mc = r.get("market_cap")
            if mc is None or mc < filters["mcap_min"]:
                continue
        out.append(r)
    return out


def apply_custom_formula(results: list, formula: str) -> tuple[list, str | None]:
    """
    Wendet eine benutzerdefinierte Formel auf Screening-Ergebnisse an.
    Unterstützte Variablen: Close, SMA20, SMA50, RSI, MACD, Signal, PE, MarketCap, Pct52wHigh, Volume
    Gibt (gefilterte Liste, Fehlermeldung) zurück.
    """
    if not formula.strip():
        return results, None

    valid_results = []
    error_msg = None

    for r in results:
        if r is None:
            continue
        # Variablen-Mapping
        ns = {
            "Close": r.get("price") or 0,
            "SMA20": r.get("sma20") or 0,
            "SMA50": r.get("sma50") or 0,
            "RSI": r.get("rsi") or 0,
            "MACD": r.get("macd_val") or 0,
            "Signal": r.get("macd_sig") or 0,
            "PE": r.get("pe") or 0,
            "MarketCap": r.get("market_cap") or 0,
            "Pct52wHigh": r.get("pct_from_52high") or 0,
            "Volume": r.get("volume") or 0,
            # Erlaubte Python-Builtins
            "abs": abs, "min": min, "max": max,
        }
        try:
            result = eval(formula, {"__builtins__": {}}, ns)
            if result:
                valid_results.append(r)
        except ZeroDivisionError:
            pass
        except (SyntaxError, NameError, TypeError, ValueError) as e:
            error_msg = f"Formel-Fehler: {e}"
            break
        except Exception as e:
            error_msg = f"Unbekannter Fehler in Formel: {e}"
            break

    return valid_results, error_msg


# ============================================================================
# DISPLAY
# ============================================================================

def display_screener():
    st.markdown("## Aktien-Screener")

    # --- Universum ---
    st.markdown("#### 1️⃣ Universum")
    bookmarks = load_bookmarks()
    universe_options = list(INDEX_UNIVERSES.keys())
    if bookmarks:
        universe_options = ["★ Meine Watchlist"] + universe_options
    universe_options.append("Eigene Ticker")

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        selected_universe = st.selectbox("Welche Aktien durchsuchen?", universe_options)
    with col_u2:
        st.markdown("<br>", unsafe_allow_html=True)
        combine_with_watchlist = False
        if selected_universe != "★ Meine Watchlist" and bookmarks:
            combine_with_watchlist = st.checkbox("+ Watchlist hinzufügen", value=False)

    if selected_universe == "★ Meine Watchlist":
        tickers_to_screen = list(bookmarks.keys())
    elif selected_universe == "Eigene Ticker":
        custom_input = st.text_input("Ticker eingeben (kommagetrennt)",
                                     placeholder="z.B. AAPL, SAP.DE, MSFT, BMW.DE")
        tickers_to_screen = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    else:
        tickers_to_screen = INDEX_UNIVERSES[selected_universe].copy()

    if combine_with_watchlist:
        tickers_to_screen = list(set(tickers_to_screen + list(bookmarks.keys())))
    if not tickers_to_screen:
        st.info("Bitte ein Universum wählen oder Ticker eingeben.")
        return
    st.caption(f"{len(tickers_to_screen)} Aktien im Universum")

    # --- Filter ---
    st.markdown("#### 2️⃣ Strategie / Filter")
    filter_mode = st.radio("Modus", ["Vordefinierte Strategie", "Manuelle Filter", "Custom Formula"],
                           horizontal=True)
    active_filters = {}
    custom_formula = ""

    if filter_mode == "Vordefinierte Strategie":
        preset_name = st.selectbox("Strategie wählen", list(PRESET_STRATEGIES.keys()))
        preset = PRESET_STRATEGIES[preset_name]
        st.info(preset["desc"])
        active_filters = preset["filters"].copy()

    elif filter_mode == "Manuelle Filter":
        st.markdown("**Technische Filter**")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            rsi_filter = st.checkbox("RSI-Filter")
            if rsi_filter:
                rsi_min_v = st.slider("RSI min", 0, 100, 0)
                rsi_max_v = st.slider("RSI max", 0, 100, 30)
                if rsi_min_v > 0:
                    active_filters["rsi_min"] = rsi_min_v
                active_filters["rsi_max"] = rsi_max_v
        with tc2:
            sma_filter = st.selectbox("SMA-Crossover",
                ["Kein Filter", "Golden Cross (SMA20 > SMA50)",
                 "Death Cross (SMA20 < SMA50)", "Kurs über SMA50", "Kurs unter SMA50"])
            if sma_filter == "Golden Cross (SMA20 > SMA50)":
                active_filters["golden_cross"] = True
            elif sma_filter == "Death Cross (SMA20 < SMA50)":
                active_filters["death_cross"] = True
            elif sma_filter == "Kurs über SMA50":
                active_filters["above_sma50"] = True
            elif sma_filter == "Kurs unter SMA50":
                active_filters["below_sma50"] = True
        with tc3:
            if st.checkbox("MACD Bullish Crossover"):
                active_filters["macd_bullish"] = True
            if st.checkbox("Nahe 52W-Hoch (< 5 % entfernt)"):
                active_filters["near_52w_high"] = True

        st.markdown("**Fundamentale Filter**")
        fc1, fc2, _fc3 = st.columns(3)
        with fc1:
            pe_filter = st.checkbox("KGV-Filter")
            if pe_filter:
                pe_max_v = st.number_input("KGV max", min_value=1, max_value=200, value=20)
                active_filters["pe_max"] = pe_max_v
        with fc2:
            mcap_filter = st.selectbox("Min. Marktkapitalisierung",
                ["Kein Filter", "> 1 Mrd.", "> 5 Mrd.", "> 10 Mrd.", "> 50 Mrd."])
            mcap_map = {"Kein Filter": 0, "> 1 Mrd.": 1e9, "> 5 Mrd.": 5e9,
                        "> 10 Mrd.": 10e9, "> 50 Mrd.": 50e9}
            if mcap_map.get(mcap_filter, 0) > 0:
                active_filters["mcap_min"] = mcap_map[mcap_filter]

    else:  # Custom Formula (Phase 2: TradingView-Stil)
        st.markdown("#### Custom Formula Builder")
        with st.expander("📖 Verfügbare Variablen & Syntax", expanded=True):
            cols = st.columns(2)
            vars_list = list(CUSTOM_FORMULA_VARS.items())
            half = len(vars_list) // 2
            with cols[0]:
                for var, desc in vars_list[:half]:
                    st.markdown(f"- **`{var}`** — {desc}")
            with cols[1]:
                for var, desc in vars_list[half:]:
                    st.markdown(f"- **`{var}`** — {desc}")
            st.markdown("""
**Operatoren:** `>`, `<`, `>=`, `<=`, `==`, `!=`, `and`, `or`, `not`
**Funktionen:** `abs()`, `min()`, `max()`

**Beispiele:**
```
(SMA20 > SMA50) and (RSI < 40) and (Volume > 100000)
(RSI < 30) and (PE > 0) and (PE < 15)
(Pct52wHigh > -5) and (MACD > Signal)
MarketCap > 1000000000 and RSI < 50
```
""")
        custom_formula = st.text_area(
            "Filterformel eingeben",
            placeholder="(SMA20 > SMA50) and (RSI < 40) and (Volume > 100000)",
            height=100,
            help="Python-artige Bedingungen. Werden auf alle Aktien im Universum angewendet."
        )
        if not custom_formula.strip():
            st.info("Gib eine Formel ein, um den Custom Screener zu nutzen.")

    # --- Scan ---
    st.markdown("#### 3️⃣ Scan")
    can_run = (
        (filter_mode != "Custom Formula" and active_filters) or
        (filter_mode == "Custom Formula" and custom_formula.strip())
    )
    if not can_run:
        if filter_mode != "Custom Formula":
            st.warning("Bitte mindestens einen Filter aktivieren.")
        return

    if st.button("Screener starten", type="primary"):
        progress = st.progress(0, text="Screener läuft…")
        raw_results = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(screen_ticker, sym): sym for sym in tickers_to_screen}
            for i, fut in enumerate(as_completed(futures)):
                raw_results.append(fut.result())
                progress.progress((i + 1) / len(tickers_to_screen),
                                   text=f"Analysiert: {i+1}/{len(tickers_to_screen)}")
        progress.empty()

        if filter_mode == "Custom Formula":
            matches, formula_error = apply_custom_formula(raw_results, custom_formula)
            if formula_error:
                st.error(f"⚠ {formula_error}")
                st.session_state["screener_results"] = []
            else:
                st.session_state["screener_results"] = matches
        else:
            matches = apply_filters(raw_results, active_filters)
            st.session_state["screener_results"] = matches

        st.session_state["screener_total"] = len(tickers_to_screen)

    # --- Ergebnisse ---
    matches = st.session_state.get("screener_results")
    total_scanned = st.session_state.get("screener_total", 0)

    if matches is not None:
        if not matches:
            st.warning("Keine Aktien gefunden, die alle Filter erfüllen.")
            return
        st.success(f"✓ **{len(matches)} Treffer** von {total_scanned} gescannten Aktien")
        rows = []
        for r in sorted(matches, key=lambda x: x.get("rsi") or 999):
            rsi_v = r.get("rsi")
            rows.append({
                "Ticker": r["ticker"], "Name": r["name"],
                "Kurs": f"{r['price']:.2f}" if r.get("price") else "—",
                "RSI (14)": f"{rsi_v:.1f}" if rsi_v else "—",
                "SMA20": f"{r['sma20']:.2f}" if r.get("sma20") else "—",
                "SMA50": f"{r['sma50']:.2f}" if r.get("sma50") else "—",
                "MACD": f"{r['macd_val']:.3f}" if r.get("macd_val") else "—",
                "Signal": f"{r['macd_sig']:.3f}" if r.get("macd_sig") else "—",
                "vom 52W-Hoch": f"{r['pct_from_52high']:+.1f} %" if r.get("pct_from_52high") is not None else "—",
                "KGV": f"{r['pe']:.1f}" if r.get("pe") else "—",
                "Mkt. Cap": fmt_number(r.get("market_cap"), large=True),
            })
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Als CSV exportieren", csv_data,
                           "screener_ergebnisse.csv", "text/csv")

        st.markdown("---")
        st.markdown("**Treffer direkt analysieren:**")
        ticker_buttons = [r["ticker"] for r in matches]
        cols = st.columns(min(len(ticker_buttons), 6))
        for i, sym in enumerate(ticker_buttons[:12]):
            with cols[i % 6]:
                if st.button(sym, key=f"screen_jump_{sym}"):
                    st.session_state["jump_ticker"] = sym
                    st.rerun()
