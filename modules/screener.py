# ============================================================================
# modules/screener.py — Aktien-Screener-Logik inkl. Custom Formula Builder
# ============================================================================

import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.utils import fmt_number, logger
from modules.bookmarks import load_bookmarks
from modules.index_utils import get_full_universe


# ============================================================================
# UNIVERSUM & PRESET-STRATEGIEN
# ============================================================================

# Verfügbare Index-Universen — Ticker werden dynamisch aus index_constituents.json geladen
INDEX_UNIVERSES = {
    "DAX 40": "dax",
    "S&P 500": "sp500",
    "Nasdaq 100": "nasdaq100",
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
    " TTM Squeeze Pro Fired": {
        "desc": "Squeeze wurde in den letzten 2 Tagen beendet, Ausbruch nach oben (Momentum positiv).",
        "filters": {"ttm_squeeze_fired": True}
    },
    "◆ In Squeeze (Kompression)": {
        "desc": "Bollinger Bänder liegen innerhalb der Keltner Kanäle — Volatilität wird komprimiert.",
        "filters": {"ttm_is_sqz": True}
    },
    " Donchian Breakout (20 Tage)": {
        "desc": "Schlusskurs hat das höchste Hoch der letzten 20 Tage überschritten.",
        "filters": {"donchian_breakout": True}
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
    "TTM_Squeeze": "Squeeze-Ausbruch (Fired) nach oben (True/False)",
    "TTM_In_Squeeze": "Aktuell im Squeeze (True/False)",
    "Donchian_Break": "Donchian 20-Tage Breakout (True/False)",
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
        
        # yfinance info can be unreliable and raise TypeErrors internally
        try:
            info = t.info
        except Exception:
            info = {}
            
        if info is None:
            info = {}

        price = getattr(fi, "last_price", None) if fi else None
        sma20 = sma50 = rsi = macd_val = macd_sig = volume = None
        pct_from_high = None
        ttm_fired = False
        ttm_is_sqz = False
        donchian_break = False
        ret_12m = None  # 12-Monats-Rendite für RS-Rating

        # 12-Monats-Daten für RS-Rating (252 Handelstage)
        df = yf.download(sym, period="252d", interval="1d",
                         auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                # Flache Spalten, falls MultiIndex (Passiert oft bei neueren yfinance-Versionen)
                df.columns = df.columns.get_level_values(0)
            
            # Falls durch das Flachen Duplikate entstanden sind (z.B. zwei Close-Spalten), 
            # nehmen wir nur die erste.
            df = df.loc[:, ~df.columns.duplicated()]

            # Sicherstellen, dass wir Series erhalten, keine DataFrames
            def get_s(name, fallback=None):
                if name in df.columns:
                    s = df[name]
                    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
                return fallback

            close = get_s("Close")
            if close is None:
                # Letzter Ausweg: Erste verfügbare Spalte
                close = df.iloc[:, 0]
            
            high = get_s("High", fallback=close)
            low = get_s("Low", fallback=close)

            if len(close) >= 50:
                sma20 = float(close.rolling(20).mean().ffill().iloc[-1])
                sma50 = float(close.rolling(50).mean().ffill().iloc[-1])
            # RS-Rating: 12-Monats-Rendite (oder so viel wie verfügbar)
            if len(close) >= 2:
                close_f = close.ffill().dropna()
                ret_12m = float((close_f.iloc[-1] / close_f.iloc[0]) - 1) * 100
            if len(close) >= 14:
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = (-delta.clip(upper=0))
                # Wilder's Smoothing (EWM) - Konsistent mit backtesting.py
                avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
                avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
                rs = avg_gain / avg_loss.replace(0, float("nan"))
                rsi_series = 100 - (100 / (1 + rs))
                rsi = float(rsi_series.ffill().iloc[-1])
            if len(close) >= 26:
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd_line = ema12 - ema26
                sig_line = macd_line.ewm(span=9).mean()
                macd_val = float(macd_line.ffill().iloc[-1])
                macd_sig = float(sig_line.ffill().iloc[-1])
                if len(macd_line) >= 2:
                    macd_line_f = macd_line.ffill()
                    sig_line_f = sig_line.ffill()
                    macd_cross_bull = bool(
                        (macd_line_f.iloc[-2] < sig_line_f.iloc[-2]) and (macd_val > macd_sig))
            vol_s = get_s("Volume")
            if vol_s is not None:
                volume = float(vol_s.ffill().iloc[-1])

            # --- NEU: Donchian Breakout (20 Tage) ---
            if len(close) >= 21:
                highest_high = close.rolling(20).max().shift(1)
                # Prüfen, ob der heutige Kurs (oder der letzte verfügbare) das 20-Tage-Hoch gebrochen hat
                highest_high_f = highest_high.ffill()
                close_f = close.ffill()
                if not pd.isna(highest_high_f.iloc[-1]):
                    donchian_break = bool(close_f.iloc[-1] > highest_high_f.iloc[-1])

            # --- NEU: TTM Squeeze Pro (vereinfacht für Screener-Signal) ---
            if len(close) >= 20:
                length = 20
                basis = close.rolling(length).mean()
                dev = 2.0 * close.rolling(length).std(ddof=0)
                bb_upper = basis + dev
                bb_lower = basis - dev

                tr1 = high - low
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                dev_kc = tr.rolling(length).mean()
                    
                kc_upper_low = basis + dev_kc * 1.5
                kc_lower_low = basis - dev_kc * 1.5

                # Squeeze is 'on' when Bollinger Bands are INSIDE Keltner Channels
                sqz_on = (bb_lower > kc_lower_low) & (bb_upper < kc_upper_low)
                
                # Fired: transition from Squeeze-On to Squeeze-Off (release of volatility)
                # We check the last 2 bars for the transition to be more sensitive.
                sqz_fired_series = (sqz_on.shift(1) == True) & (sqz_on == False)
                
                highest = high.rolling(length).max()
                lowest = low.rolling(length).min()
                avg_hl_basis = ((highest + lowest) / 2 + basis) / 2
                val = close - avg_hl_basis

                x_diff = np.arange(length) - (length - 1) / 2
                sum_x2 = length * (length**2 - 1) / 12
                weights = (1.0 / length) + (x_diff / sum_x2) * ((length - 1) / 2)
                mom = val.rolling(length).apply(lambda y: np.dot(y, weights), raw=True)

                # Results for the latest bars
                if not pd.isna(sqz_on.iloc[-1]):
                    ttm_is_sqz = bool(sqz_on.iloc[-1])
                
                # Check last 2 days for "Fired" signal
                if len(sqz_fired_series) >= 2:
                    recent_fired = sqz_fired_series.iloc[-2:].any()
                    last_mom = mom.iloc[-1]
                    if not pd.isna(recent_fired) and not pd.isna(last_mom):
                        ttm_fired = bool(recent_fired and last_mom > 0)

        w52_high = info.get("fiftyTwoWeekHigh")
        if price and w52_high and w52_high > 0:
            pct_from_high = (price / w52_high - 1) * 100

        return {
            "ticker": sym, "name": info.get("shortName", sym)[:25],
            "price": price, "sma20": sma20, "sma50": sma50, "rsi": rsi,
            "macd_val": macd_val, "macd_sig": macd_sig,
            "macd_cross_bull": macd_cross_bull,
            "ttm_fired": ttm_fired,
            "ttm_is_sqz": ttm_is_sqz,
            "donchian_break": donchian_break,
            "pe": info.get("trailingPE"), "market_cap": info.get("marketCap"),
            "pct_from_52high": pct_from_high, "w52_high": w52_high,
            "volume": volume,
            "ret_12m": ret_12m,  # Für RS-Rating Post-Processing
        }
    except Exception as e:
        logger.warning(f"Screener-Fehler für {sym}: {type(e).__name__}: {e}")
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
        if filters.get("ttm_squeeze_fired"):
            if not r.get("ttm_fired"):
                continue
        if filters.get("ttm_is_sqz"):
            if not r.get("ttm_is_sqz"):
                continue
        if filters.get("donchian_breakout"):
            if not r.get("donchian_break"):
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
            "Close": r.get("price") or 0.0,
            "SMA20": r.get("sma20") or 0.0,
            "SMA50": r.get("sma50") or 0.0,
            "RSI": r.get("rsi") or 0.0,
            "MACD": r.get("macd_val") or 0.0,
            "Signal": r.get("macd_sig") or 0.0,
            "PE": r.get("pe") or 0.0,
            "MarketCap": r.get("market_cap") or 0.0,
            "Pct52wHigh": r.get("pct_from_52high") or 0.0,
            "Volume": r.get("volume") or 0.0,
            "TTM_Squeeze": r.get("ttm_fired") or False,      
            "TTM_In_Squeeze": r.get("ttm_is_sqz") or False,
            "Donchian_Break": r.get("donchian_break") or False,
            "nan": float("nan"), "inf": float("inf"),
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
    st.markdown("####  Universum")
    bookmarks = load_bookmarks()
    universe_options = list(INDEX_UNIVERSES.keys())
    # Neue vollständige Universen hinzufügen
    # (Nicht mehr nötig, da oben bereits als Full definiert, wir lassen die Logik aber robust)
    
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
        # Generischer Fallback: get_full_universe() für alle registrierten Indizes
        tickers_to_screen = get_full_universe(selected_universe)
        if not tickers_to_screen:
            st.warning(f"Keine Ticker für '{selected_universe}' gefunden.")
            return

    if combine_with_watchlist:
        tickers_to_screen = list(set(tickers_to_screen + list(bookmarks.keys())))
    
    if not tickers_to_screen:
        st.info("Bitte ein Universum wählen oder Ticker eingeben.")
        return
    st.caption(f"{len(tickers_to_screen)} Aktien im Universum geladen")

    # --- Filter ---
    st.markdown("####  Strategie / Filter")
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
            
        st.markdown("**Spezial-Filter (TTM & Donchian)**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.checkbox("TTM Squeeze Fired (Breakout)", help="Squeeze-Ausbruch in den letzten 2 Tagen."):
                active_filters["ttm_squeeze_fired"] = True
        with sc2:
            if st.checkbox("Aktuell im Squeeze", help="Volatilitäts-Kompression (BB innerhalb KC)."):
                active_filters["ttm_is_sqz"] = True
        with sc3:
            if st.checkbox("Donchian Breakout (20T)"):
                active_filters["donchian_breakout"] = True

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
        with st.expander(" Verfügbare Variablen & Syntax", expanded=True):
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
    st.markdown("####  Scan")
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

        # RS-Rating: Percentil-Rang der 12M-Rendite im Universum (Item #9)
        all_ret12m = [r["ret_12m"] for r in raw_results if r and r.get("ret_12m") is not None]
        if all_ret12m:
            ret_arr = np.array(all_ret12m)
            ret_sorted = np.sort(ret_arr)
            for r in raw_results:
                if r and r.get("ret_12m") is not None:
                    r["rs_rating"] = int(np.searchsorted(ret_sorted, r["ret_12m"]) / len(ret_arr) * 99)
                elif r:
                    r["rs_rating"] = None


        # Item #4: Ticker-Liste persistieren
        st.session_state["screener_last_tickers"] = tickers_to_screen

        if filter_mode == "Custom Formula":
            matches, formula_error = apply_custom_formula(raw_results, custom_formula)
            if formula_error:
                st.error(f" {formula_error}")
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
        st.success(f"**{len(matches)} Treffer** von {total_scanned} gescannten Aktien")
        rows = []
        for r in sorted(matches, key=lambda x: x.get("rsi") or 999):
            rsi_v = r.get("rsi")
            rs = r.get("rs_rating")
            rows.append({
                "Ticker": r["ticker"], "Name": r["name"],
                "Kurs": f"{r['price']:.2f}" if r.get("price") else "—",
                "RS-Rating": f"{rs}" if rs is not None else "—",
                "RSI (14)": f"{rsi_v:.1f}" if rsi_v else "—",
                "TTM Squeeze": "Fired!" if r.get("ttm_fired") else ("In Squeeze" if r.get("ttm_is_sqz") else "—"),
                "Donchian Brk": "Signal" if r.get("donchian_break") else "—",
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
        st.markdown("**Analyse & Backtest:**")
        
        # New: Jump to Backtesting button
        if st.button("Ergebnisse im Backtest prüfen", use_container_width=True):
            st.session_state["bt_universe_mode"] = True
            st.session_state["bt_universe_choice"] = "Screener Ergebnisse"
            st.session_state["bt_run_immediately"] = True
            st.switch_page("pages/3_Backtesting.py")
            
        ticker_buttons = [r["ticker"] for r in matches]
        cols = st.columns(min(len(ticker_buttons), 6))
        for i, sym in enumerate(ticker_buttons[:12]):
            with cols[i % 6]:
                if st.button(sym, key=f"screen_jump_{sym}"):
                    st.session_state["jump_ticker"] = sym
                    st.rerun()
