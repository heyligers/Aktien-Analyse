# ============================================================================
# modules/index_utils.py — Hilfsfunktionen zum Abrufen von Index-Bestandteilen
# ============================================================================

import json
import os
import logging
import streamlit as st
from datetime import datetime
from modules.utils import logger

# Pfad zur lokalen JSON-Datei mit den Index-Bestandteilen
CONSTITUENTS_FILE = os.path.join(os.path.dirname(__file__), "index_constituents.json")
CONSTITUENTS_META_FILE = os.path.join(os.path.dirname(__file__), "index_constituents_meta.json")


def _load_from_json(index_key):
    """Lädt Ticker aus der lokalen JSON-Datei."""
    try:
        if os.path.exists(CONSTITUENTS_FILE):
            with open(CONSTITUENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get(index_key, [])
    except Exception as e:
        logger.error(f"Fehler beim Laden von {CONSTITUENTS_FILE}: {e}")
    return []


def get_sp500_tickers():
    """Gibt die S&P 500 Liste zurück (bevorzugt aus lokaler Datei)."""
    tickers = _load_from_json("sp500")
    if not tickers:
        logger.warning("S&P 500 Ticker konnten nicht aus JSON geladen werden.")
    return tickers


def get_nasdaq100_tickers():
    """Gibt die Nasdaq 100 Liste zurück (bevorzugt aus lokaler Datei)."""
    tickers = _load_from_json("nasdaq100")
    if not tickers:
        logger.warning("Nasdaq 100 Ticker konnten nicht aus JSON geladen werden.")
    return tickers


def get_dax_tickers():
    """Gibt die DAX 40 Liste zurück (bevorzugt aus lokaler Datei)."""
    tickers = _load_from_json("dax")
    if not tickers:
        logger.warning("DAX Ticker konnten nicht aus JSON geladen werden.")
    return tickers


def get_full_universe(index_name):
    """
    Gibt die vollständige Ticker-Liste für einen Index zurück.
    Unterstützt mehrere Namens-Varianten (mit/ohne '(Full)'-Suffix).
    """
    _map = {
        "S&P 500":           get_sp500_tickers,
        "S&P 500 (Full)":    get_sp500_tickers,
        "sp500":             get_sp500_tickers,
        "Nasdaq 100":        get_nasdaq100_tickers,
        "Nasdaq 100 (Full)": get_nasdaq100_tickers,
        "nasdaq100":         get_nasdaq100_tickers,
        "DAX 40":            get_dax_tickers,
        "DAX 40 (Full)":     get_dax_tickers,
        "DAX":               get_dax_tickers,
        "dax":               get_dax_tickers,
    }
    fn = _map.get(index_name)
    if fn:
        return fn()
    logger.warning(f"Unbekannter Index-Name: '{index_name}'")
    return []


# ============================================================================
# ITEM #12: AUTO-UPDATE via Wikipedia (quartalsweise)
# ============================================================================

def _scrape_sp500_wikipedia() -> list:
    """Scrapt S&P 500 Bestandteile von Wikipedia."""
    import pandas as pd
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        attrs={"id": "constituents"}, flavor="lxml"
    )
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    tickers = [t.replace(".", "-") for t in df[col].tolist() if isinstance(t, str) and t.strip()]
    return sorted(set(tickers))


def _scrape_nasdaq100_wikipedia() -> list:
    """Scrapt Nasdaq 100 Bestandteile von Wikipedia."""
    import pandas as pd
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        attrs={"id": "constituents"}, flavor="lxml"
    )
    df = tables[0]
    col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else df.columns[1])
    tickers = [t.replace(".", "-") for t in df[col].tolist() if isinstance(t, str) and t.strip()]
    return sorted(set(tickers))


def _scrape_dax_wikipedia() -> list:
    """Scrapt DAX 40 Bestandteile von Wikipedia."""
    import pandas as pd
    tables = pd.read_html("https://en.wikipedia.org/wiki/DAX", flavor="lxml")
    for df in tables:
        if "Ticker symbol" in df.columns or "Symbol" in df.columns:
            col = "Ticker symbol" if "Ticker symbol" in df.columns else "Symbol"
            tickers = [t.strip() for t in df[col].tolist() if isinstance(t, str) and "." in t]
            if len(tickers) >= 30:
                return sorted(set(tickers))
    return []


def _get_meta() -> dict:
    """Liest die Metadaten (Timestamp des letzten Updates)."""
    try:
        if os.path.exists(CONSTITUENTS_META_FILE):
            with open(CONSTITUENTS_META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_meta(meta: dict):
    try:
        with open(CONSTITUENTS_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.error(f"Meta-Datei konnte nicht gespeichert werden: {e}")


def auto_update_constituents_if_stale(max_age_days: int = 90, force: bool = False) -> dict:
    """
    Aktualisiert index_constituents.json falls älter als max_age_days Tage.
    Scrapt Wikipedia für S&P 500, Nasdaq 100 und DAX 40.
    Validiert Ticker-Anzahl mit ±10%-Toleranz.

    Args:
        max_age_days: Maximales Alter der Datei in Tagen (Standard: 90)
        force: Ignoriert das Alter und erzwingt eine Aktualisierung
    Returns:
        dict mit keys: updated (bool), results (dict), errors (list)
    """
    meta = _get_meta()
    last_update_str = meta.get("last_update")
    status = {"updated": False, "results": {}, "errors": []}

    if not force and last_update_str:
        try:
            last_update = datetime.fromisoformat(last_update_str)
            age = (datetime.now() - last_update).days
            if age < max_age_days:
                status["skipped"] = True
                status["age_days"] = age
                return status
        except Exception:
            pass

    # Aktuelle Daten laden
    try:
        with open(CONSTITUENTS_FILE, "r", encoding="utf-8") as f:
            current_data = json.load(f)
    except Exception:
        current_data = {}

    expected_sizes = {"sp500": 500, "nasdaq100": 100, "dax": 40}
    scrapers = {
        "sp500":     _scrape_sp500_wikipedia,
        "nasdaq100": _scrape_nasdaq100_wikipedia,
        "dax":       _scrape_dax_wikipedia,
    }

    new_data = dict(current_data)

    for key, scrape_fn in scrapers.items():
        try:
            tickers = scrape_fn()
            expected = expected_sizes[key]
            tolerance = 0.10  # ±10%
            min_ok = int(expected * (1 - tolerance))
            max_ok = int(expected * (1 + tolerance))

            if min_ok <= len(tickers) <= max_ok:
                old_count = len(current_data.get(key, []))
                new_data[key] = tickers
                status["results"][key] = {
                    "ok": True,
                    "count": len(tickers),
                    "prev_count": old_count,
                    "delta": len(tickers) - old_count
                }
                logger.info(f"Auto-Update {key}: {old_count} -> {len(tickers)} Ticker")
            else:
                status["results"][key] = {
                    "ok": False,
                    "count": len(tickers),
                    "reason": f"Unerwartete Groesse: {len(tickers)} (erwartet {min_ok}-{max_ok})"
                }
                status["errors"].append(f"{key}: Ungueltige Ticker-Anzahl ({len(tickers)})")
        except Exception as e:
            status["errors"].append(f"{key}: {type(e).__name__}: {e}")
            logger.error(f"Auto-Update Fehler fuer {key}: {e}")

    # Nur schreiben wenn mindestens ein Index erfolgreich war
    if any(v.get("ok") for v in status["results"].values()):
        try:
            with open(CONSTITUENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            meta["last_update"] = datetime.now().isoformat()
            meta["last_counts"] = {k: v.get("count") for k, v in status["results"].items()}
            _save_meta(meta)
            status["updated"] = True
        except Exception as e:
            status["errors"].append(f"Schreibfehler: {e}")

    return status


def display_constituents_update_ui():
    """Streamlit-UI fuer manuelles und automatisches Constituent-Update (Item #12)."""
    st.markdown("### Index-Bestandteile aktualisieren")
    st.caption(
        "Scrapt Wikipedia fuer S&P 500, Nasdaq 100 und DAX 40. "
        "Aktualisiert automatisch alle 90 Tage."
    )

    meta = _get_meta()
    last_update = meta.get("last_update", "Noch nie")
    last_counts = meta.get("last_counts", {})

    col1, col2 = st.columns(2)
    col1.metric("Letztes Update", last_update[:10] if last_update != "Noch nie" else "-")
    if last_counts:
        col2.metric(
            "Letzte Ticker-Anzahl",
            f"S&P {last_counts.get('sp500', '?')} * NDX {last_counts.get('nasdaq100', '?')} * DAX {last_counts.get('dax', '?')}"
        )

    if st.button("Jetzt aktualisieren (Wikipedia)", type="primary", key="update_constituents"):
        with st.spinner("Scrappe Wikipedia..."):
            result = auto_update_constituents_if_stale(force=True)

        if result.get("updated"):
            st.success("Datei erfolgreich aktualisiert!")
            for key, r in result["results"].items():
                if r.get("ok"):
                    delta_str = f"{r['delta']:+d}" if r.get("delta") is not None else ""
                    st.markdown(f"OK **{key.upper()}**: {r['prev_count']} -> {r['count']} Ticker {delta_str}")
                else:
                    st.warning(f"Warnung **{key.upper()}**: {r.get('reason', 'Fehler')}")
        else:
            st.info("Keine Aktualisierung durchgefuehrt.")

        if result.get("errors"):
            for err in result["errors"]:
                st.error(err)
