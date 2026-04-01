# ============================================================================
# modules/utils.py — Hilfsfunktionen & globales Setup
# ============================================================================

import os
import ssl
import logging
import requests
import streamlit as st


import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SSL-Fix für Windows
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context

# --- LOGGING ---
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger("aktien_analyse")

# --- Basis-Pfade ---
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKMARK_FILE = os.path.join(_BASE_DIR, "bookmarks.json")
PORTFOLIO_FILE = os.path.join(_BASE_DIR, "portfolio.json")


def fmt_number(v, pct=False, large=False, decimals=2):
    """Zentrale Formatierungsfunktion für Zahlen."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (ValueError, TypeError):
        return str(v)
    if v != v:  # NaN
        return "—"
    if pct:
        return f"{v * 100:.{decimals}f} %".replace(".", ",")
    if large:
        if abs(v) >= 1e12:
            return f"{v/1e12:.{decimals}f}".replace(".", ",") + " Bio."
        elif abs(v) >= 1e9:
            return f"{v/1e9:.{decimals}f}".replace(".", ",") + " Mrd."
        elif abs(v) >= 1e6:
            return f"{v/1e6:.{decimals}f}".replace(".", ",") + " Mio."
        else:
            return f"{v:,.0f}"
    return f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def safe_get(url, **kwargs):
    """HTTP GET mit SSL-Fallback und Logging."""
    kwargs.setdefault('timeout', 8)
    kwargs.setdefault('headers', {'User-Agent': 'Mozilla/5.0'})
    try:
        return requests.get(url, verify=True, **kwargs)
    except requests.exceptions.SSLError:
        logger.warning(f"SSL-Fehler für {url[:60]}, Fallback auf verify=False")
        return requests.get(url, verify=False, **kwargs)


def get_api_key_value():
    """Priorisiert: st.secrets > os.environ > leerer String."""
    try:
        return st.secrets["NEWSAPI_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.environ.get("NEWSAPI_KEY", "")
