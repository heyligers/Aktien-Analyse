# ============================================================================
# modules/ai_gemini.py — Google Gemini KI-Integration
# ============================================================================

import os
import streamlit as st
import requests
from datetime import datetime
from modules.utils import logger

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def get_gemini_key() -> str:
    key = ""
    try:
        key = st.secrets.get("GEMINI_KEY", "") or st.secrets.get("AI_API_KEY", "")
    except (FileNotFoundError, AttributeError):
        pass
    if not key:
        key = os.environ.get("GEMINI_KEY", "") or os.environ.get("AI_API_KEY", "")
    return key


@st.cache_data(ttl=900, show_spinner=False)
def get_gemini_summary(ticker_sym: str, company_name: str,
                       api_key: str, model: str, news_context: str,
                       lang: str = "de") -> dict:
    """Ruft KI-Zusammenfassung via Gemini REST API ab."""
    if lang == "de":
        prompt = (f"Hier sind die aktuellsten Nachrichten zu {company_name} ({ticker_sym}):\n\n"
                  f"{news_context}\n\n---\n"
                  f"Aufgabe: Analysiere diese Nachrichten als erfahrener Finanzanalyst.\n"
                  f"1. Fasse die wichtigsten Entwicklungen zusammen.\n"
                  f"2. Bewerte die Gesamtstimmung (bullisch/neutral/bärisch).\n"
                  f"3. Nenne mögliche Risiken und Chancen.\n"
                  f"4. Gib eine kurze Einschätzung zur weiteren Kursentwicklung.\n"
                  f"Antworte auf Deutsch, strukturiert mit Überschriften.")
    else:
        prompt = (f"Here are the latest news about {company_name} ({ticker_sym}):\n\n"
                  f"{news_context}\n\n---\n"
                  f"Task: Analyze these news as an experienced financial analyst.\n"
                  f"1. Summarize the key developments.\n"
                  f"2. Assess the overall sentiment (bullish/neutral/bearish).\n"
                  f"3. Highlight potential risks and opportunities.\n"
                  f"4. Provide a brief outlook on price development.\n"
                  f"Be structured with headings.")
    try:
        url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": "Du bist ein erfahrener Finanzanalyst. Analysiere die bereitgestellten aktuellen Nachrichten prägnant und faktenbasiert."}]
            },
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}
        }
        try:
            res = requests.post(url, json=payload, timeout=30, verify=True)
        except requests.exceptions.SSLError:
            res = requests.post(url, json=payload, timeout=30, verify=False)
        res.raise_for_status()
        data = res.json()
        candidates = data.get("candidates", [])
        if not candidates:
            error_msg = data.get("error", {}).get("message", "Keine Antwort erhalten.")
            return {"content": "", "error": error_msg}
        content = ""
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            content += part.get("text", "")
        return {"content": content, "error": None}
    except requests.exceptions.HTTPError as e:
        try:
            err_detail = e.response.json().get("error", {}).get("message", e.response.text[:200])
        except (ValueError, AttributeError):
            err_detail = str(e)
        return {"content": "", "error": f"API-Fehler {e.response.status_code}: {err_detail}"}
    except Exception as e:
        return {"content": "", "error": str(e)}


def display_ai_news(ticker_sym: str, company_name: str, lang: str):
    """Rendert den KI-News Tab — lädt zuerst Live-News, dann Gemini-Analyse."""
    st.markdown("#### KI-Analyse (Gemini)")
    st.caption("Analysiert aktuelle News aus yfinance, Google und NewsAPI mit Gemini.")

    gemini_key = get_gemini_key()

    with st.expander("Einstellungen"):
        key_input = st.text_input("Gemini API-Key", value=gemini_key, type="password",
                                  key="gemini_key_input",
                                  help="Google AI API-Key von aistudio.google.com/apikey")
        model_choice = st.selectbox("Modell", [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ], key="gemini_model")
        if key_input:
            gemini_key = key_input

    if not gemini_key:
        st.info("""**Gemini API-Key benötigt**

1. Gehe zu [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Klicke **"API-Schlüssel erstellen"**
3. Key oben eintragen

Oder dauerhaft speichern in `.streamlit/secrets.toml`: `GEMINI_KEY = "dein-key"`""")
        return

    if st.button("KI-Analyse starten", type="primary", key="run_ai"):
        from modules.news_api import get_combined_news
        with st.spinner("Aktuelle News werden geladen…"):
            y_reg = "DE" if lang == "de" else "US"
            n_api_key = ""
            try:
                n_api_key = st.secrets.get("NEWSAPI_KEY", "") or os.environ.get("NEWSAPI_KEY", "")
            except (FileNotFoundError, AttributeError):
                n_api_key = os.environ.get("NEWSAPI_KEY", "")
            live_news = get_combined_news(
                ticker_sym, company_name, n_api_key,
                y_reg, f"{lang}-{y_reg}", lang, datetime.today().date()
            )

        if not live_news:
            st.warning("Keine aktuellen News gefunden — KI-Analyse nicht möglich.")
            return

        news_lines = []
        for i, n in enumerate(live_news[:15], 1):
            line = f"{i}. [{n.get('publisher', '')}] {n.get('title', '')}"
            news_lines.append(line)
        news_context = "\n".join(news_lines)

        st.info(f"**{len(live_news)} News geladen** — Gemini analysiert…")

        with st.spinner(f"Gemini ({model_choice}) analysiert {len(news_lines)} Artikel…"):
            result = get_gemini_summary(
                ticker_sym, company_name, gemini_key,
                model_choice, news_context, lang
            )
        if result["error"]:
            st.error(f"Fehler: {result['error']}")
        elif result["content"]:
            st.markdown(result["content"])
        else:
            st.warning("Keine Antwort erhalten.")
