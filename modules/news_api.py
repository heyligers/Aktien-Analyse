# ============================================================================
# modules/news_api.py — News-Quellen: yfinance, Google RSS, NewsAPI
# ============================================================================

import streamlit as st
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_tz, mktime_tz
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib

from modules.utils import safe_get, logger


# ============================================================================
# SENTIMENT
# ============================================================================

def sentiment_score(title: str, lang: str) -> tuple:
    """Berechnet Sentiment. Gibt (Emoji, score) zurück."""
    if lang == "de":
        positive = ["steigt", "plus", "gewinn", "wachstum", "rekord", "starkes",
                    "erhöht", "besser", "übertrifft", "ausbau", "rally", "kauf",
                    "aufwärts", "positiv", "erfolgreich", "stark", "steigend"]
        negative = ["fällt", "minus", "verlust", "krise", "schwach", "absturz",
                    "warnung", "risiko", "insolvenz", "entlassungen", "rückgang",
                    "abwärts", "negativ", "enttäuschend", "einbruch", "sinkt"]
        t = title.lower()
        pos = sum(1 for w in positive if w in t)
        neg = sum(1 for w in negative if w in t)
        score = (pos - neg) / max(pos + neg, 1) if (pos + neg) > 0 else 0.0
    else:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(title)['compound']
        except Exception:
            score = 0.0
    mood = "▲" if score > 0.05 else "▼" if score < -0.05 else "○"
    return mood, score


def display_sentiment_summary(news_data, lang):
    if not news_data:
        return
    scores = [sentiment_score(n['title'], lang)[1] for n in news_data]
    avg = sum(scores) / len(scores)
    pos_count = sum(1 for s in scores if s > 0.05)
    neg_count = sum(1 for s in scores if s < -0.05)
    neu_count = len(scores) - pos_count - neg_count
    total = len(scores)
    pos_pct = int(pos_count / total * 100) if total else 0
    neg_pct = int(neg_count / total * 100) if total else 0
    emoji = "▲" if avg > 0.05 else "▼" if avg < -0.05 else "○"
    label = "Positiv" if avg > 0.05 else "Negativ" if avg < -0.05 else "Neutral"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 14px;margin-bottom:12px;">
      <div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">News-Stimmung</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:1.5rem;">{emoji}</span>
        <div style="flex:1;">
          <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;">
            <div style="width:{pos_pct}%;background:#26a69a;"></div>
            <div style="width:{100-pos_pct-neg_pct}%;background:#555;"></div>
            <div style="width:{neg_pct}%;background:#ef5350;"></div>
          </div>
        </div>
        <span style="font-size:0.85rem;color:#ddd;">{label}</span>
      </div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">
        ▲ {pos_count} positiv · ○ {neu_count} neutral · ▼ {neg_count} negativ · {total} Artikel
      </div>
    </div>""", unsafe_allow_html=True)


def deduplicate_news(news_list: list, threshold: float = 0.72) -> list:
    seen_urls = set()
    seen_titles = []
    unique = []
    for n in news_list:
        url = n['link'] if n['link'] != "#" else ""
        if url and url in seen_urls:
            continue
        title_norm = n['title'].lower().strip()
        is_dup = any(
            difflib.SequenceMatcher(None, title_norm, t).ratio() >= threshold
            for t in seen_titles
        )
        if is_dup:
            continue
        if url:
            seen_urls.add(url)
        seen_titles.append(title_norm)
        unique.append(n)
    return unique


# ============================================================================
# NEWS-QUELLEN
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_yfinance_news(ticker_sym):
    import yfinance as yf
    news_list = []
    try:
        items = yf.Ticker(ticker_sym).news or []
        for art in items[:10]:
            content = art.get("content", {})
            if not content:
                title = art.get("title", "")
                link = art.get("link", "#")
                pub = art.get("providerPublishTime", None)
                provider = art.get("publisher", "Yahoo Finance")
                thumb = art.get("thumbnail", {})
                resolutions = (thumb or {}).get("resolutions", [])
                image_url = resolutions[0].get("url", "") if resolutions else ""
                try:
                    dt = datetime.fromtimestamp(pub) if pub else datetime.now()
                except (ValueError, OSError, TypeError):
                    dt = datetime.now()
            else:
                title = content.get("title", "")
                link = content.get("canonicalUrl", {}).get("url", "#")
                pub = content.get("pubDate", "")
                provider = content.get("provider", {}).get("displayName", "Yahoo Finance")
                thumb = content.get("thumbnail", {}) or {}
                resolutions = thumb.get("resolutions", [])
                image_url = resolutions[0].get("url", "") if resolutions else ""
                try:
                    dt = datetime.strptime(pub[:19], "%Y-%m-%dT%H:%M:%S")
                except (ValueError, TypeError):
                    dt = datetime.now()
            if not image_url:
                image_url = f"https://ui-avatars.com/api/?name={ticker_sym[:2]}&background=0D6EFD&color=fff&size=150"
            if title:
                news_list.append({'title': title, 'link': link, 'publisher': provider,
                                  'dt': dt, 'image': image_url, 'source_tag': 'yfinance'})
    except (requests.RequestException, KeyError, TypeError) as e:
        logger.warning(f"yfinance News Fehler: {e}")
    return news_list


@st.cache_data(ttl=300, show_spinner=False)
def get_google_news_rss(query, lang="en", region="US", target_date=None):
    ceid = f"{region}:{lang}"
    if target_date and target_date < datetime.today().date():
        date_after = target_date.strftime("%Y-%m-%d")
        date_before = (target_date + timedelta(days=2)).strftime("%Y-%m-%d")
        q_full = f"{query} after:{date_after} before:{date_before}"
    else:
        q_full = query
    encoded = urllib.parse.quote(q_full)
    url = (f"https://news.google.com/rss/search?q={encoded}"
           f"&hl={lang}-{region}&gl={region}&ceid={ceid}")
    news_list = []
    try:
        res = safe_get(url)
        res.raise_for_status()
        root = ET.fromstring(res.content)
        for item in root.findall('.//item')[:10]:
            title = item.findtext('title', '')
            link = item.findtext('link', '#')
            date_str = item.findtext('pubDate', '')
            src_el = item.find('source')
            source = (src_el.text if src_el is not None and src_el.text else "Google News")
            try:
                dt = datetime.fromtimestamp(mktime_tz(parsedate_tz(date_str)))
            except (ValueError, TypeError, OSError):
                dt = datetime.now()
            if title:
                news_list.append({
                    'title': title, 'link': link, 'publisher': source, 'dt': dt,
                    'image': "https://ui-avatars.com/api/?name=GN&background=4285F4&color=fff&size=150",
                    'source_tag': 'google'
                })
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning(f"Google RSS ({query[:30]}): {e}")
    return news_list


def _fetch_rss_source(url, label) -> list:
    items = []
    try:
        res = safe_get(url, timeout=5)
        if res.status_code != 200:
            return items
        root = ET.fromstring(res.content)
        for item in root.findall('.//item')[:8]:
            title = item.findtext('title', '')
            link = item.findtext('link', '#')
            date_str = item.findtext('pubDate', '')
            try:
                dt = datetime.fromtimestamp(mktime_tz(parsedate_tz(date_str)))
            except (ValueError, TypeError, OSError):
                dt = datetime.now()
            bg = "003399" if "DE" in label or "de" in label.lower() else "1a1a2e"
            image_url = f"https://ui-avatars.com/api/?name={urllib.parse.quote(label[:2])}&background={bg}&color=fff&size=150"
            if title:
                items.append({'title': title, 'link': link, 'publisher': label,
                              'dt': dt, 'image': image_url, 'source_tag': 'rss'})
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning(f"RSS Fehler ({label}): {e}")
    return items


@st.cache_data(ttl=300, show_spinner=False)
def get_de_market_rss():
    sources = [
        ("https://www.finanzen.net/rss/news", "finanzen.net"),
        ("https://www.boerse-frankfurt.de/rss/news", "Börse Frankfurt"),
        ("https://feeds.reuters.com/reuters/DETopNews", "Reuters DE"),
        ("https://www.handelsblatt.com/rss", "Handelsblatt"),
        ("https://www.tagesschau.de/wirtschaft/index~rss2.xml", "Tagesschau"),
    ]
    news_list = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_fetch_rss_source, url, label): label for url, label in sources}
        for fut in as_completed(futures):
            news_list.extend(fut.result())
    news_list.sort(key=lambda x: x['dt'], reverse=True)
    return news_list[:10]


@st.cache_data(ttl=300, show_spinner=False)
def get_us_market_rss():
    sources = [
        ("https://feeds.content.dowjones.io/public/rss/mw_marketpulse", "MarketWatch"),
        ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "WSJ Markets"),
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters Business"),
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC Markets"),
    ]
    news_list = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_rss_source, url, label): label for url, label in sources}
        for fut in as_completed(futures):
            news_list.extend(fut.result())
    news_list.sort(key=lambda x: x['dt'], reverse=True)
    return news_list[:10]


@st.cache_data(ttl=300, show_spinner=False)
def get_newsapi(query, api_key, lang, target_date):
    if not api_key:
        return []
    date_str = target_date.strftime("%Y-%m-%d")
    params = {
        "q": query, "sortBy": "relevancy", "language": lang,
        "pageSize": 10, "apiKey": api_key, "from": date_str, "to": date_str
    }
    ignore_list = ["slashdot", "reddit", "biztoc", "yahoo entertainment"]
    news_list = []
    try:
        res = safe_get("https://newsapi.org/v2/everything", params=params, timeout=5).json()
        if res.get("status") == "ok":
            for art in res.get("articles", []):
                title = art.get("title", "")
                if not title or title == "[Removed]":
                    continue
                publisher = art.get("source", {}).get("name", "Unbekannt")
                if any(b in publisher.lower() for b in ignore_list):
                    continue
                link = art.get("url", "#")
                image_url = art.get("urlToImage") or \
                    "https://ui-avatars.com/api/?name=NW&background=random&color=fff&size=150"
                pub_time = art.get("publishedAt", "")
                try:
                    dt = datetime.strptime(pub_time, "%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, TypeError):
                    dt = datetime.now()
                news_list.append({'title': title, 'link': link, 'publisher': publisher,
                                  'dt': dt, 'image': image_url, 'source_tag': 'newsapi'})
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning(f"NewsAPI Fehler: {e}")
    return news_list[:6]


def get_combined_news(ticker_sym, company_name, api_key, y_reg, y_lang, n_lang, target_date):
    today = datetime.today().date()
    short_name = company_name.split()[0] if company_name else ticker_sym
    if n_lang == "de":
        g_queries = [f"{ticker_sym} Aktie", f"{short_name} Börse Kurs"]
        api_query = f'"{company_name}" AND (Aktie OR Quartalszahlen OR Dividende OR Gewinn)'
    else:
        g_queries = [f"{ticker_sym} stock", f"{short_name} earnings shares"]
        api_query = f'"{company_name}" AND (stock OR earnings OR shares OR dividend)'

    results = {"yf": [], "google": [], "newsapi": []}
    def _yf():
        return get_yfinance_news(ticker_sym) if target_date >= today else []
    def _google(q):
        return get_google_news_rss(q, lang=n_lang, region=y_reg, target_date=target_date)
    def _napi():
        return get_newsapi(api_query, api_key, n_lang, target_date)

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_yf = ex.submit(_yf)
        fut_g = [ex.submit(_google, q) for q in g_queries]
        fut_napi = ex.submit(_napi)
        results["yf"] = fut_yf.result()
        results["google"] = [item for f in fut_g for item in f.result()]
        results["newsapi"] = fut_napi.result()

    combined = results["yf"] + results["google"] + results["newsapi"]
    combined.sort(key=lambda x: x['dt'], reverse=True)
    return deduplicate_news(combined)[:10]


def get_combined_market_news(market_proxy, market_query, api_key, y_reg, y_lang, n_lang, target_date):
    today = datetime.today().date()
    is_today = target_date >= today
    if market_proxy == "^GDAXI":
        g_query = "DAX Börse Frankfurt Markt"
    elif market_proxy == "^FTSE":
        g_query = "FTSE London stock market"
    elif market_proxy == "^N225":
        g_query = "Nikkei Japan stock market"
    else:
        g_query = "S&P 500 Wall Street market"

    def _rss():
        if market_proxy == "^GDAXI" and is_today:
            return get_de_market_rss()
        elif market_proxy == "^GSPC" and is_today:
            return get_us_market_rss()
        return []
    def _google():
        return get_google_news_rss(g_query, lang=n_lang, region=y_reg, target_date=target_date)
    def _napi():
        return get_newsapi(market_query, api_key, n_lang, target_date)

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_rss = ex.submit(_rss)
        fut_g = ex.submit(_google)
        fut_napi = ex.submit(_napi)
        rss_news = fut_rss.result()
        google_news = fut_g.result()
        n_api = fut_napi.result()

    combined = rss_news + google_news + n_api
    combined.sort(key=lambda x: x['dt'], reverse=True)
    return deduplicate_news(combined)[:10]


def display_news_aesthetic(news_data, lang):
    if not news_data:
        st.warning("Momentan keine News gefunden.")
        return
    display_sentiment_summary(news_data, lang)
    for n in news_data:
        mood, _ = sentiment_score(n['title'], lang)
        formatted_date = n['dt'].strftime('%d.%m.%Y %H:%M')
        with st.container():
            col_img, col_text = st.columns([1, 3])
            with col_img:
                st.image(n['image'], use_container_width=True)
            with col_text:
                st.markdown(f"**[{mood} {n['title']}]({n['link']})**")
                st.caption(f"{formatted_date} · {n['publisher']}")
        st.divider()
