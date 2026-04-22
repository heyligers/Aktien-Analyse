# ============================================================================
# modules/data_api.py — Yahoo Finance, Macro, Risk-Metriken
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import streamlit as st
import types
import urllib.parse
from datetime import datetime, timedelta

from modules.utils import fmt_number, logger, safe_get



@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker(query):
    if not query:
        return "AAPL"
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=1"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        return res['quotes'][0]['symbol'] if res.get('quotes') else query
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        logger.warning(f"Ticker-Suche fehlgeschlagen für '{query}': {e}")
        return query


@st.cache_data(ttl=300, show_spinner="Daten werden geladen...")
def load_data(ticker, start, end, interval):
    end_yf = end + timedelta(days=1)
    df = yf.download(ticker, start=start, end=end_yf, interval=interval, auto_adjust=False)
    if not df.empty:
        df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def get_full_ticker_data(ticker_sym: str) -> dict:
    """Zentraler Cache: Alle Ticker-Daten in einem API-Aufruf."""
    result = {
        "info": {}, "fast_info": {"last_price": None, "currency": ""},
        "quarterly_financials": None, "calendar": None,
        "upgrades_downgrades": None, "dividends": None
    }
    try:
        t = yf.Ticker(ticker_sym)
        try:
            result["info"] = t.info
        except Exception:
            pass
        try:
            fi = t.fast_info
            result["fast_info"]["last_price"] = getattr(fi, "last_price", None)
            result["fast_info"]["currency"] = getattr(fi, "currency", "")
        except Exception:
            pass
        try:
            qf = t.quarterly_financials
            if qf is not None and not qf.empty:
                result["quarterly_financials"] = qf
        except Exception:
            pass
        try:
            result["calendar"] = t.calendar
        except Exception:
            pass
        try:
            ud = t.upgrades_downgrades
            if ud is not None and not ud.empty:
                result["upgrades_downgrades"] = ud
        except Exception:
            pass
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                result["dividends"] = divs.tail(20)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten für {ticker_sym}: {e}")
    return result


def get_ticker_info(ticker_sym: str) -> dict:
    data = get_full_ticker_data(ticker_sym)
    info = data["info"]
    fi = data["fast_info"]
    return {
        "name": info.get("shortName", ticker_sym).split(',')[0],
        "country": info.get("country", ""),
        "currency": fi.get("currency") or info.get("currency", ""),
    }


def get_fundamentals(ticker_sym: str) -> dict:
    data = get_full_ticker_data(ticker_sym)
    info = data["info"]
    result = {
        "kpis": {}, "quarterly_financials": None,
        "analyst": {}, "calendar": None, "dividends": None,
        "upgrades_downgrades": None
    }
    try:
        kpis = {}
        # Versuche zuerst standard info kpis
        if info:
            kpis = {
                "KGV (TTM)": fmt_number(info.get("trailingPE")),
                "KGV (Forward)": fmt_number(info.get("forwardPE")),
                "KBV": fmt_number(info.get("priceToBook")),
                "EV/EBITDA": fmt_number(info.get("enterpriseToEbitda")),
                "Marktkapitalisierung": fmt_number(info.get("marketCap"), large=True),
                "Umsatz (TTM)": fmt_number(info.get("totalRevenue"), large=True),
                "Gewinnmarge": fmt_number(info.get("profitMargins"), pct=True),
                "ROE": fmt_number(info.get("returnOnEquity"), pct=True),
                "Dividendenrendite": (f"{float(info['dividendYield']):.2f} %".replace(".", ",")
                                      if info.get("dividendYield") is not None else "—"),
                "Beta": fmt_number(info.get("beta")),
                "52W Hoch": fmt_number(info.get("fiftyTwoWeekHigh")),
                "52W Tief": fmt_number(info.get("fiftyTwoWeekLow")),
            }
        
        # Fallback falls info leer ist (Streamlit Cloud Block) -> Nutze fast_info
        if not info or not kpis.get("Marktkapitalisierung") or kpis.get("Marktkapitalisierung") == "—":
            try:
                t = yf.Ticker(ticker_sym)
                fi = t.fast_info
                kpis["Marktkapitalisierung"] = fmt_number(getattr(fi, "market_cap", None), large=True)
                kpis["52W Hoch"] = fmt_number(getattr(fi, "year_high", None))
                kpis["52W Tief"] = fmt_number(getattr(fi, "year_low", None))
                kpis["Letzter Preis"] = fmt_number(getattr(fi, "last_price", None))
                # Fülle Rest mit Strichen, da Yahoo API info blockiert
                for key in ["KGV (TTM)", "KGV (Forward)", "KBV", "EV/EBITDA", "Umsatz (TTM)", "Gewinnmarge", "ROE", "Dividendenrendite", "Beta"]:
                    if key not in kpis:
                        kpis[key] = "—"
            except Exception:
                pass
                
        result["kpis"] = kpis
    except (TypeError, ValueError) as e:
        logger.warning(f"KPI-Formatierung fehlgeschlagen: {e}")

    qf = data["quarterly_financials"]
    if qf is not None:
        try:
            rows_q = {}
            for row_name in ["Total Revenue", "Net Income"]:
                if row_name in qf.index:
                    rows_q[row_name] = qf.loc[row_name]
            if rows_q:
                result["quarterly_financials"] = pd.DataFrame(rows_q).T
        except (KeyError, ValueError):
            pass

    try:
        result["analyst"] = {
            "Kursziel (Ø)": fmt_number(info.get("targetMeanPrice")),
            "Kursziel Hoch": fmt_number(info.get("targetHighPrice")),
            "Kursziel Tief": fmt_number(info.get("targetLowPrice")),
            "Empfehlung": info.get("recommendationKey", "—").replace("_", " ").title(),
            "Analysten": str(info.get("numberOfAnalystOpinions", "—")),
            "Score": info.get("recommendationMean"),
        }
    except (TypeError, ValueError):
        pass

    ud = data["upgrades_downgrades"]
    if ud is not None:
        try:
            ud = ud.sort_index(ascending=False)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            ud_recent = ud[ud.index >= cutoff].head(6)
            if ud_recent.empty:
                ud_recent = ud.head(5)
            result["upgrades_downgrades"] = ud_recent.reset_index()
        except Exception:
            pass

    cal = data["calendar"]
    if cal is not None:
        try:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date", None)
                if ed:
                    result["calendar"] = str(ed[0])[:10] if isinstance(ed, list) else str(ed)[:10]
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                result["calendar"] = str(cal.iloc[0, 0])[:10]
        except (IndexError, TypeError):
            pass

    result["dividends"] = data["dividends"]
    return result


@st.cache_data(ttl=900, show_spinner=False)
def get_options_data(ticker_sym: str) -> dict:
    result = {"iv": None, "pcr": None, "expiry": None, "error": None}
    try:
        t = yf.Ticker(ticker_sym)
        exps = t.options
        if not exps:
            result["error"] = "Keine Optionsdaten verfügbar (kein US-Listing?)"
            return result
        exp = exps[0]
        result["expiry"] = exp
        chain = t.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        total_call_oi = calls["openInterest"].sum()
        total_put_oi = puts["openInterest"].sum()
        if total_call_oi > 0:
            result["pcr"] = round(total_put_oi / total_call_oi, 3)
        current_price = t.fast_info.last_price
        if current_price:
            calls_copy = calls.copy()
            calls_copy["dist"] = abs(calls_copy["strike"] - current_price)
            atm_call = calls_copy.loc[calls_copy["dist"].idxmin()]
            result["iv"] = round(float(atm_call.get("impliedVolatility", 0)) * 100, 2)
    except Exception as e:
        result["error"] = str(e)
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def get_macro_context(ticker_sym: str, market_proxy: str) -> dict:
    result = {"corr_market": None, "corr_sector": None,
              "sector_etf": None, "sector_perf": None,
              "market_perf": None, "sector_name": None}
    try:
        data = get_full_ticker_data(ticker_sym)
        info = data["info"]
        sector_etf_map = {
            "Technology": "XLK", "Financial Services": "XLF",
            "Healthcare": "XLV", "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP", "Industrials": "XLI",
            "Energy": "XLE", "Utilities": "XLU",
            "Real Estate": "XLRE", "Basic Materials": "XLB",
            "Communication Services": "XLC",
        }
        sector = info.get("sector", "")
        sector_etf = sector_etf_map.get(sector)
        result["sector_name"] = sector
        result["sector_etf"] = sector_etf

        end = pd.Timestamp.now(tz="Europe/Berlin").date()
        start = end - timedelta(days=365)
        df_t = yf.download(ticker_sym, start=start, end=end, interval="1d",
                           auto_adjust=True, progress=False)
        if df_t.empty:
            return result
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
        ret_t = df_t["Close"].pct_change().dropna()

        df_m = yf.download(market_proxy, start=start, end=end, interval="1d",
                           auto_adjust=True, progress=False)
        if isinstance(df_m.columns, pd.MultiIndex):
            df_m.columns = df_m.columns.get_level_values(0)
        if not df_m.empty:
            ret_m = df_m["Close"].pct_change().dropna()
            aligned = pd.concat([ret_t, ret_m], axis=1).dropna()
            if len(aligned) > 20:
                result["corr_market"] = round(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]), 3)
            result["market_perf"] = round(
                (float(df_m["Close"].iloc[-1]) / float(df_m["Close"].iloc[0]) - 1) * 100, 2)

        if sector_etf:
            df_s = yf.download(sector_etf, start=start, end=end, interval="1d",
                               auto_adjust=True, progress=False)
            if isinstance(df_s.columns, pd.MultiIndex):
                df_s.columns = df_s.columns.get_level_values(0)
            if not df_s.empty:
                ret_s = df_s["Close"].pct_change().dropna()
                aligned_s = pd.concat([ret_t, ret_s], axis=1).dropna()
                if len(aligned_s) > 20:
                    result["corr_sector"] = round(
                        aligned_s.iloc[:, 0].corr(aligned_s.iloc[:, 1]), 3)
                result["sector_perf"] = round(
                    (float(df_s["Close"].iloc[-1]) / float(df_s["Close"].iloc[0]) - 1) * 100, 2)
    except Exception as e:
        logger.warning(f"Makro-Kontext Fehler: {e}")
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def calculate_risk_metrics(tickers: list) -> dict:
    result = {}
    end = pd.Timestamp.now(tz="Europe/Berlin").date()
    start = end - timedelta(days=365)
    for sym in tickers:
        try:
            df = yf.download(sym, start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty or len(df) < 30:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].squeeze()
            returns = close.pct_change().dropna()
            if len(returns) < 20:
                continue
            mu = float(returns.mean()) * 252
            sigma = float(returns.std()) * np.sqrt(252)
            rf = 0.04
            sharpe = (mu - rf) / sigma if sigma > 0 else 0
            downside = returns[returns < 0].std() * np.sqrt(252)
            sortino = (mu - rf) / downside if downside > 0 else 0
            cum = (1 + returns).cumprod()
            peak = cum.cummax()
            drawdown = ((cum - peak) / peak)
            max_dd = float(drawdown.min()) * 100
            var_95 = float(np.percentile(returns, 5)) * 100
            result[sym] = {
                "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
                "max_drawdown": round(max_dd, 2), "var_95": round(var_95, 2),
                "ann_return": round(mu * 100, 2), "ann_vol": round(sigma * 100, 2),
            }
        except Exception as e:
            logger.warning(f"Risk-Metrik Fehler für {sym}: {e}")
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def calculate_correlation_matrix(tickers: list) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="Europe/Berlin").date()
    start = end - timedelta(days=365)
    all_returns = {}
    for sym in tickers:
        try:
            df = yf.download(sym, start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            all_returns[sym] = df["Close"].squeeze().pct_change().dropna()
        except Exception:
            pass
    if len(all_returns) < 2:
        return pd.DataFrame()
    ret_df = pd.DataFrame(all_returns).dropna()
    return ret_df.corr().round(3)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_bulk_prices(tickers: tuple) -> pd.DataFrame:
    """Interner Cache für Bulk-Preisdaten."""
    if not tickers:
        return pd.DataFrame()
    try:
        # Bei vielen Tickern kann yfinance manchmal hängen, wir begrenzen die Liste zur Sicherheit
        df = yf.download(list(tickers), period="7d", interval="1d", 
                         group_by="ticker", auto_adjust=True, progress=False)
        return df
    except Exception as e:
        logger.error(f"Bulk-Download fehlgeschlagen: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def _get_ticker_meta_heatmap_cached(sym: str) -> dict | None:
    """Extrem langlebiger Cache für Sektor/Name (ändert sich selten)."""
    try:
        t = yf.Ticker(sym)
        info = t.info
        return {
            "name": info.get("shortName", sym)[:15],
            "sector": info.get("sector", "Sonstiges"),
            "mcap": info.get("marketCap", 0)
        }
    except Exception:
        return None

def get_heatmap_data(universe_name: str, tickers: list, progress_cb=None) -> list:
    """
    Haupt-Koordinator (NICHT GECACHED), damit progress_cb sicher aufgerufen werden kann.
    Nutzt interne Caches für Preise und Metadaten.
    """
    if not tickers:
        return []
        
    if progress_cb: progress_cb(5, "Bulk-Kursdaten werden geladen...")
    df_bulk = _get_bulk_prices(tuple(tickers)) # Tuple für Hash-Fähigkeit
    
    if df_bulk.empty:
        # Fallback: Wenn Bulk-Download komplett fehlschlägt, trotzdem Einzelabrufe versuchen
        logger.warning("Bulk-Download leer — versuche Einzelabrufe für Heatmap")
        if progress_cb: progress_cb(10, "Bulk fehlgeschlagen, lade einzeln...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    
    def _process_single(sym):
        try:
            # 1. Preisdaten aus Bulk-DF
            t_df = None
            if isinstance(df_bulk.columns, pd.MultiIndex):
                if sym in df_bulk.columns.levels[0]:
                    t_df = df_bulk[sym]
            else:
                # Falls nur ein Ticker angefragt wurde (selten bei Heatmap)
                if not df_bulk.empty:
                    t_df = df_bulk
            
            # FALLBACK: Falls Ticker im Bulk-Download fehlt, einzeln nachladen
            if t_df is None or t_df["Close"].dropna().empty:
                try:
                    t_df = yf.download(sym, period="7d", interval="1d", 
                                       auto_adjust=True, progress=False)
                except Exception:
                    return None
            
            close_s = t_df["Close"].ffill().dropna()
            if len(close_s) < 2:
                return None
            
            price = float(close_s.iloc[-1])
            prev = float(close_s.iloc[-2])
            change = (price / prev - 1) * 100
            
            # 2. Metadaten aus Cache (oder Fetch)
            meta = _get_ticker_meta_heatmap_cached(sym)
            if not meta:
                # Letzter Versuch: Einzel-Meta-Fetch (wird im Cache für 24h gespeichert)
                meta = {"name": sym, "sector": "Unbekannt", "mcap": 0}
                
            return {
                "ticker": sym, 
                "name": meta["name"], 
                "sector": meta["sector"],
                "change": round(change, 2), 
                "mcap": meta["mcap"], 
                "price": price
            }
        except Exception as e:
            logger.debug(f"Heatmap-Einzelverarbeitung Fehler für {sym}: {e}")
            return None

    total = len(tickers)
    processed = 0
    # Wir reduzieren max_workers etwas, um Rate-Limits bei info-Calls zu vermeiden
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(_process_single, s) for s in tickers]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
            processed += 1
            if progress_cb and processed % 5 == 0:
                p_val = min(15 + int((processed / total) * 85), 100)
                progress_cb(p_val, f"Verarbeite: {processed}/{total}")
    
    if progress_cb: progress_cb(100, "Fertig!")
    return results


@st.cache_data(ttl=1800, show_spinner=False)
def get_insider_data(ticker_sym: str) -> list:
    import re
    transactions = []
    # OpenInsider unterstützt primär US-Aktien ohne Suffix (z.B. AAPL statt AAPL.DE)
    ticker_clean = ticker_sym.split(".")[0].upper().strip()
    try:
        url = f"http://openinsider.com/screener?s={ticker_clean}&fd=90&cnt=20"
        res = safe_get(url, timeout=8)
        if res.status_code != 200:
            return transactions
        table_match = re.search(r'<table[^>]*class="tinytable"[^>]*>(.*?)</table>',
                                res.text, re.DOTALL)
        if not table_match:
            return transactions
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_match.group(1), re.DOTALL)
        for row in rows[1:21]:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) >= 12:
                def clean(s):
                    return re.sub(r'<[^>]+>', '', s).strip()
                filing_date = clean(cells[1])
                trade_date = clean(cells[2])
                insider_name = clean(cells[4])
                title_role = clean(cells[5])
                trade_type = clean(cells[6])
                price = clean(cells[8])
                qty = clean(cells[9])
                value = clean(cells[11])
                is_buy = "Purchase" in trade_type or "Buy" in trade_type
                transactions.append({
                    "date": trade_date, "name": insider_name[:25],
                    "role": title_role[:20], "type": trade_type[:15],
                    "price": price, "qty": qty, "value": value,
                    "is_buy": is_buy
                })
    except Exception as e:
        logger.warning(f"Insider-Daten Fehler für {ticker_sym}: {e}")
    return transactions


@st.cache_data(ttl=600, show_spinner=False)
def get_reddit_posts(ticker_sym: str, company_name: str = None) -> list:
    posts = []
    subreddits = ["wallstreetbets", "stocks", "investing"]
    headers = {'User-Agent': 'Mozilla/5.0 (Stock Screener Bot v1.0)'}
    
    # Query erweitern: Ticker oder Name
    if company_name:
        q = f'({ticker_sym} OR "{company_name}")'
    else:
        q = ticker_sym
        
    for sub in subreddits:
        try:
            url = (f"https://www.reddit.com/r/{sub}/search.json"
                   f"?q={urllib.parse.quote(q)}&sort=new&limit=5&restrict_sr=1")
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code != 200:
                continue
            data = res.json()
            from datetime import datetime
            for child in data.get("data", {}).get("children", []):
                p = child.get("data", {})
                title = p.get("title", "")
                if not title:
                    continue
                score = p.get("score", 0)
                num_comments = p.get("num_comments", 0)
                created = p.get("created_utc", 0)
                permalink = p.get("permalink", "")
                try:
                    dt = datetime.fromtimestamp(created)
                except (OSError, ValueError):
                    dt = datetime.now()
                posts.append({
                    "title": title, "score": score, "comments": num_comments,
                    "subreddit": sub, "dt": dt,
                    "url": f"https://reddit.com{permalink}" if permalink else "#",
                })
        except Exception as e:
            logger.warning(f"Reddit Fehler für r/{sub}: {e}")
    posts.sort(key=lambda x: x['dt'], reverse=True)
    return posts[:15]


@st.cache_data(ttl=1800, show_spinner=False)
def get_economic_calendar() -> list:
    from datetime import datetime, timedelta
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_tz, mktime_tz

    events = []
    static_events = [
        {"date": "2026-04-29", "event": "Fed Zinsentscheid", "region": "US", "importance": "!!!"},
        {"date": "2026-04-30", "event": "EZB Zinsentscheid", "region": "EU", "importance": "!!!"},
        {"date": "2026-06-11", "event": "EZB Zinsentscheid", "region": "EU", "importance": "!!!"},
        {"date": "2026-06-17", "event": "Fed Zinsentscheid", "region": "US", "importance": "!!!"},
        {"date": "2026-07-23", "event": "EZB Zinsentscheid", "region": "EU", "importance": "!!!"},
        {"date": "2026-07-29", "event": "Fed Zinsentscheid", "region": "US", "importance": "!!!"},
        {"date": "2026-09-10", "event": "EZB Zinsentscheid", "region": "EU", "importance": "!!!"},
        {"date": "2026-09-16", "event": "Fed Zinsentscheid", "region": "US", "importance": "!!!"},
        {"date": "2026-10-28", "event": "Fed Zinsentscheid", "region": "US", "importance": "!!!"},
        {"date": "2026-10-29", "event": "EZB Zinsentscheid", "region": "EU", "importance": "!!!"},
        {"date": "2026-12-09", "event": "Fed Zinsentscheid", "region": "US", "importance": "!!!"},
        {"date": "2026-12-17", "event": "EZB Zinsentscheid", "region": "EU", "importance": "!!!"},
    ]
    today = pd.Timestamp.now(tz="Europe/Berlin").date()
    for ev in static_events:
        d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        if d >= today - timedelta(days=14):
            ev_copy = {**ev, "days_away": (d - today).days}
            events.append(ev_copy)

    try:
        res = safe_get("https://www.ecb.europa.eu/rss/press.html", timeout=5)
        if res.status_code == 200:
            root = ET.fromstring(res.content)
            for item in root.findall('.//item')[:5]:
                title = item.findtext('title', '')
                date_str = item.findtext('pubDate', '')
                try:
                    dt = datetime.fromtimestamp(mktime_tz(parsedate_tz(date_str))).date()
                except (ValueError, TypeError, OSError):
                    dt = today
                events.append({
                    "date": dt.strftime("%Y-%m-%d"), "event": f"EZB: {title[:60]}",
                    "region": "EU", "importance": "!!",
                    "days_away": (dt - today).days
                })
    except Exception:
        pass

    for month_offset in range(0, 4):
        # Korrekte Monatsarithmetik ohne timedelta(days=32)-Trick
        target_month = today.month + month_offset
        target_year = today.year + (target_month - 1) // 12
        target_month = ((target_month - 1) % 12) + 1
        # Ersten Tag des Zielmonats bestimmen
        d = today.replace(year=target_year, month=target_month, day=1)
        # Ersten Freitag des Monats finden
        while d.weekday() != 4:  # 4 = Freitag
            d += timedelta(days=1)
        if d >= today:
            events.append({
                "date": d.strftime("%Y-%m-%d"),
                "event": "US Non-Farm Payrolls (geschätzt)",
                "region": "US", "importance": "!!!",
                "days_away": (d - today).days
            })

    events.sort(key=lambda x: x.get("date", "9999"))
    return events


@st.cache_data(ttl=60, show_spinner=False)
def get_index_ticker_data() -> list:
    """Holt Live-Kursdaten für den Home-Ticker (Indizes, Forex, Rohstoffe, Crypto).
    Nutzt Tagesdaten, da diese bei Yahoo Finance untertägig in Echtzeit aktualisiert werden,
    während der vorherige Schlusskurs korrekt erhalten bleibt."""
    symbols = {
        "^GDAXI": "DAX", "^GSPC": "S&P 500", "^NDX": "Nasdaq 100", "^DJI": "Dow Jones",
        "^FTSE": "FTSE 100", "^FCHI": "CAC 40", "^N225": "Nikkei 225",
        "EURUSD=X": "EUR/USD", "JPY=X": "USD/JPY", "GBPUSD=X": "GBP/USD",
        "GC=F": "Gold", "SI=F": "Silber", "BZ=F": "Brent Öl", "BTC-USD": "Bitcoin"
    }
    results = []
    try:
        # 5-Tage-Tagesdaten für den korrekten Schlusskurs des Vortags (prev_price)
        df_1d = yf.download(
            list(symbols.keys()), period="5d", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False
        )
        
        # 1-Minuten-Daten für den echten Live-Kurs (last_price)
        df_1m = yf.download(
            list(symbols.keys()), period="1d", interval="1m",
            group_by="ticker", auto_adjust=True, progress=False
        )

        for sym, name in symbols.items():
            try:
                last_price = None
                prev_price = None
                last_price_fallback = None

                # 1. Vortagesschluss aus Tagesdaten (1d) ermitteln
                if not df_1d.empty:
                    if isinstance(df_1d.columns, pd.MultiIndex):
                        if sym in df_1d.columns.get_level_values(0):
                            s1d = df_1d[sym]["Close"].squeeze().dropna()
                            if len(s1d) >= 2:
                                prev_price = float(s1d.iloc[-2])
                                last_price_fallback = float(s1d.iloc[-1])
                    else:
                        s1d = df_1d["Close"].squeeze().dropna()
                        if len(s1d) >= 2:
                            prev_price = float(s1d.iloc[-2])
                            last_price_fallback = float(s1d.iloc[-1])

                # 2. Echten Live-Kurs aus Minuten-Daten (1m) ermitteln
                if not df_1m.empty:
                    if isinstance(df_1m.columns, pd.MultiIndex):
                        if sym in df_1m.columns.get_level_values(0):
                            s1m = df_1m[sym]["Close"].squeeze().dropna()
                            if not s1m.empty:
                                last_price = float(s1m.iloc[-1])
                    else:
                        s1m = df_1m["Close"].squeeze().dropna()
                        if not s1m.empty:
                            last_price = float(s1m.iloc[-1])

                # Fallback, falls 1m-Daten komplett leer sind (passiert bei geschlossenen Märkten manchmal)
                if last_price is None:
                    last_price = last_price_fallback

                if last_price is None or prev_price is None or prev_price == 0:
                    continue

                change_pct = (last_price / prev_price - 1) * 100
                results.append({
                    "symbol": sym,
                    "name": name,
                    "price": last_price,
                    "change": change_pct
                })
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Ticker-Daten Fehler: {e}")
    return results
