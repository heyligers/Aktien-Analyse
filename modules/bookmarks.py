# ============================================================================
# modules/bookmarks.py — Bookmark / Watchlist & Portfolio-System
# ============================================================================

import json
import streamlit as st
from datetime import datetime
from modules.utils import BOOKMARK_FILE, PORTFOLIO_FILE, logger


# ============================================================================
# BOOKMARK / WATCHLIST
# ============================================================================

def load_bookmarks() -> dict:
    import os
    if os.path.exists(BOOKMARK_FILE):
        try:
            with open(BOOKMARK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Bookmark-Datei fehlerhaft: {e}")
    return {}


def save_bookmarks(bookmarks: dict):
    try:
        with open(BOOKMARK_FILE, "w", encoding="utf-8") as f:
            json.dump(bookmarks, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.error(f"Bookmark konnte nicht gespeichert werden: {e}")


def add_bookmark(ticker: str, name: str, note: str = ""):
    bookmarks = load_bookmarks()
    bookmarks[ticker] = {
        "name": name, "note": note,
        "added": datetime.now().strftime("%d.%m.%Y %H:%M")
    }
    save_bookmarks(bookmarks)


def remove_bookmark(ticker: str):
    bookmarks = load_bookmarks()
    bookmarks.pop(ticker, None)
    save_bookmarks(bookmarks)


def update_bookmark_note(ticker: str, note: str):
    bookmarks = load_bookmarks()
    if ticker in bookmarks:
        bookmarks[ticker]["note"] = note
        save_bookmarks(bookmarks)


def display_watchlist():
    bookmarks = load_bookmarks()
    st.sidebar.markdown("---")
    st.sidebar.subheader("★ Watchlist")
    if bookmarks:
        for sym, data in list(bookmarks.items()):
            col_a, col_b = st.sidebar.columns([3, 1])
            with col_a:
                label = f"**{sym}** · {data['name'][:12]}"
                if st.sidebar.button(label, key=f"wl_{sym}", use_container_width=True):
                    st.session_state["jump_ticker"] = sym
                    st.rerun()
            with col_b:
                if st.sidebar.button("✕", key=f"wl_del_{sym}"):
                    remove_bookmark(sym)
                    st.rerun()
            note = data.get("note", "")
            if note:
                st.sidebar.caption(f"{note}")

        with st.sidebar.expander("Notizen bearbeiten"):
            note_sym = st.selectbox("Ticker", list(bookmarks.keys()), key="note_edit_sym")
            current_note = bookmarks.get(note_sym, {}).get("note", "")
            new_note = st.text_area("Notiz", value=current_note, key="note_edit_text",
                                    placeholder="z.B. Kaufen unter 150€")
            if st.button("Speichern", key="save_note"):
                update_bookmark_note(note_sym, new_note)
                st.toast(f"Notiz für {note_sym} gespeichert!")
                st.rerun()
    else:
        st.sidebar.caption("Noch keine Einträge in der Watchlist.")


# ============================================================================
# PORTFOLIO
# ============================================================================

def load_portfolio() -> dict:
    import os
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Portfolio-Datei fehlerhaft: {e}")
    return {}


def save_portfolio(portfolio: dict):
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.error(f"Portfolio konnte nicht gespeichert werden: {e}")


def add_portfolio_position(ticker: str, name: str, shares: float,
                           buy_price: float, buy_date: str, note: str = ""):
    portfolio = load_portfolio()
    idx = 1
    key = ticker
    while key in portfolio:
        idx += 1
        key = f"{ticker}_{idx}"
    portfolio[key] = {
        "ticker": ticker, "name": name, "shares": shares,
        "buy_price": buy_price, "buy_date": buy_date, "note": note
    }
    save_portfolio(portfolio)


def remove_portfolio_position(key: str):
    portfolio = load_portfolio()
    portfolio.pop(key, None)
    save_portfolio(portfolio)
