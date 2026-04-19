# ============================================================================
# pages/2_Screener.py — Aktien-Screener mit Custom Formula Builder (Phase 2)
# ============================================================================

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Screener · Aktien Pro v8", layout="wide", page_icon="")

from modules.bookmarks import display_watchlist
from modules.screener import display_screener
from modules.ui_components import display_heatmap_tab, display_economic_calendar, display_stock_ticker
from modules.screener import INDEX_UNIVERSES

# Sidebar
st.sidebar.header("Navigation")
if st.sidebar.button(" Zur Analyse", use_container_width=True):
    st.switch_page("pages/1_Analyse.py")
if st.sidebar.button(" Backtesting", use_container_width=True):
    st.switch_page("pages/3_Backtesting.py")

display_stock_ticker()
display_watchlist()

# Tabs
tab_screener, tab_heatmap, tab_calendar = st.tabs([
    " Screener", " Sektor-Heatmap", " Wirtschaftskalender"
])

with tab_screener:
    display_screener()

with tab_heatmap:
    display_heatmap_tab(INDEX_UNIVERSES)

with tab_calendar:
    display_economic_calendar()


