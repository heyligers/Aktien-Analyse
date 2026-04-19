# ============================================================================
# modules/ui_components.py — Anzeige-Komponenten & Metrik-Karten
# ============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

from modules.utils import fmt_number, logger
from modules.data_api import (
    get_full_ticker_data, get_fundamentals, get_options_data, get_macro_context,
    calculate_risk_metrics, calculate_correlation_matrix,
    get_insider_data, get_reddit_posts, get_heatmap_data, get_economic_calendar,
    get_index_ticker_data
)
from modules.bookmarks import load_portfolio, add_portfolio_position, remove_portfolio_position
from modules.news_api import sentiment_score

# ============================================================================
# KPI-Erklärungen
# ============================================================================
KPI_HELP = {
    "KGV (TTM)": "**Kurs-Gewinn-Verhältnis (Trailing)** — Aktienkurs / Gewinn je Aktie der letzten 12 Monate.",
    "KGV (Forward)": "**Kurs-Gewinn-Verhältnis (Forward)** — Basierend auf dem erwarteten Gewinn der nächsten 12 Monate.",
    "KBV": "**Kurs-Buchwert-Verhältnis** — Vergleicht Börsenwert mit bilanziellem Eigenkapital. KBV < 1 = unter Buchwert.",
    "EV/EBITDA": "**Enterprise Value / EBITDA** — Gesamtwert inkl. Schulden / operativer Gewinn. Unter 10 gilt als günstig.",
    "Marktkapitalisierung": "**Marktkapitalisierung** — Gesamtwert aller Aktien. Large Cap: > 10 Mrd.",
    "Umsatz (TTM)": "**Umsatz der letzten 12 Monate**",
    "Gewinnmarge": "**Nettomarge** — Anteil des Umsatzes als Nettogewinn. 20%+ gilt als sehr gut.",
    "ROE": "**Return on Equity** — Zeigt Kapitaleffizienz. Über 15% ist solide.",
    "Dividendenrendite": "**Dividendenrendite** — Jährliche Dividende in % des Kurses.",
    "Beta": "**Beta** — Kurssensitivität vs. Gesamtmarkt. >1 = volatiler, <1 = defensiver.",
    "52W Hoch": "**52-Wochen-Hoch**",
    "52W Tief": "**52-Wochen-Tief**",
}


# ============================================================================
# FUNDAMENTALDATEN
# ============================================================================

def display_fundamentals(fund: dict, ticker_sym: str):
    if not fund["kpis"]:
        st.warning("Keine Fundamentaldaten verfügbar.")
        return

    st.markdown("#### Kennzahlen")
    kpi_items = list(fund["kpis"].items())
    half = (len(kpi_items) + 1) // 2
    c1, c2 = st.columns(2)
    with c1:
        for k, v in kpi_items[:half]:
            st.metric(k, v, help=KPI_HELP.get(k))
    with c2:
        for k, v in kpi_items[half:]:
            st.metric(k, v, help=KPI_HELP.get(k))

    if fund["analyst"]:
        st.markdown("#### Analysten-Konsens")
        rec = fund["analyst"].get("Empfehlung", "—")
        color = {"Strong Buy": "▲", "Buy": "▲", "Hold": "◆",
                 "Underperform": "▼", "Sell": "▼", "Strong Sell": "▼"}.get(rec, "○")
        n_analysts = fund["analyst"].get("Analysten", "—")
        target_mean = fund["analyst"].get("Kursziel (Ø)", "—")
        target_low = fund["analyst"].get("Kursziel Tief", "—")
        target_high = fund["analyst"].get("Kursziel Hoch", "—")
        ca, cb = st.columns(2)
        with ca:
            st.metric("Empfehlung", f"{color} {rec}", help=f"Konsens aus {n_analysts} Analysten")
        with cb:
            st.metric("Kursziel Ø", target_mean, help=f"Bandbreite: {target_low} – {target_high}")

        score = fund["analyst"].get("Score")
        if score is not None:
            try:
                score = float(score)
                labels = ["Strong\nBuy", "Buy", "Hold", "Under-\nperform", "Sell"]
                colors_seg = ["#26a69a", "#66bb6a", "#ffa726", "#ef5350", "#b71c1c"]
                filled = int(round(score)) - 1
                bar_html = "<div style='display:flex;gap:4px;margin:8px 0 4px 0;'>"
                for i, (lbl, col) in enumerate(zip(labels, colors_seg)):
                    opacity = "1.0" if i == filled else "0.25"
                    bar_html += (f"<div style='flex:1;background:{col};opacity:{opacity};"
                                 f"border-radius:4px;padding:6px 2px;text-align:center;"
                                 f"font-size:0.7rem;color:white;font-weight:bold;'>{lbl}</div>")
                bar_html += "</div>"
                bar_html += f"<div style='font-size:0.75rem;color:#888;'>Score: {score:.1f}/5,0 · {n_analysts} Analysten</div>"
                st.markdown(bar_html, unsafe_allow_html=True)
            except (ValueError, TypeError):
                pass

        try:
            data = get_full_ticker_data(ticker_sym)
            current = data["fast_info"]["last_price"]
            t_val = float(fund["analyst"]["Kursziel (Ø)"].replace(",", ".").replace(" ", ""))
            upside = (t_val / current - 1) * 100
            arrow = "↑" if upside > 0 else "↓"
            st.markdown(f"{arrow} **Kurspotenzial:** `{upside:+.1f} %` "
                        f"(aktuell ~{current:.2f} → Ø-Ziel {target_mean})")
        except (TypeError, ValueError, ZeroDivisionError, KeyError):
            pass

        ud = fund.get("upgrades_downgrades")
        if ud is not None and not ud.empty:
            st.markdown("**Letzte Analysten-Aktionen:**")
            cols_available = ud.columns.tolist()
            show_cols = [c for c in ["GradeDate", "Firm", "ToGrade", "FromGrade", "Action"]
                         if c in cols_available]
            ud_show = ud[show_cols].copy()
            if "GradeDate" in ud_show.columns:
                ud_show["GradeDate"] = pd.to_datetime(ud_show["GradeDate"]).dt.strftime("%d.%m.%Y")
            rename = {"GradeDate": "Datum", "Firm": "Analysehaus",
                      "ToGrade": "Neues Rating", "FromGrade": "Altes Rating", "Action": "Aktion"}
            ud_show = ud_show.rename(columns=rename)
            def _action_icon(a):
                a = str(a).lower()
                if "up" in a: return "↑ " + a.title()
                if "down" in a: return "↓ " + a.title()
                return a.title()
            if "Aktion" in ud_show.columns:
                ud_show["Aktion"] = ud_show["Aktion"].apply(_action_icon)
            st.dataframe(ud_show, use_container_width=True, hide_index=True)

    if fund["calendar"]:
        st.info(f"Nächster Earnings-Termin: **{fund['calendar']}**")

    if fund["quarterly_financials"] is not None:
        st.markdown("#### Quartalszahlen")
        qf = fund["quarterly_financials"]
        _label_map = {"Total Revenue": "Umsatz", "Net Income": "Nettogewinn"}
        qf.index = pd.Index([_label_map.get(i, i) for i in qf.index])
        qf.columns = [str(c)[:10] for c in qf.columns]
        st.dataframe(
            qf.apply(lambda r: r.apply(lambda x: fmt_number(x, large=True) if pd.notna(x) else "—")),
            use_container_width=True
        )

    if fund["dividends"] is not None:
        st.markdown("#### Dividendenhistorie")
        div_s = fund["dividends"]
        div_df = div_s.reset_index()
        div_df.columns = ["Datum", "Dividende"]
        div_df["Datum"] = pd.to_datetime(div_df["Datum"]).dt.strftime("%d.%m.%Y")
        div_df["Dividende"] = div_df["Dividende"].apply(lambda x: f"{x:.4f}")
        st.dataframe(div_df, use_container_width=True, hide_index=True)
        st.caption(f"Letzte {len(div_df)} Ausschüttungen")


def display_options(opt: dict, ticker_sym: str):
    if opt.get("error"):
        st.warning(f"Optionsdaten: {opt['error']}")
        return
    st.markdown(f"#### Nächste Expiry: `{opt.get('expiry', '—')}`")
    c1, c2 = st.columns(2)
    with c1:
        iv = opt.get("iv")
        if iv is not None:
            st.metric("Implied Volatility (ATM)", f"{iv:.1f}%")
            if iv > 40:
                st.caption("↑ Hohe IV — Markt erwartet starke Bewegung")
            elif iv > 20:
                st.caption("◆ Moderate IV")
            else:
                st.caption("↓ Niedrige IV — ruhiges Marktumfeld")
    with c2:
        pcr = opt.get("pcr")
        if pcr is not None:
            st.metric("Put/Call Ratio (OI)", f"{pcr:.3f}")
            if pcr > 1.2:
                st.caption("▼ Bärische Stimmung (viele Puts)")
            elif pcr < 0.7:
                st.caption("▲ Bullische Stimmung (viele Calls)")
            else:
                st.caption("○ Neutrale Stimmung")


def display_macro(macro: dict, ticker_sym: str, market_proxy: str):
    def corr_label(c):
        if c is None: return "—", "○"
        if c > 0.7: return f"{c:.2f}", "▲ Stark positiv"
        if c > 0.4: return f"{c:.2f}", "◆ Moderat"
        if c > 0.0: return f"{c:.2f}", "○ Schwach"
        return f"{c:.2f}", "● Negativ / Hedge"

    st.markdown("#### Korrelation (1 Jahr)")
    c1, c2 = st.columns(2)
    with c1:
        cv, cl = corr_label(macro.get("corr_market"))
        st.metric(f"vs. {market_proxy}", cv)
        st.caption(cl)
    with c2:
        etf = macro.get("sector_etf") or "—"
        cv2, cl2 = corr_label(macro.get("corr_sector"))
        st.metric(f"vs. Sektor-ETF ({etf})", cv2)
        st.caption(cl2)

    st.markdown("#### Performance (1 Jahr)")
    try:
        end = datetime.today().date()
        start = end - timedelta(days=365)
        df_t = yf.download(ticker_sym, start=start, end=end, interval="1d",
                           auto_adjust=True, progress=False)
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
        if not df_t.empty:
            ticker_perf = round((float(df_t["Close"].iloc[-1]) / float(df_t["Close"].iloc[0]) - 1) * 100, 2)
            perf_color = "▲" if ticker_perf >= 0 else "▼"
            st.markdown(f"**{ticker_sym}**: {perf_color} `{ticker_perf:+.2f}%`")
    except Exception:
        pass
    if macro.get("market_perf") is not None:
        val = macro["market_perf"]
        st.markdown(f"**{market_proxy}**: {'▲' if val >= 0 else '▼'} `{val:+.2f}%`")
    if macro.get("sector_perf") is not None:
        val = macro["sector_perf"]
        st.markdown(f"**{macro['sector_etf']}**: {'▲' if val >= 0 else '▼'} `{val:+.2f}%`")
    if macro.get("sector_name"):
        st.caption(f"Sektor: {macro['sector_name']}")


# ============================================================================
# MULTI-TICKER VERGLEICH
# ============================================================================

def display_comparison(ticker, start_date, end_date):
    compare_input = st.text_input("Vergleichs-Ticker (kommagetrennt)",
                                  placeholder="z.B. MSFT, GOOGL, AMZN",
                                  key="compare_input")
    if not compare_input:
        st.caption("Gib Ticker ein, um den Performance-Vergleich zu starten.")
        return
    compare_tickers = [t.strip().upper() for t in compare_input.split(",") if t.strip()]
    if not compare_tickers:
        return
    all_tickers = [ticker] + compare_tickers
    comparison_data = {}
    with st.spinner("Vergleichsdaten werden geladen…"):
        for sym in all_tickers:
            try:
                df_c = yf.download(sym, start=start_date, end=end_date,
                                   interval="1d", progress=False, auto_adjust=True)
                if not df_c.empty:
                    if isinstance(df_c.columns, pd.MultiIndex):
                        df_c.columns = df_c.columns.get_level_values(0)
                    close = df_c["Close"].squeeze()
                    comparison_data[sym] = (close / float(close.iloc[0]) * 100)
            except Exception as e:
                logger.warning(f"Vergleich: {sym} fehlgeschlagen: {e}")
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        st.line_chart(comp_df)
        st.caption("Normalisiert auf 100 am Startdatum")
    else:
        st.warning("Keine Vergleichsdaten gefunden.")


# ============================================================================
# PORTFOLIO
# ============================================================================

def display_portfolio():
    st.markdown("### Portfolio-Tracker")
    from modules.data_api import get_ticker_info
    portfolio = load_portfolio()

    if portfolio:
        rows = []
        total_invested = 0.0
        total_current = 0.0
        for key, pos in portfolio.items():
            sym = pos.get("ticker", key)
            try:
                data = get_full_ticker_data(sym)
                current_price = data["fast_info"]["last_price"]
                if current_price is None:
                    continue
                invested = pos['shares'] * pos['buy_price']
                current_val = pos['shares'] * current_price
                pl = current_val - invested
                pl_pct = (pl / invested) * 100 if invested > 0 else 0
                total_invested += invested
                total_current += current_val
                rows.append({
                    "Ticker": sym, "Name": pos.get('name', sym)[:20],
                    "Stück": pos['shares'],
                    "Kaufkurs": f"{pos['buy_price']:.2f}",
                    "Aktuell": f"{current_price:.2f}",
                    "Investiert": fmt_number(invested, large=True),
                    "Wert": fmt_number(current_val, large=True),
                    "G/V": f"{pl:+,.2f}",
                    "G/V %": f"{pl_pct:+.1f}%",
                })
            except Exception as e:
                logger.warning(f"Portfolio-Fehler für {sym}: {e}")

        if rows:
            total_pl = total_current - total_invested
            total_pl_pct = (total_pl / total_invested) * 100 if total_invested > 0 else 0
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Investiert", fmt_number(total_invested, large=True))
            mc2.metric("Wert", fmt_number(total_current, large=True))
            mc3.metric("G/V", f"{total_pl:+,.2f}", delta=f"{total_pl_pct:+.1f}%")
            mc4.metric("Positionen", len(rows))
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                del_options = {f"{pos.get('ticker', k)} ({pos.get('shares', '?')} Stk. @ {pos.get('buy_price', '?')})": k
                               for k, pos in portfolio.items()}
                del_choice = st.selectbox("Position zum Löschen", list(del_options.keys()))
            with del_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("× Löschen", key="del_pos"):
                    remove_portfolio_position(del_options[del_choice])
                    st.toast("Position gelöscht.")
                    st.rerun()
    else:
        st.info("Noch keine Positionen vorhanden.")

    st.markdown("---")
    st.markdown("#### Position hinzufügen")
    with st.form("add_position", clear_on_submit=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        sym_input = fc1.text_input("Ticker", placeholder="z.B. AAPL")
        shares_input = fc2.number_input("Stück", min_value=0.01, value=1.0, step=1.0)
        price_input = fc3.number_input("Kaufkurs", min_value=0.01, value=100.0, step=0.01)
        date_input = fc4.date_input("Kaufdatum")
        note_input = st.text_input("Notiz (optional)")
        submitted = st.form_submit_button("Speichern", type="primary")
        if submitted and sym_input:
            from modules.data_api import get_ticker_info
            sym_clean = sym_input.strip().upper()
            info = get_ticker_info(sym_clean)
            add_portfolio_position(sym_clean, info.get("name", sym_clean),
                                   shares_input, price_input,
                                   date_input.strftime("%d.%m.%Y"), note_input)
            st.success(f"✓ {sym_clean} hinzugefügt!")
            st.rerun()


# ============================================================================
# KORRELATIONS-HEATMAP
# ============================================================================

def display_correlation_heatmap(corr_df: pd.DataFrame):
    if corr_df.empty:
        st.info("Nicht genug Daten für eine Korrelations-Matrix.")
        return
    tickers = corr_df.columns.tolist()
    cells_html = ""
    for i, t1 in enumerate(tickers):
        row = ""
        for j, t2 in enumerate(tickers):
            val = corr_df.loc[t1, t2]
            if val >= 0.7:
                bg = f"rgba(239,83,80,{min(val, 1.0)*0.8})"
            elif val >= 0.4:
                bg = f"rgba(255,152,0,{val*0.6})"
            elif val >= 0:
                bg = f"rgba(255,255,255,{val*0.3})"
            elif val >= -0.4:
                bg = f"rgba(66,165,245,{abs(val)*0.5})"
            else:
                bg = f"rgba(30,136,229,{abs(val)*0.8})"
            row += f'<td style="background:{bg};padding:6px 8px;text-align:center;font-size:0.8rem;border:1px solid #333;">{val:.2f}</td>'
        cells_html += f'<tr><td style="padding:4px 8px;font-weight:bold;font-size:0.8rem;color:#ddd;border:1px solid #333;">{t1}</td>{row}</tr>'
    header = "".join(f'<th style="padding:4px 6px;font-size:0.75rem;color:#aaa;border:1px solid #333;writing-mode:vertical-lr;transform:rotate(180deg);">{t}</th>' for t in tickers)
    html = f"""<div style="overflow-x:auto;">
    <table style="border-collapse:collapse;background:#1a1a2e;">
      <tr><th></th>{header}</tr>
      {cells_html}
    </table>
    <div style="font-size:0.7rem;color:#888;margin-top:4px;">▲ Stark positiv · ◇ Moderat · ○ Schwach · ▼ Negativ</div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# SEKTOR-HEATMAP
# ============================================================================

def display_heatmap_tab(index_universes: dict):
    st.markdown("### Sektor-Heatmap")

    hm_col1, hm_col2, hm_col3 = st.columns([2, 1, 1])
    with hm_col1:
        universe_choice = st.selectbox("Index wählen", list(index_universes.keys()), key="heatmap_idx")
    with hm_col2:
        heatmap_view = st.radio("Ansicht", [" Grid", " Treemap"], horizontal=True, key="heatmap_view")
    with hm_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Cache leeren", key="clear_hm_cache",
                     help="Sektordaten und Preise neu laden"):
            from modules.data_api import _get_bulk_prices, _get_ticker_meta_heatmap_cached
            _get_bulk_prices.clear()
            _get_ticker_meta_heatmap_cached.clear()
            st.session_state["heatmap_loaded"] = False
            st.success("Cache geleert! Bitte Heatmap neu laden.")
            st.rerun()

    if st.button("Heatmap laden", type="primary", key="load_heatmap") or \
       st.session_state.get("heatmap_loaded"):
        st.session_state["heatmap_loaded"] = True

        # 1. Ticker-Liste ermitteln
        from modules.index_utils import get_full_universe
        tickers = get_full_universe(universe_choice)
        if not tickers:
            tickers = index_universes.get(universe_choice, [])

        # 2. Daten laden mit Progress Bar
        progress_container = st.empty()
        def _update_progress(val, text):
            progress_container.progress(val / 100, text=text)

        data = get_heatmap_data(universe_choice, tickers, progress_cb=_update_progress)
        progress_container.empty()
        if not data:
            st.warning("Keine Daten verfügbar.")
            return
        # -- Ansicht: Treemap oder Grid --
        use_treemap = "Treemap" in heatmap_view

        if use_treemap:
            # === ITEM #10: TREEMAP-HEATMAP (Manual with go.Treemap) ===
            df_hm = pd.DataFrame(data)
            df_hm["mcap_safe"] = df_hm["mcap"].fillna(1e8).clip(lower=1e7)
            df_hm["sector"] = df_hm["sector"].fillna("Sonstiges")
            df_hm["change"] = pd.to_numeric(df_hm["change"], errors='coerce').fillna(0.0)
            df_hm["price"] = pd.to_numeric(df_hm["price"], errors='coerce').fillna(0.0)

            labels = []
            parents = []
            values = []
            color_vals = []
            custom_changes = []
            custom_prices = []
            
            root_label = universe_choice
            labels.append(root_label)
            parents.append("")
            
            total_mcap = df_hm["mcap_safe"].sum()
            values.append(total_mcap)
            
            root_change = (df_hm["change"] * df_hm["mcap_safe"]).sum() / total_mcap if total_mcap else 0.0
            color_vals.append(root_change)
            custom_changes.append(f"{root_change:+.2f}%".replace('.', ','))
            custom_prices.append("—")
            
            for sector, grp in df_hm.groupby("sector"):
                labels.append(sector)
                parents.append(root_label)
                sec_mcap = grp["mcap_safe"].sum()
                values.append(sec_mcap)
                
                sec_chg = (grp["change"] * grp["mcap_safe"]).sum() / sec_mcap if sec_mcap else 0.0
                color_vals.append(sec_chg)
                custom_changes.append(f"{sec_chg:+.2f}%".replace('.', ','))
                custom_prices.append("—")
                
                for _, row in grp.iterrows():
                    # Ticker names need to be unique across the tree. If duplicate tickers exist, we could append sector.
                    labels.append(row["ticker"])
                    parents.append(sector)
                    values.append(row["mcap_safe"])
                    color_vals.append(row["change"])
                    custom_changes.append(f"{row['change']:+.2f}%".replace('.', ','))
                    p = row["price"]
                    custom_prices.append(f"{p:.2f}".replace('.', ',') if p else "—")
            
            customdata = list(zip(custom_changes, custom_prices))
            
            fig_tree = go.Figure(go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                marker=dict(
                    colors=color_vals,
                    colorscale=[(0.0, "#880e0e"), (0.3, "#c62828"),
                                (0.5, "#1e2230"),
                                (0.7, "#2e7d32"), (1.0, "#1b5e20")],
                    cmid=0,
                    showscale=False
                ),
                customdata=customdata,
                texttemplate="<b>%{label}</b><br>%{customdata[0]}",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Change: %{customdata[0]}<br>"
                    "Price: %{customdata[1]}<extra></extra>"
                ),
                branchvalues="total"
            ))
            
            fig_tree.update_layout(
                title=f"{universe_choice} — Treemap nach Marktkapitalisierung",
                paper_bgcolor="#131722", font_color="#d1d4dc",
                height=700, margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_tree, use_container_width=True)

        else:
            # === BESTEHENDE GRID-ANSICHT ===
            from itertools import groupby
            import textwrap

            # WICHTIG: groupby() gruppiert nur aufeinanderfolgende gleiche Elemente.
            # Deshalb MUSS vorher nach Sektor sortiert werden, sonst erscheint
            # jeder Sektor mehrmals mit jeweils nur einer Aktie.
            data.sort(key=lambda x: (str(x.get('sector', 'Sonstiges')), -(x.get('mcap') or 0)))

            for sector, group in groupby(data, key=lambda x: str(x.get('sector', 'Sonstiges'))):
                group_list = list(group)
                header_html = textwrap.dedent(f"""
                    <div style="margin-top:20px; margin-bottom:10px; font-weight:bold;
                         border-bottom:1px solid #444; color:#00d4ff; padding:4px; font-size:1.1rem;">
                        {sector} <span style="font-size:0.8rem; color:#888; font-weight:normal;">({len(group_list)} Titel)</span>
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(130px, 1fr)); gap:8px; padding:4px;">
                """).strip()
                cells_html = ""
                for d in group_list:
                    ch = d.get('change', 0)
                    if ch > 2: bg = "#1b5e20"
                    elif ch > 1: bg = "#2e7d32"
                    elif ch > 0: bg = "#388e3c"
                    elif ch > -1: bg = "#c62828"
                    elif ch > -2: bg = "#b71c1c"
                    else: bg = "#880e0e"
                    arrow = "▲" if ch > 0 else "▼" if ch < 0 else "●"
                    mcap_label = fmt_number(d.get('mcap', 0), large=True)
                    cells_html += textwrap.dedent(f"""
                        <div style="background:{bg}; border-radius:6px; padding:10px;
                            text-align:center; min-width:110px;
                            box-shadow: 2px 2px 6px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);">
                            <div style="font-weight:bold; font-size:0.85rem; color:white; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{d.get('ticker', '—')}</div>
                            <div style="font-size:0.65rem; color:rgba(255,255,255,0.8); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-bottom:4px;">{d.get('name', '—')[:15]}</div>
                            <div style="font-size:1rem; font-weight:bold; color:white; margin:2px 0;">{arrow} {ch:+.2f}%</div>
                            <div style="font-size:0.6rem; color:rgba(255,255,255,0.6);">{mcap_label}</div>
                        </div>
                    """).strip()
                st.markdown(header_html + cells_html + "</div>", unsafe_allow_html=True)

        total = len(data)
        up = sum(1 for d in data if d['change'] > 0)
        down = total - up
        avg_ch = sum(d['change'] for d in data) / total if total else 0
        st.markdown(f"**Zusammenfassung:** ▲ {up} steigend · ▼ {down} fallend · "
                    f"Ø Veränderung: `{avg_ch:+.2f}%`")
        cols = st.columns(min(len(data), 8))
        for i, d in enumerate(sorted(data, key=lambda x: x['change'])[:8]):
            with cols[i % 8]:
                if st.button(d['ticker'], key=f"hm_jump_{d['ticker']}"):
                    st.session_state["jump_ticker"] = d['ticker']
                    st.rerun()


# ============================================================================
# WIRTSCHAFTSKALENDER
# ============================================================================

def display_economic_calendar():
    st.markdown("### Wirtschaftskalender")
    events = get_economic_calendar()
    if not events:
        st.info("Keine anstehenden Events gefunden.")
        return

    fc1, fc2 = st.columns(2)
    with fc1:
        region_filter = st.multiselect("Region", ["US", "EU"], default=["US", "EU"])
    with fc2:
        time_filter = st.selectbox("Zeitraum",
            ["Alle", "Diese Woche", "Nächste 30 Tage", "Nächste 90 Tage"], key="cal_time")

    filtered = [e for e in events if e["region"] in region_filter]
    if time_filter == "Diese Woche":
        filtered = [e for e in filtered if 0 <= e.get("days_away", 999) <= 7]
    elif time_filter == "Nächste 30 Tage":
        filtered = [e for e in filtered if 0 <= e.get("days_away", 999) <= 30]
    elif time_filter == "Nächste 90 Tage":
        filtered = [e for e in filtered if 0 <= e.get("days_away", 999) <= 90]

    if not filtered:
        st.info("Keine Events im gewählten Zeitraum.")
        return

    for ev in filtered:
        days = ev.get("days_away", 0)
        badge = "● HEUTE" if days == 0 else "◇ Morgen" if days == 1 else f"In {days} Tagen"
        st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
            background:rgba(255,255,255,0.04);border-radius:6px;margin-bottom:6px;
            border-left:3px solid {'#ef5350' if days<=1 else '#ffa726' if days<=7 else '#555'};">
            <span style="font-size:1.3rem;">{ev['region']}</span>
            <div style="flex:1;">
                <div style="font-weight:bold;color:#eee;">{ev['event']}</div>
                <div style="font-size:0.75rem;color:#888;">{ev['date']} · {ev['importance']}</div>
            </div>
            <span style="font-size:0.8rem;color:#aaa;">{badge}</span>
        </div>""", unsafe_allow_html=True)
    st.caption(f"{len(filtered)} Events angezeigt")


# ============================================================================
# INSIDER-TRANSAKTIONEN
# ============================================================================

def display_insider(ticker_sym: str):
    st.markdown("#### Insider-Transaktionen")
    st.caption("Daten von OpenInsider.com — nur US-Aktien · Letzte 90 Tage")
    
    if "." in ticker_sym and not ticker_sym.endswith(".US"):
        st.info("Hinweis: OpenInsider unterstützt nur US-Aktien. Es wird das Basis-Symbol gesucht.")
        
    txns = get_insider_data(ticker_sym)
    if not txns:
        st.info("Keine Insider-Transaktionen gefunden.")
        return
    buys = sum(1 for t in txns if t["is_buy"])
    sells = len(txns) - buys
    c1, c2, c3 = st.columns(3)
    c1.metric("▲ Käufe", buys)
    c2.metric("▼ Verkäufe", sells)
    ratio = buys / max(buys + sells, 1)
    c3.metric("Buy-Ratio", f"{ratio:.0%}")
    for t in txns:
        emoji = "▲" if t["is_buy"] else "▼"
        st.markdown(f"{emoji} **{t['name']}** ({t['role']}) — "
                    f"{t['type']} · ${t['price']} × {t['qty']} = **{t['value']}** · {t['date']}")


# ============================================================================
# SOCIAL SENTIMENT
# ============================================================================

def display_social_sentiment(ticker_sym: str, company_name: str, lang: str):
    st.markdown("#### Social Sentiment")
    st.caption("Reddit: r/wallstreetbets · r/stocks · r/investing")
    posts = get_reddit_posts(ticker_sym, company_name)
    if not posts:
        st.info(f"Keine Reddit-Posts zu `{ticker_sym}` oder `{company_name}` gefunden.")
        return
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(p['title'])['compound'] for p in posts]
    except Exception:
        scores = [0.0] * len(posts)
    avg_score = sum(scores) / len(scores)
    pos = sum(1 for s in scores if s > 0.05)
    neg = sum(1 for s in scores if s < -0.05)
    emoji = "▲" if avg_score > 0.05 else "▼" if avg_score < -0.05 else "○"
    st.markdown(f"{emoji} **Social Sentiment: `{avg_score:+.3f}`** · "
                f"▲ {pos} positiv · ▼ {neg} negativ · {len(posts)} Posts")
    st.markdown("---")
    for p, score in zip(posts, scores):
        s_emoji = "▲" if score > 0.05 else "▼" if score < -0.05 else "○"
        age = (datetime.now() - p['dt'])
        age_str = f"vor {age.days} T" if age.days > 0 else f"vor {age.seconds // 3600} Std"
        st.markdown(f"""{s_emoji} **[{p['title'][:80]}]({p['url']})**
<span style="font-size:0.75rem;color:#888;">r/{p['subreddit']} · ↑ {p['score']} · {p['comments']} Komm. · {age_str}</span>""",
                    unsafe_allow_html=True)


def display_stock_ticker():
    """Zeigt einen scrollenden Börsen-Ticker im N24/WELT-Stil."""
    data = get_index_ticker_data()
    if not data:
        return

    # Erzeuge HTML für die Ticker-Items
    ticker_items = ""
    for d in data:
        color = "#26a69a" if d["change"] >= 0 else "#ef5350"
        arrow = "▲" if d["change"] >= 0 else "▼"
        price_fmt = f"{d['price']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        ticker_items += f'<div class="ticker-item"><span class="ticker-name">{d["name"]}</span><span class="ticker-price">{price_fmt}</span><span class="ticker-change" style="color: {color};">{arrow} {abs(d["change"]):.2f}%</span></div>'

    # CSS für das Ticker-Design
    ticker_css = """<style>
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: #131722;
    padding: 10px 0;
    border-bottom: 1px solid #2a2e39;
    margin-bottom: 20px;
    white-space: nowrap;
}
.ticker-move {
    display: inline-flex;
    white-space: nowrap;
    animation: ticker-scroll 100s linear infinite;
}
.ticker-move:hover {
    animation-play-state: paused;
}
.ticker-item {
    display: inline-block;
    padding: 0 40px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.ticker-name { color: #d1d4dc; margin-right: 8px; }
.ticker-price { color: #ffffff; font-weight: bold; margin-right: 8px; }

@keyframes ticker-scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
</style>"""

    ticker_html = f'{ticker_css}<div class="ticker-wrap"><div class="ticker-move">{ticker_items}{ticker_items}</div></div>'
    st.markdown(ticker_html, unsafe_allow_html=True)
