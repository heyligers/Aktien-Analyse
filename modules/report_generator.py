# ============================================================================
# modules/report_generator.py — PDF-Report-Generator (Phase 4)
# ============================================================================
# Abhängigkeiten: fpdf2, kaleido, plotly
# Installation: pip install fpdf2 kaleido plotly
# ============================================================================

import streamlit as st
import pandas as pd
import os
import io
import tempfile
from datetime import datetime

from modules.utils import fmt_number, logger


def _check_dependencies() -> tuple[bool, str]:
    """Prüft ob fpdf2 und kaleido installiert sind."""
    missing = []
    try:
        from fpdf import FPDF
    except ImportError:
        missing.append("fpdf2")
    try:
        import kaleido
    except ImportError:
        missing.append("kaleido")
    if missing:
        return False, f"Fehlende Pakete: {', '.join(missing)}. Installiere mit: pip install {' '.join(missing)}"
    return True, ""


def _export_plotly_to_png(fig, width=700, height=300) -> bytes | None:
    """Exportiert einen Plotly-Chart als PNG-Bytes via kaleido."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=1.5)
    except Exception as e:
        logger.warning(f"Kaleido PNG-Export fehlgeschlagen: {e}")
        return None


def generate_pdf_report(
    ticker: str,
    company_name: str,
    fund_data: dict,
    ai_summary: str = "",
    chart_fig=None,
    bt_metrics: dict = None,
) -> bytes | None:
    """
    Generiert einen A4-PDF-Aktien-Steckbrief.

    Args:
        ticker: Ticker-Symbol
        company_name: Unternehmensname
        fund_data: Fundamentaldaten-Dict aus get_fundamentals()
        ai_summary: Gemini KI-Zusammenfassung (optional)
        chart_fig: Plotly-Figure für den Chart (optional)
        bt_metrics: Backtesting-Kennzahlen-Dict (optional)

    Returns:
        PDF als bytes oder None bei Fehler
    """
    ok, err = _check_dependencies()
    if not ok:
        st.error(f"PDF-Generator: {err}")
        return None

    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                # Farbiger Header-Balken
                self.set_fill_color(19, 23, 34)   # #131722 dunkelblau
                self.rect(0, 0, 210, 20, 'F')
                self.set_text_color(209, 212, 220)
                self.set_font("Helvetica", "B", 14)
                self.set_y(5)
                self.cell(0, 10, f"Aktien-Analyse: {ticker}  |  {company_name[:40]}", align="L")
                self.set_font("Helvetica", "", 8)
                self.set_text_color(120, 123, 134)
                self.cell(0, 10, f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
                          align="R")
                self.ln(14)

            def footer(self):
                self.set_y(-12)
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(120, 123, 134)
                self.cell(0, 10,
                          f"Seite {self.page_no()} | Aktien Analyse Pro | Nur zu Informationszwecken",
                          align="C")

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_margins(12, 22, 12)

        # ── Hilfsfunktionen ──────────────────────────────────────────────────
        def section_header(title: str):
            pdf.set_fill_color(30, 34, 45)
            pdf.set_text_color(41, 98, 255)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(1)

        def kv_row(key: str, value: str, odd: bool = False):
            if odd:
                pdf.set_fill_color(245, 245, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(50, 50, 60)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(65, 6, key, fill=True)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT", fill=True)

        # ── 1. Kurz-Info Zeile ───────────────────────────────────────────────
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 110)
        pdf.cell(0, 5,
                 f"Ticker: {ticker}  |  Unternehmen: {company_name}  |  "
                 f"Stand: {datetime.now().strftime('%d.%m.%Y')}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # ── 2. Chart ─────────────────────────────────────────────────────────
        if chart_fig is not None:
            png_bytes = _export_plotly_to_png(chart_fig, width=680, height=260)
            if png_bytes:
                section_header("📈 Kurschart")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(png_bytes)
                    tmp_path = tmp.name
                try:
                    pdf.image(tmp_path, x=12, w=186, h=70)
                finally:
                    os.unlink(tmp_path)
                pdf.ln(4)

        # ── 3. Fundamentaldaten ───────────────────────────────────────────────
        kpis = fund_data.get("kpis", {})
        if kpis:
            section_header("📊 Fundamentale Kennzahlen")
            items = list(kpis.items())
            # Zweispaltige Tabelle
            half = (len(items) + 1) // 2
            col_width = 93
            odd = True
            for i in range(max(len(items[:half]), len(items[half:]))):
                if odd:
                    pdf.set_fill_color(245, 245, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)
                # Linke Spalte
                if i < len(items[:half]):
                    k, v = items[:half][i]
                    pdf.set_font("Helvetica", "", 8.5)
                    pdf.set_text_color(80, 80, 90)
                    pdf.cell(32, 5.5, k, fill=True)
                    pdf.set_font("Helvetica", "B", 8.5)
                    pdf.set_text_color(30, 30, 40)
                    pdf.cell(col_width - 32, 5.5, str(v), fill=True)
                else:
                    pdf.cell(col_width, 5.5, "", fill=True)
                # Rechte Spalte
                if i < len(items[half:]):
                    k2, v2 = items[half:][i]
                    pdf.set_font("Helvetica", "", 8.5)
                    pdf.set_text_color(80, 80, 90)
                    pdf.cell(32, 5.5, k2, fill=True)
                    pdf.set_font("Helvetica", "B", 8.5)
                    pdf.set_text_color(30, 30, 40)
                    pdf.cell(0, 5.5, str(v2), new_x="LMARGIN", new_y="NEXT", fill=True)
                else:
                    pdf.cell(0, 5.5, "", new_x="LMARGIN", new_y="NEXT", fill=True)
                odd = not odd
            pdf.ln(3)

        # Analysten-Konsens
        analyst = fund_data.get("analyst", {})
        if analyst:
            section_header("🎯 Analysten-Konsens")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 60)
            rec = analyst.get("Empfehlung", "—")
            target = analyst.get("Kursziel (Ø)", "—")
            n = analyst.get("Analysten", "—")
            pdf.cell(0, 5.5,
                     f"Empfehlung: {rec}  |  Kursziel (Ø): {target}  |  Analysten: {n}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        # ── 4. Backtesting-Ergebnisse ─────────────────────────────────────────
        if bt_metrics:
            section_header("⚡ Backtesting-Ergebnisse")
            bt_items = [
                ("Net Profit", f"${bt_metrics.get('net_profit', 0):+,.2f} ({bt_metrics.get('net_profit_pct', 0):+.1f}%)"),
                ("Max Drawdown", f"{bt_metrics.get('max_drawdown', 0):.1f}%"),
                ("Sharpe Ratio", f"{bt_metrics.get('sharpe_ratio', 0):.3f}"),
                ("Win Rate", f"{bt_metrics.get('win_rate', 0):.1f}%"),
                ("Gesamt Trades", str(bt_metrics.get("total_trades", 0))),
                ("Buy & Hold", f"{bt_metrics.get('buy_hold_return', 0):+.1f}%"),
            ]
            for i, (k, v) in enumerate(bt_items):
                kv_row(k, v, i % 2 == 0)
            pdf.ln(3)

        # ── 5. KI-Analyse ─────────────────────────────────────────────────────
        if ai_summary and ai_summary.strip():
            section_header("🤖 KI-Analyse (Gemini)")
            # Markdown-Symbole vereinfachen
            clean_text = (ai_summary
                          .replace("**", "")
                          .replace("###", "")
                          .replace("##", "")
                          .replace("#", "")
                          .replace("*", "•")
                          .strip())
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(50, 50, 60)
            # Zeilenumbrüche verarbeiten
            for line in clean_text.split("\n"):
                line = line.strip()
                if not line:
                    pdf.ln(2)
                    continue
                # Überschriften erkennen (fett machen)
                if line.startswith("•") or line.endswith(":"):
                    pdf.set_font("Helvetica", "B", 8.5)
                    pdf.multi_cell(0, 5, line)
                    pdf.set_font("Helvetica", "", 8.5)
                else:
                    pdf.multi_cell(0, 5, line)
            pdf.ln(3)

        # ── 6. Rechtlicher Hinweis ────────────────────────────────────────────
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(160, 160, 170)
        pdf.multi_cell(0, 4,
            "Haftungsausschluss: Dieser Report dient ausschliesslich zu Informationszwecken "
            "und stellt keine Anlageberatung dar. Alle Angaben ohne Gewähr. "
            "Investitionen in Wertpapiere sind mit Risiken verbunden.")

        # PDF als Bytes zurückgeben
        return bytes(pdf.output())

    except Exception as e:
        logger.error(f"PDF-Generierung fehlgeschlagen: {e}")
        st.error(f"PDF-Fehler: {e}")
        return None


# ============================================================================
# STREAMLIT DISPLAY
# ============================================================================

def display_pdf_export(ticker: str, company_name: str,
                       fund_data: dict, ai_summary: str = "",
                       chart_fig=None, bt_metrics: dict = None):
    """Zeigt den PDF-Export-Button in der Analyse-Ansicht."""
    ok, err = _check_dependencies()

    with st.expander("📄 PDF-Report exportieren"):
        if not ok:
            st.warning(f"PDF-Export nicht verfügbar: {err}")
            st.code("pip install fpdf2 kaleido", language="bash")
            return

        st.markdown("Erstellt einen professionellen **A4-Aktien-Steckbrief** mit:")
        st.markdown("- Kurschart (wenn Chart geladen)")
        st.markdown("- Fundamentale Kennzahlen & Analysten-Konsens")
        st.markdown("- Backtesting-Ergebnisse (wenn vorhanden)")
        st.markdown("- KI-Analyse (wenn Gemini-Key gesetzt)")

        if st.button("PDF generieren", type="primary", key=f"gen_pdf_{ticker}"):
            with st.spinner("PDF wird erstellt…"):
                pdf_bytes = generate_pdf_report(
                    ticker=ticker,
                    company_name=company_name,
                    fund_data=fund_data,
                    ai_summary=ai_summary,
                    chart_fig=chart_fig,
                    bt_metrics=bt_metrics,
                )
            if pdf_bytes:
                filename = f"aktien_report_{ticker}_{datetime.now().strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label=f"⬇ {filename} herunterladen",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                )
                st.success("✓ PDF erfolgreich erstellt!")
