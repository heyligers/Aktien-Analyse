# Aktien Analyse Pro v8 — Multi-Page Streamlit App

## Projektstruktur

```
aktien_app/
├── Home.py                    ← Einstiegspunkt (streamlit run Home.py)
├── requirements.txt
├── .streamlit/
│   ├── config.toml            ← Dark-Theme Konfiguration
│   └── secrets.toml.example   ← API-Key Vorlage (umbenennen!)
├── pages/
│   ├── 1_Analyse.py           ← Chart, News, Fundamentals, KI, PDF-Export
│   ├── 2_Screener.py          ← Screener + Custom Formula + Heatmap + Kalender
│   └── 3_Backtesting.py       ← Backtesting + Portfolio + Risk-Metriken
└── modules/
    ├── utils.py               ← Hilfsfunktionen, Logging, SSL-Fix
    ├── bookmarks.py           ← Watchlist & Portfolio (JSON-Persistenz)
    ├── data_api.py            ← Yahoo Finance, Makro, Risk, Insider, Reddit
    ├── news_api.py            ← yfinance News, Google RSS, NewsAPI, Sentiment
    ├── technical_analysis.py  ← Candlestick-Pattern, Pivot, Fibonacci
    ├── charting.py            ← LightweightCharts HTML-Builder
    ├── screener.py            ← Screener-Kern + Custom Formula Builder (Phase 2)
    ├── ai_gemini.py           ← Google Gemini KI-Integration
    ├── ui_components.py       ← Fundamentals, Heatmap, Kalender, Social, Insider
    ├── backtesting.py         ← Vektorisiertes Backtesting (Phase 3)
    └── report_generator.py    ← PDF-Report-Generator (Phase 4)
```

## Installation & Start

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. API-Keys konfigurieren (optional aber empfohlen)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# → Datei öffnen und Keys eintragen

# 3. App starten
streamlit run Home.py
```

## Features nach Phase

### Phase 1 — Modularisierung ✅
Saubere Aufteilung der 3.200-Zeilen-Monodatei `appv8.py` in wiederverwendbare Module.
Multi-Page-App mit schnelleren Ladezeiten.

### Phase 2 — Custom Screener Formula Builder ✅
TradingView-ähnlicher Formel-Editor im Screener.
Unterstützte Variablen: `Close`, `SMA20`, `SMA50`, `RSI`, `MACD`, `Signal`,
`PE`, `MarketCap`, `Pct52wHigh`, `Volume`

Beispiele:
```
(SMA20 > SMA50) and (RSI < 40) and (Volume > 100000)
(RSI < 30) and (PE > 0) and (PE < 15)
MarketCap > 5000000000 and Pct52wHigh > -5
```

### Phase 3 — Backtesting ✅
Vollständig vektorisiertes Backtesting ohne externe Bibliothek.
Strategien: RSI, Golden Cross, MACD Crossover, Bollinger Bands.
Ausgabe: Equity Curve, Trade-Marker, Sharpe, Drawdown, Win Rate.

### Phase 4 — PDF-Report-Generator ✅
A4-Report mit Chart, Kennzahlen, KI-Analyse, Backtesting via fpdf2.
Download-Button direkt in der Analyse-Seite.

## API-Keys

| Key | Wo holen | Wozu |
|-----|----------|------|
| `NEWSAPI_KEY` | newsapi.org (kostenlos) | Historische News |
| `GEMINI_KEY` | aistudio.google.com/apikey (kostenlos) | KI-Analyse |

## Hinweis
Alle Daten dienen nur zu Informationszwecken. Keine Anlageberatung.
