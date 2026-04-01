# ============================================================================
# modules/backtesting.py — High-Performance Backtesting mit VectorBT (Phase 3)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.utils import fmt_number, logger


# ============================================================================
# STRATEGIEN-LOGIK
# ============================================================================

def strategy_rsi_oversold(close: pd.Series, rsi_period=14, rsi_buy=30, rsi_sell=70):
    """RSI-Strategie: Kauf unter rsi_buy, Verkauf über rsi_sell."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    entries = rsi < rsi_buy
    exits = rsi > rsi_sell
    return entries, exits, rsi


def strategy_golden_cross(close: pd.Series, fast=20, slow=50):
    """Golden/Death Cross SMA-Strategie."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
    return entries, exits, sma_fast, sma_slow


def strategy_macd_cross(close: pd.Series):
    """MACD-Crossover-Strategie."""
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    entries = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    exits = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    return entries, exits, macd, signal


def strategy_bollinger_bands(close: pd.Series, window=20, std_dev=2.0):
    """Bollinger Bands Mean-Reversion-Strategie."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = sma - std_dev * std
    upper = sma + std_dev * std
    entries = close < lower
    exits = close > upper
    return entries, exits, upper, lower, sma


# ============================================================================
# BACKTEST-KERN (ohne externe Bibliothek — vektorisiert mit numpy/pandas)
# ============================================================================

def run_backtest(close: pd.Series, entries: pd.Series, exits: pd.Series,
                 initial_capital: float = 10000.0) -> dict:
    """
    Führt einen vektorisierten Backtest durch.
    Gibt Performance-Kennzahlen und Trade-Historie zurück.
    """
    close = close.dropna()
    entries = entries.reindex(close.index).fillna(False)
    exits = exits.reindex(close.index).fillna(False)

    equity = []
    cash = initial_capital
    position = 0.0
    entry_price = 0.0
    trades = []
    in_trade = False
    equity_curve = pd.Series(index=close.index, dtype=float)

    for i, (date, price) in enumerate(close.items()):
        if not in_trade and entries.loc[date]:
            # Einstieg: gesamtes Kapital investieren
            position = cash / price
            entry_price = price
            cash = 0.0
            in_trade = True
            trades.append({"entry_date": date, "entry_price": price})

        elif in_trade and exits.loc[date]:
            # Ausstieg
            cash = position * price
            pnl = cash - initial_capital if not trades[:-1] else cash - (trades[-1]["entry_price"] * position if trades else 0)
            if trades:
                trades[-1].update({
                    "exit_date": date, "exit_price": price,
                    "pnl": position * (price - entry_price),
                    "pnl_pct": (price / entry_price - 1) * 100
                })
            position = 0.0
            in_trade = False

        # Equity-Berechnung
        current_equity = cash + position * price
        equity_curve.loc[date] = current_equity

    # Letzter offener Trade
    if in_trade and not close.empty:
        last_price = close.iloc[-1]
        cash = position * last_price
        if trades and "exit_date" not in trades[-1]:
            trades[-1].update({
                "exit_date": close.index[-1], "exit_price": last_price,
                "pnl": position * (last_price - entry_price),
                "pnl_pct": (last_price / entry_price - 1) * 100,
                "open": True
            })

    final_equity = equity_curve.iloc[-1] if not equity_curve.empty else initial_capital
    completed_trades = [t for t in trades if "exit_date" in t]

    # Performance-Kennzahlen
    net_profit = final_equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0.0

    win_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]
    loss_trades = [t for t in completed_trades if t.get("pnl", 0) <= 0]
    win_rate = len(win_trades) / len(completed_trades) * 100 if completed_trades else 0

    avg_win = np.mean([t["pnl_pct"] for t in win_trades]) if win_trades else 0
    avg_loss = np.mean([t["pnl_pct"] for t in loss_trades]) if loss_trades else 0
    profit_factor = (sum(t["pnl"] for t in win_trades) /
                     abs(sum(t["pnl"] for t in loss_trades))) if loss_trades else float("inf")

    # Buy & Hold Vergleich
    bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 else 0

    return {
        "equity_curve": equity_curve,
        "trades": completed_trades,
        "metrics": {
            "net_profit": net_profit,
            "net_profit_pct": net_profit_pct,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": round(sharpe, 3),
            "win_rate": win_rate,
            "total_trades": len(completed_trades),
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "buy_hold_return": bh_return,
            "initial_capital": initial_capital,
            "final_equity": final_equity,
        }
    }


# ============================================================================
# PLOTLY EQUITY CURVE + TRADE-MARKER
# ============================================================================

def plot_equity_curve(equity_curve: pd.Series, close: pd.Series,
                      trades: list, ticker: str, strategy_name: str) -> go.Figure:
    """Erstellt interaktiven Plotly-Chart: Equity Curve + Kurschart mit Buy/Sell-Markern."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.04,
        subplot_titles=[f"Equity Curve — {strategy_name}", f"{ticker} Kursverlauf + Trades"]
    )

    # --- Equity Curve ---
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode='lines', name='Portfolio-Wert',
        line=dict(color='#26a69a', width=2),
        fill='tozeroy', fillcolor='rgba(38,166,154,0.1)'
    ), row=1, col=1)

    # Drawdown-Bereich
    peak = equity_curve.cummax()
    fig.add_trace(go.Scatter(
        x=peak.index, y=peak.values,
        mode='lines', name='Peak',
        line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'),
        showlegend=False
    ), row=1, col=1)

    # --- Kursverlauf ---
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        mode='lines', name=ticker,
        line=dict(color='#d1d4dc', width=1.5)
    ), row=2, col=1)

    # Buy-/Sell-Marker
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    for trade in trades:
        if trade.get("entry_date") and trade.get("entry_price"):
            buy_dates.append(trade["entry_date"])
            buy_prices.append(trade["entry_price"])
        if trade.get("exit_date") and trade.get("exit_price"):
            sell_dates.append(trade["exit_date"])
            sell_prices.append(trade["exit_price"])

    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode='markers', name='Kauf ▲',
            marker=dict(symbol='triangle-up', size=12, color='#26a69a',
                        line=dict(width=1, color='white'))
        ), row=2, col=1)
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode='markers', name='Verkauf ▼',
            marker=dict(symbol='triangle-down', size=12, color='#ef5350',
                        line=dict(width=1, color='white'))
        ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        font=dict(color='#d1d4dc'),
        height=550,
        legend=dict(orientation='h', y=1.02, x=0),
        margin=dict(l=0, r=10, t=40, b=0),
        xaxis2=dict(rangeslider=dict(visible=False)),
    )
    fig.update_yaxes(gridcolor='#1e2230')
    fig.update_xaxes(gridcolor='#1e2230')
    return fig


# ============================================================================
# DISPLAY
# ============================================================================

def display_backtesting():
    st.markdown("## Backtesting")
    st.caption("Strategie-Tests auf historischen Kursdaten · Vektorisierte Simulation")

    # --- Einstellungen ---
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_input = st.text_input("Ticker", value="AAPL", key="bt_ticker")
    with col2:
        period_options = {"6 Monate": 180, "1 Jahr": 365, "2 Jahre": 730,
                          "3 Jahre": 1095, "5 Jahre": 1825}
        period_sel = st.selectbox("Zeitraum", list(period_options.keys()),
                                  index=2, key="bt_period")
    with col3:
        initial_capital = st.number_input("Startkapital ($)", min_value=1000,
                                          max_value=10_000_000, value=10_000,
                                          step=1000, key="bt_capital")

    strategy_options = [
        "RSI Überverkauft/Überkauft",
        "Golden Cross (SMA 20/50)",
        "MACD Crossover",
        "Bollinger Bands Mean Reversion",
    ]
    strategy = st.selectbox("Strategie", strategy_options, key="bt_strategy")

    # Strategie-Parameter
    with st.expander("⚙ Strategie-Parameter"):
        if strategy == "RSI Überverkauft/Überkauft":
            rsi_period = st.slider("RSI Periode", 5, 30, 14)
            rsi_buy = st.slider("RSI Kauf (Überverkauft)", 10, 45, 30)
            rsi_sell = st.slider("RSI Verkauf (Überkauft)", 55, 90, 70)
        elif strategy in ["Golden Cross (SMA 20/50)"]:
            sma_fast = st.slider("SMA Schnell", 5, 50, 20)
            sma_slow = st.slider("SMA Langsam", 20, 200, 50)
        elif strategy == "Bollinger Bands Mean Reversion":
            bb_window = st.slider("BB Fenster", 10, 50, 20)
            bb_std = st.slider("BB Std-Abweichung", 1.0, 3.5, 2.0, step=0.1)

    if st.button("Backtest starten", type="primary", key="run_bt"):
        with st.spinner(f"Lade Kursdaten für {ticker_input}…"):
            try:
                end = datetime.today().date()
                start = end - timedelta(days=period_options[period_sel])
                df = yf.download(ticker_input, start=start, end=end,
                                 interval="1d", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.empty or len(df) < 60:
                    st.error(f"Zu wenige Daten für **{ticker_input}**. Bitte anderen Ticker oder längeren Zeitraum wählen.")
                    return
                close = df["Close"].squeeze()
            except Exception as e:
                st.error(f"Fehler beim Laden: {e}")
                return

        with st.spinner("Strategie wird simuliert…"):
            try:
                if strategy == "RSI Überverkauft/Überkauft":
                    entries, exits, rsi_series = strategy_rsi_oversold(
                        close, rsi_period, rsi_buy, rsi_sell)
                elif strategy == "Golden Cross (SMA 20/50)":
                    entries, exits, sma_f, sma_s = strategy_golden_cross(
                        close, sma_fast, sma_slow)
                elif strategy == "MACD Crossover":
                    entries, exits, macd_s, signal_s = strategy_macd_cross(close)
                elif strategy == "Bollinger Bands Mean Reversion":
                    entries, exits, bb_up, bb_lo, bb_mid = strategy_bollinger_bands(
                        close, bb_window, bb_std)

                result = run_backtest(close, entries, exits, float(initial_capital))
                st.session_state["bt_result"] = result
                st.session_state["bt_close"] = close
                st.session_state["bt_ticker"] = ticker_input
                st.session_state["bt_strategy"] = strategy
            except Exception as e:
                st.error(f"Backtest-Fehler: {e}")
                logger.exception(e)
                return

    # --- Ergebnisse anzeigen ---
    if "bt_result" in st.session_state:
        result = st.session_state["bt_result"]
        close = st.session_state["bt_close"]
        m = result["metrics"]
        ticker_sym = st.session_state.get("bt_ticker", "")
        strat_name = st.session_state.get("bt_strategy", "")

        st.markdown("---")
        st.markdown("### Ergebnisse")

        # Kern-Metriken
        col1, col2, col3, col4 = st.columns(4)
        net_color = "normal" if m["net_profit"] >= 0 else "inverse"
        col1.metric("Net Profit", f"${m['net_profit']:+,.2f}",
                    delta=f"{m['net_profit_pct']:+.1f}%", delta_color=net_color)
        col2.metric("Max Drawdown", f"{m['max_drawdown']:.1f}%")
        col3.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.3f}")
        col4.metric("Win Rate", f"{m['win_rate']:.1f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Trades gesamt", m["total_trades"])
        col6.metric("Ø Gewinn", f"{m['avg_win_pct']:.1f}%")
        col7.metric("Ø Verlust", f"{m['avg_loss_pct']:.1f}%")
        bh_color = "normal" if m["buy_hold_return"] >= 0 else "inverse"
        col8.metric("Buy & Hold", f"{m['buy_hold_return']:+.1f}%", delta_color=bh_color)

        # Profit Factor
        pf = m["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        st.markdown(f"**Profit Factor:** `{pf_str}` · "
                    f"**Startkapital:** `${m['initial_capital']:,.0f}` → "
                    f"**Endwert:** `${m['final_equity']:,.2f}`")

        # Vergleich mit Buy & Hold
        outperform = m["net_profit_pct"] - m["buy_hold_return"]
        icon = "▲" if outperform >= 0 else "▼"
        st.info(f"{icon} Strategie vs. Buy & Hold: **{outperform:+.1f}%** "
                f"({'Outperformance' if outperform >= 0 else 'Underperformance'})")

        # Equity Curve Chart
        fig = plot_equity_curve(
            result["equity_curve"], close,
            result["trades"], ticker_sym, strat_name
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trade-Tabelle
        if result["trades"]:
            st.markdown("#### Trade-Historie")
            trade_rows = []
            for t in result["trades"]:
                pnl = t.get("pnl", 0)
                trade_rows.append({
                    "Einstieg": str(t.get("entry_date", ""))[:10],
                    "Kauf": f"{t.get('entry_price', 0):.2f}",
                    "Ausstieg": str(t.get("exit_date", ""))[:10],
                    "Verkauf": f"{t.get('exit_price', 0):.2f}",
                    "PnL ($)": f"{pnl:+.2f}",
                    "PnL (%)": f"{t.get('pnl_pct', 0):+.1f}%",
                    "Status": "🟢 Offen" if t.get("open") else ("✓" if pnl > 0 else "✗"),
                })
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

            # CSV Export
            csv = pd.DataFrame(trade_rows).to_csv(index=False).encode("utf-8")
            st.download_button("Trade-Liste als CSV", csv,
                               f"backtest_{ticker_sym}.csv", "text/csv")
