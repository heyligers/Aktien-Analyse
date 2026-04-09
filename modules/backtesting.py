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
from modules.data_api import get_ticker
from modules.index_utils import get_full_universe
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# STRATEGIEN-LOGIK
# ============================================================================

def strategy_rsi_oversold(close: pd.Series, rsi_period=14, rsi_buy=30, rsi_sell=70):
    """RSI-Strategie: Kauf unter rsi_buy, Verkauf über rsi_sell. (Wilder's Smoothing)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, float("nan"))
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
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
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

def strategy_ttm_squeeze_pro(high: pd.Series, low: pd.Series, close: pd.Series, 
                         length=20, bb_mult=2.0, 
                         kc_mult_high=1.0, kc_mult_mid=1.5, kc_mult_low=2.0,
                         exit_days=2):
    """
    Exakte Replikation des 'Beardy Squeeze Pro' inklusive aller 3 Keltner Channel Level.
    """
    # 1. Bollinger Bands (BB)
    basis = close.rolling(length).mean()
    dev = bb_mult * close.rolling(length).std(ddof=0)
    bb_upper = basis + dev
    bb_lower = basis - dev

    # 2. Keltner Channels (KC)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
    dev_kc = tr.rolling(length).mean()
        
    # KC #1 (Faktor 1.0)
    kc_upper_high = basis + dev_kc * kc_mult_high
    kc_lower_high = basis - dev_kc * kc_mult_high
        
    # KC #2 (Faktor 1.5)
    kc_upper_mid = basis + dev_kc * kc_mult_mid
    kc_lower_mid = basis - dev_kc * kc_mult_mid
        
    # KC #3 (Faktor 2.0)
    kc_upper_low = basis + dev_kc * kc_mult_low
    kc_lower_low = basis - dev_kc * kc_mult_low

    # 3. Squeeze Conditions (wie im Pine Script)
    no_sqz = (bb_lower < kc_lower_low) | (bb_upper > kc_upper_low)      # GREEN
    low_sqz = (bb_lower >= kc_lower_low) & (bb_upper <= kc_upper_low)   # BLACK
    mid_sqz = (bb_lower >= kc_lower_mid) & (bb_upper <= kc_upper_mid)   # RED
    high_sqz = (bb_lower >= kc_lower_high) & (bb_upper <= kc_upper_high) # ORANGE
        
    # Optimierter Squeeze-Ausbruch: Wir nutzen den Mid-Kanal (1.5) statt dem extrem weiten (2.0)
    # um früher in den Trend einzusteigen (wie beim Standard TTM).
    no_sqz_early = (bb_lower < kc_lower_mid) | (bb_upper > kc_upper_mid)
    sqz_fired = (~no_sqz_early.shift(1).fillna(False)) & no_sqz_early

    # 4. Momentum Oscillator
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    avg_hl_basis = ((highest + lowest) / 2 + basis) / 2
    val = close - avg_hl_basis

    x_diff = np.arange(length) - (length - 1) / 2
    sum_x2 = length * (length**2 - 1) / 12
    weights = (1.0 / length) + (x_diff / sum_x2) * ((length - 1) / 2)
    mom = val.rolling(length).apply(lambda y: np.dot(y, weights), raw=True)

    # 5. ENTRY & EXIT Logik
    entries = sqz_fired & (mom > 0)
    
    # 6. EXITS
    # Klassisch: Null-Durchgang
    exit_classic = mom < 0
    # Aggressiv: Momentum-Abnahme für N aufeinanderfolgende Tage
    decreasing = mom < mom.shift(1)
    exit_aggressive = (decreasing.rolling(window=exit_days).sum() == exit_days) & (mom > 0)

    # Rückgabe erweitert
    return entries, exit_classic, exit_aggressive, mom, no_sqz, low_sqz, mid_sqz, high_sqz

def strategy_ema_cross(close: pd.Series, fast=8, slow=21):
    """EMA Crossover (kurzfristige Trendfolge)."""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    entries = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    exits = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    return entries, exits, ema_fast, ema_slow


def strategy_donchian_breakout(close: pd.Series, entry_window=20, exit_window=10):
    """Donchian Channel Breakout (Turtle Trading)."""
    highest_high = close.rolling(entry_window).max().shift(1)
    lowest_low = close.rolling(exit_window).min().shift(1)
    entries = close > highest_high
    exits = close < lowest_low
    return entries, exits, highest_high, lowest_low


def strategy_buy_and_hold(close: pd.Series):
    """Buy & Hold Investmentstrategie."""
    entries = pd.Series(False, index=close.index)
    if not entries.empty:
        entries.iloc[0] = True
    exits = pd.Series(False, index=close.index)
    return entries, exits


def _get_signals_internal(strategy, close, high, low, 
                          rsi_args=None, sma_args=None, ema_args=None, 
                          bb_args=None, ttm_args=None, donchian_args=None):
    """Interne Hilfsfunktion zur Signalberechnung für alle Strategien."""
    if strategy == "RSI Überverkauft/Überkauft":
        entries, exits, _ = strategy_rsi_oversold(close, *(rsi_args or (14, 30, 70)))
    elif strategy == "Golden Cross (SMA 20/50)":
        entries, exits, _, _ = strategy_golden_cross(close, *(sma_args or (20, 50)))
    elif strategy == "EMA Crossover":
        entries, exits, _, _ = strategy_ema_cross(close, *(ema_args or (8, 21)))
    elif strategy == "MACD Crossover":
        entries, exits, _, _ = strategy_macd_cross(close)
    elif strategy == "Bollinger Bands Mean Reversion":
        entries, exits, _, _, _ = strategy_bollinger_bands(close, *(bb_args or (20, 2.0)))
    elif strategy == "TTM Squeeze Pro Enhanced":
        # ttm_args: (length, bb_mult, kc_h, kc_m, kc_l, exit_days, agg_exit)
        args = ttm_args or (20, 2.0, 1.0, 1.5, 2.0, 2, False)
        ent, ex_c, ex_a, _, _, _, _, _ = strategy_ttm_squeeze_pro(high, low, close, *args[:6])
        entries = ent
        exits = ex_a if args[6] else ex_c
    elif strategy == "Donchian Breakout (Turtle)":
        entries, exits, _, _ = strategy_donchian_breakout(close, *(donchian_args or (20, 10)))
    elif strategy == "Buy & Hold (Einmalanlage)":
        entries, exits = strategy_buy_and_hold(close)
    else:
        entries = pd.Series(False, index=close.index)
        exits = pd.Series(False, index=close.index)
    
    return entries, exits


# ============================================================================
# BACKTEST-KERN (ohne externe Bibliothek — vektorisiert mit numpy/pandas)
# ============================================================================

def run_backtest(df: pd.DataFrame, entries: pd.Series, exits: pd.Series,
             initial_capital: float = 10000.0,
             take_profit_pct: float = 0.0, commission_pct: float = 0.1) -> dict:
    """
    Führt einen realistischen Backtest durch (Next-Day-Open Execution).
    - Signale heute -> Kauf/Verkauf zum Open von morgen.
    - Berücksichtigt Provisionen (commission_pct).
    - Optionaler Take-Profit (basiert auf Tageshoch).
    """
    df = df.dropna()
    close = df["Close"].squeeze()
    open_p = df["Open"].squeeze()
    entries = entries.reindex(close.index).fillna(False)
    exits = exits.reindex(close.index).fillna(False)

    equity_curve = pd.Series(index=close.index, dtype=float)
    cash = initial_capital
    position = 0.0
    entry_price = 0.0
    entry_comm_paid = 0.0
    total_commission = 0.0
    trades = []
    in_trade = False
    
    comm_rate = commission_pct / 100.0

    for i in range(len(df)):
        date = df.index[i]
        price_close = close.iloc[i]
        open_val = open_p.iloc[i]
        
        # 1. EXITS verarbeiten (Signal GESTERN -> Verkauf HEUTE OPEN)
        if i > 0 and in_trade and exits.iloc[i-1]:
            exit_p = open_val
            comm_exit = position * exit_p * comm_rate
            total_commission += comm_exit
            proceeds = position * exit_p - comm_exit
            cash = proceeds
            
            gross_cost = position * entry_price + entry_comm_paid
            pnl_actual = proceeds - gross_cost
            net_pct = (proceeds / gross_cost - 1) * 100 if gross_cost > 0 else 0
            
            if trades:
                trades[-1].update({
                    "exit_date": date, "exit_price": exit_p,
                    "pnl": pnl_actual,
                    "pnl_pct": (exit_p / entry_price - 1) * 100,
                    "net_pnl_pct": net_pct,
                    "exit_reason": "Signal"
                })
            position = 0.0
            in_trade = False

        # 2. ENTRIES verarbeiten (Signal GESTERN -> Kauf HEUTE OPEN)
        if i > 0 and not in_trade and entries.iloc[i-1]:
            entry_p = open_val
            if entry_p > 0:
                position = cash / (entry_p * (1 + comm_rate))
                comm_entry = position * entry_p * comm_rate
                total_commission += comm_entry
                entry_comm_paid = comm_entry
                entry_price = entry_p
                cash = 0.0
                in_trade = True
                trades.append({
                    "entry_date": date, "entry_price": entry_p, "exit_reason": ""
                })

        # 3. TAKE PROFIT INTRADAY (Check heutiges High)
        if in_trade and take_profit_pct > 0:
            high_p = df["High"].iloc[i]
            tp_price = entry_price * (1 + (take_profit_pct / 100))
            if high_p >= tp_price:
                # Same-day limits checking (vermeidet Lookahead auf Gaps, falls am selben Tag gekauft)
                is_same_day = trades and (trades[-1]["entry_date"] == date)
                exit_price = max(open_val, tp_price) if not is_same_day else tp_price
                
                comm_exit = position * exit_price * comm_rate
                total_commission += comm_exit
                proceeds = position * exit_price - comm_exit
                cash = proceeds
                
                gross_cost = position * entry_price + entry_comm_paid
                pnl_actual = proceeds - gross_cost
                net_pct = (proceeds / gross_cost - 1) * 100 if gross_cost > 0 else 0
                
                if trades:
                    trades[-1].update({
                        "exit_date": date, "exit_price": exit_price,
                        "pnl": pnl_actual,
                        "pnl_pct": (exit_price / entry_price - 1) * 100,
                        "net_pnl_pct": net_pct,
                        "exit_reason": "Take-Profit"
                    })
                position = 0.0
                in_trade = False

        # 4. Equity Calculation (based on Close of today)
        current_equity = cash + (position * price_close)
        equity_curve.iloc[i] = current_equity

    # End-of-period closing of open positions
    if in_trade and not close.empty:
        last_price = close.iloc[-1]
        gross_cost = position * entry_price + entry_comm_paid
        proceeds = position * last_price # Offene Positionen haben noch keine Exit-Gebühr
        pnl_actual = proceeds - gross_cost
        net_pct = (proceeds / gross_cost - 1) * 100 if gross_cost > 0 else 0
        cash = proceeds
        
        if trades and "exit_date" not in trades[-1]:
            trades[-1].update({
                "exit_date": close.index[-1], "exit_price": last_price,
                "pnl": pnl_actual,
                "pnl_pct": (last_price / entry_price - 1) * 100,
                "net_pnl_pct": net_pct,
                "open": True,
                "exit_reason": "Offen"
            })

    final_equity = equity_curve.iloc[-1] if not equity_curve.empty else initial_capital
    completed_trades = [t for t in trades if "exit_date" in t]

    # Performance-Kennzahlen
    net_profit = final_equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    returns = equity_curve.pct_change().dropna()
    ann_return = returns.mean() * 252 if len(returns) > 1 else 0.0
    rf = 0.04

    if len(returns) > 1 and returns.std() > 0:
        sharpe = (ann_return - rf) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino Ratio (nur negative Returns)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0.0
    sortino = (ann_return - rf) / downside_vol if downside_vol > 0 else 0.0

    # CAGR (Compound Annual Growth Rate)
    trading_days = len(equity_curve.dropna())
    years = trading_days / 252
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0.1 else net_profit_pct

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

    bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 else 0

    return {
        "equity_curve": equity_curve,
        "trades": completed_trades,
        "metrics": {
            "net_profit": net_profit, "net_profit_pct": net_profit_pct,
            "cagr": round(cagr, 2),
            "max_drawdown": max_drawdown, "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "win_rate": win_rate, "total_trades": len(completed_trades),
            "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor, "buy_hold_return": bh_return,
            "initial_capital": initial_capital, "final_equity": final_equity,
            "total_commission": total_commission,
        }
    }

def run_backtest_dca(close: pd.Series, monthly_amount: float = 100.0, 
                     commission_pct: float = 0.1) -> dict:
    """
    Simuliert einen monatlichen Sparplan (Dollar Cost Averaging).
    Kauft am ersten verfügbaren Handelstag jedes Monats inkl. Provision.
    """
    close = close.dropna()
    equity_curve = pd.Series(index=close.index, dtype=float)
    trades = []
    
    position = 0.0
    total_invested = 0.0
    total_commission = 0.0
    comm_rate = commission_pct / 100.0
    
    buy_dates = set(close.groupby([close.index.year, close.index.month]).head(1).index)
    
    for date, price in close.items():
        if date in buy_dates:
            # Provision beim Kauf abziehen
            comm = monthly_amount * comm_rate
            total_commission += comm
            shares_bought = (monthly_amount - comm) / price
            position += shares_bought
            total_invested += monthly_amount
            trades.append({
                "entry_date": date, 
                "entry_price": price, 
                "pnl": 0, "pnl_pct": 0, "open": True
            })
        
        current_equity = position * price
        equity_curve.loc[date] = current_equity
        
    final_equity = equity_curve.iloc[-1] if not equity_curve.empty else 0.0
    net_profit = final_equity - total_invested
    net_profit_pct = (net_profit / total_invested) * 100 if total_invested > 0 else 0
    
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0
        
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0.0
    
    bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 else 0

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": {
            "net_profit": net_profit,
            "net_profit_pct": net_profit_pct,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": round(sharpe, 3),
            "win_rate": 100.0 if net_profit > 0 else 0.0,
            "total_trades": len(buy_dates),
            "avg_win_pct": net_profit_pct,
            "avg_loss_pct": 0.0,
            "profit_factor": float("inf") if net_profit > 0 else 0.0,
            "buy_hold_return": bh_return,
            "initial_capital": total_invested,
            "final_equity": final_equity,
            "total_commission": total_commission,
        }
    }


def optimize_parameters(df: pd.DataFrame, strategy: str, initial_capital: float, commission: float) -> dict:
    """
    Führt Grid-Search Optimierung für die gewählte Strategie durch.
    Parallelisiert mit ThreadPoolExecutor für maximale Geschwindigkeit.
    """
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    tp_grid = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
    combinations = []

    if strategy == "RSI Überverkauft/Überkauft":
        for p in [7, 10, 14, 21]:
            for b in [20, 25, 30, 35]:
                for s in [65, 70, 75, 80]:
                    for tp in tp_grid:
                        combinations.append({"rsi_period": p, "rsi_buy": b, "rsi_sell": s, "tp": tp})

    elif strategy == "Golden Cross (SMA 20/50)":
        for f in [5, 10, 20, 30, 50]:
            for s in [50, 100, 150, 200]:
                if f < s:
                    for tp in tp_grid:
                        combinations.append({"sma_fast": f, "sma_slow": s, "tp": tp})

    elif strategy == "EMA Crossover":
        for f in [5, 8, 13, 21]:
            for s in [21, 34, 55, 100]:
                if f < s:
                    for tp in tp_grid:
                        combinations.append({"ema_fast": f, "ema_slow": s, "tp": tp})

    elif strategy == "Bollinger Bands Mean Reversion":
        for w in [10, 20, 30]:
            for d in [1.5, 1.7, 2.0, 2.2, 2.5]:
                for tp in tp_grid:
                    combinations.append({"bb_window": w, "bb_std": d, "tp": tp})

    elif strategy == "TTM Squeeze Pro Enhanced":
        for l in [10, 20, 30]:
            for m in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
                for agg in [True, False]:
                    days_grid = [1, 2, 3, 5, 8, 10] if agg else [1]
                    for d in days_grid:
                        for tp in tp_grid:
                            combinations.append({
                                "ttm_length": l, "ttm_kc_mid": m, "agg_exit": agg,
                                "ttm_exit_days": d, "tp": tp
                            })

    elif strategy == "Donchian Breakout (Turtle)":
        for en in [10, 20, 30, 40, 50]:
            for ex in [5, 10, 15, 20]:
                if en > ex:
                    for tp in tp_grid:
                        combinations.append({"entry_window": en, "exit_window": ex, "tp": tp})

    elif strategy == "MACD Crossover":
        st.warning("MACD hat keine optimierbaren Parameter.")
        return {}

    total = len(combinations)
    progress_bar = st.progress(0, text=f"Optimierung läuft (0/{total} Kombinationen)...")
    completed_count = [0]  # mutable für Closure

    def _run_combo(p):
        """Einzelne Paramter-Kombination ausführen (Thread-sicher)."""
        try:
            if strategy == "RSI Überverkauft/Überkauft":
                entries, exits, _ = strategy_rsi_oversold(close, p["rsi_period"], p["rsi_buy"], p["rsi_sell"])
            elif strategy == "Golden Cross (SMA 20/50)":
                entries, exits, _, _ = strategy_golden_cross(close, p["sma_fast"], p["sma_slow"])
            elif strategy == "EMA Crossover":
                entries, exits, _, _ = strategy_ema_cross(close, p["ema_fast"], p["ema_slow"])
            elif strategy == "Bollinger Bands Mean Reversion":
                entries, exits, _, _, _ = strategy_bollinger_bands(close, p["bb_window"], p["bb_std"])
            elif strategy == "TTM Squeeze Pro Enhanced":
                ent, ex_c, ex_a, _, _, _, _, _ = strategy_ttm_squeeze_pro(
                    high, low, close, p["ttm_length"], 2.0, 1.0, p["ttm_kc_mid"], 2.0, p.get("ttm_exit_days", 2))
                entries, exits = ent, (ex_a if p["agg_exit"] else ex_c)
            elif strategy == "Donchian Breakout (Turtle)":
                entries, exits, _, _ = strategy_donchian_breakout(close, p["entry_window"], p["exit_window"])
            else:
                return None

            res = run_backtest(df, entries, exits, initial_capital, take_profit_pct=p["tp"], commission_pct=commission)
            return {"params": p, "profit": res["metrics"]["net_profit_pct"], "metrics": res["metrics"]}
        except Exception:
            return None

    # Parallele Ausführung
    results_all = [None] * total
    batch_size = max(1, total // 20)  # Progress alle ~5%

    with ThreadPoolExecutor(max_workers=min(8, total)) as executor:
        futures = {executor.submit(_run_combo, p): i for i, p in enumerate(combinations)}
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            results_all[idx] = future.result()
            done_count += 1
            if done_count % batch_size == 0 or done_count == total:
                progress_bar.progress(done_count / total,
                                      text=f"Optimierung läuft ({done_count}/{total} Kombinationen)...")

    progress_bar.empty()

    best_profit = -float("inf")
    best_params = {}
    best_metrics = {}
    for r in results_all:
        if r and r["profit"] > best_profit:
            best_profit = r["profit"]
            best_params = r["params"]
            best_metrics = r["metrics"]

    return {"params": best_params, "metrics": best_metrics, "total_tested": total}


def run_multi_backtest(tickers: list, strategy: str, period_days: int, initial_capital: float, 
                      take_profit: float, commission: float, 
                      rsi_args=None, sma_args=None, ema_args=None, bb_args=None, 
                      ttm_args=None, donchian_args=None) -> dict:
    """
    Führt Backtests für eine Liste von Tickern parallel aus und aggregiert die Ergebnisse.
    """
    results = []
    end = datetime.today().date()
    start = end - timedelta(days=period_days)
    
    def _single_run(sym):
        try:
            df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Robustes Handling von Duplikaten (YFinance MultiIndex Problem)
            df = df.loc[:, ~df.columns.duplicated()]
            
            if df.empty or len(df) < 60:
                return None
            
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            
            if strategy == "RSI Überverkauft/Überkauft":
                entries, exits, _ = strategy_rsi_oversold(close, *rsi_args)
            elif strategy == "Golden Cross (SMA 20/50)":
                entries, exits, _, _ = strategy_golden_cross(close, *sma_args)
            elif strategy == "EMA Crossover":
                entries, exits, _, _ = strategy_ema_cross(close, *ema_args)
            elif strategy == "MACD Crossover":
                entries, exits, _, _ = strategy_macd_cross(close)
            elif strategy == "Bollinger Bands Mean Reversion":
                entries, exits, _, _, _ = strategy_bollinger_bands(close, *bb_args)
            elif strategy == "TTM Squeeze Pro Enhanced":
                ent, ex_c, ex_a, _, _, _, _, _ = strategy_ttm_squeeze_pro(high, low, close, *ttm_args[:6])
                exits = ex_a if ttm_args[6] else ex_c
                entries = ent
            elif strategy == "Donchian Breakout (Turtle)":
                entries, exits, _, _ = strategy_donchian_breakout(close, *donchian_args)
            elif strategy == "Buy & Hold (Einmalanlage)":
                entries, exits = strategy_buy_and_hold(close)
            else:
                return None
            
            res = run_backtest(df, entries, exits, initial_capital, take_profit_pct=take_profit, commission_pct=commission)
            res["ticker"] = sym
            return res
        except Exception as e:
            logger.error(f"Multi-BT Fehler für {sym}: {type(e).__name__}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(_single_run, s) for s in tickers]
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
                
    if not results:
        return {}
    
    # Aggregation
    avg_profit = np.mean([r["metrics"]["net_profit_pct"] for r in results])
    avg_win_rate = np.mean([r["metrics"]["win_rate"] for r in results])
    avg_drawdown = np.mean([r["metrics"]["max_drawdown"] for r in results])
    total_trades = sum([r["metrics"]["total_trades"] for r in results])
    
    return {
        "status": "success",
        "results": results,
        "metrics": {
            "avg_profit": avg_profit,
            "avg_win_rate": avg_win_rate,
            "avg_drawdown": avg_drawdown,
            "total_trades": total_trades,
            "num_tickers": len(results)
        }
    }


def optimize_multi_ticker(tickers: list, strategy: str, period_days: int, 
                         initial_capital: float, commission: float) -> dict:
    """
    Optimiert Strategie-Parameter über ein ganzes Aktien-Universum.
    """
    end = datetime.today().date()
    start = end - timedelta(days=period_days)
    
    # 1. Daten für alle Ticker vorab laden (Cache)
    ticker_data = {}
    st.info(f"Lade Kursdaten für {len(tickers)} Ticker...")
    for s in tickers:
        try:
            df = yf.download(s, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > 60:
                ticker_data[s] = df
        except Exception:
            continue
            
    if not ticker_data:
        return {"error": "Keine Daten verfügbar."}
    
    # 2. Grid Definition (FULL GRID for granular results)
    tp_grid = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
    combinations = []
    
    if strategy == "RSI Überverkauft/Überkauft":
        for p in [7, 10, 14, 21]:
            for b in [20, 25, 30, 35]:
                for s in [65, 70, 75, 80]:
                    for tp in tp_grid:
                        combinations.append({"rsi_period": p, "rsi_buy": b, "rsi_sell": s, "tp": tp})
    
    elif strategy == "Golden Cross (SMA 20/50)":
        for f in [5, 10, 20, 30, 50]:
            for s in [50, 100, 150, 200]:
                if f < s:
                    for tp in tp_grid:
                        combinations.append({"sma_fast": f, "sma_slow": s, "tp": tp})

    elif strategy == "EMA Crossover":
        for f in [5, 8, 13, 21]:
            for s in [21, 34, 55, 100]:
                if f < s:
                    for tp in tp_grid:
                        combinations.append({"ema_fast": f, "ema_slow": s, "tp": tp})

    elif strategy == "TTM Squeeze Pro Enhanced":
        # Verfeinerte KC_MID Suche (0.1 Schritte)
        for l in [10, 20, 30]:
            for m in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
                for agg in [True, False]:
                    days_grid = [1, 2, 3, 5, 8, 10] if agg else [1]
                    for d in days_grid:
                        for tp in tp_grid:
                            combinations.append({
                                "ttm_length": l, "ttm_kc_mid": m, "agg_exit": agg, 
                                "ttm_exit_days": d, "tp": tp
                            })
    
    elif strategy == "Donchian Breakout (Turtle)":
        for en in [10, 20, 30, 40, 50]:
            for ex in [5, 10, 15, 20]:
                if en > ex:
                    for tp in tp_grid:
                        combinations.append({"entry_window": en, "exit_window": ex, "tp": tp})
    else:
        # Fallback wenn Strategie noch nicht für Multi-Opt vorbereitet ist
        return {"error": "Strategie noch nicht für Multi-Ticker-Optimierung vorbereitet."}

    best_score = -float("inf")
    best_params = {}
    best_metrics = {}
    
    total = len(combinations)
    prog = st.progress(0, text=f"Universum-Optimierung (0/{total})")
    
    for i, p in enumerate(combinations):
        prog.progress((i+1)/total, text=f"Testen: {p} ({i+1}/{total})")
        
        # Berechne Metriken für alle Ticker mit diesem Parameter-Set
        batch_profits = []
        batch_trades = 0
        for s, df in ticker_data.items():
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            
            if strategy == "RSI Überverkauft/Überkauft":
                ent, ext, _ = strategy_rsi_oversold(close, p["rsi_period"], p["rsi_buy"], p["rsi_sell"])
            elif strategy == "Golden Cross (SMA 20/50)":
                ent, ext, _, _ = strategy_golden_cross(close, p["sma_fast"], p["sma_slow"])
            elif strategy == "EMA Crossover":
                ent, ext, _, _ = strategy_ema_cross(close, p["ema_fast"], p["ema_slow"])
            elif strategy == "Bollinger Bands Mean Reversion":
                ent, ext, _, _, _ = strategy_bollinger_bands(close, p["bb_window"], p["bb_std"])
            elif strategy == "TTM Squeeze Pro Enhanced":
                e, ex_c, ex_a, _, _, _, _, _ = strategy_ttm_squeeze_pro(high, low, close, p["ttm_length"], 2.0, 1.0, p["ttm_kc_mid"], 2.0, p["ttm_exit_days"])
                ent, ext = e, (ex_a if p["agg_exit"] else ex_c)
            elif strategy == "Donchian Breakout (Turtle)":
                ent, ext, _, _ = strategy_donchian_breakout(close, p["entry_window"], p["exit_window"])
            else:
                continue
            
            res = run_backtest(df, ent, ext, initial_capital, p["tp"], commission)
            batch_profits.append(res["metrics"]["net_profit_pct"])
            batch_trades += res["metrics"]["total_trades"]
            
        avg_p = np.mean(batch_profits)
        # Score-Mechanismus: Profitabilität gewichtet mit Trade-Aktivität
        # Wenn 0 Trades stattfanden, ist der Score extrem niedrig (bestraft Inaktivität).
        score = avg_p if batch_trades > 0 else -1000.0
        
        if score > best_score:
            best_score = score
            best_params = p
            best_metrics = {
                "net_profit_pct": avg_p,
                "avg_profit": avg_p,     
                "num_tickers": len(batch_profits),
                "total_trades": batch_trades
            }
            
    prog.empty()
    return {"params": best_params, "metrics": best_metrics, "total_tested": total}


# ============================================================================
# PLOTLY EQUITY CURVE + TRADE-MARKER
# ============================================================================

def plot_equity_curve(equity_curve: pd.Series, close: pd.Series,
                      trades: list, ticker: str, strategy_name: str) -> go.Figure:
    """Erstellt interaktiven Plotly-Chart: Equity Curve + Kurschart mit Buy/Sell-Markern."""
    show_equity = not equity_curve.empty
    
    if show_equity:
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
        
        price_row = 2
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{ticker} Kursverlauf + Trades"])
        price_row = 1

    # --- Kursverlauf ---
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        mode='lines', name=ticker,
        line=dict(color='#d1d4dc', width=1.5)
    ), row=price_row, col=1)

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
        ), row=price_row, col=1)
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode='markers', name='Verkauf ▼',
            marker=dict(symbol='triangle-down', size=12, color='#ef5350',
                        line=dict(width=1, color='white'))
        ), row=price_row, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        font=dict(color='#d1d4dc'),
        height=550,
        legend=dict(orientation='h', y=1.02, x=0),
        margin=dict(l=0, r=10, t=40, b=0),
    )
    if show_equity:
        fig.update_layout(xaxis2=dict(rangeslider=dict(visible=False)))
    fig.update_yaxes(gridcolor='#1e2230')
    fig.update_xaxes(gridcolor='#1e2230')
    return fig


def run_backtest_with_params(ticker_input, period_sel, period_options, strategy, initial_capital, take_profit, commission, 
                             rsi_args=None, sma_args=None, ema_args=None, bb_args=None, ttm_args=None, donchian_args=None, 
                             monthly_amount=200, universe_mode=False, universe_choice=None):
    """Hilfsfunktion für die Backtest-Ausführung."""
    if universe_mode and universe_choice:
        with st.spinner(f"Führe Multi-Ticker Backtest für {universe_choice} aus..."):
            if universe_choice == "Screener Ergebnisse":
                screener_res = st.session_state.get("screener_results", [])
                universe_tickers = [r["ticker"] for r in screener_res if r.get("ticker")]
                if not universe_tickers:
                    st.error("Keine gültigen Ticker in den Screener-Ergebnissen gefunden.")
                    return
            else:
                universe_tickers = get_full_universe(universe_choice)
                
            if not universe_tickers:
                st.error("Universum konnte nicht geladen werden.")
                return
            result = run_multi_backtest(universe_tickers, strategy, period_options[period_sel], initial_capital, 
                                       take_profit, commission, rsi_args, sma_args, ema_args, 
                                       bb_args, ttm_args, donchian_args)
            st.session_state["bt_multi_result"] = result
            st.session_state["bt_run_strategy"] = strategy
            st.session_state.pop("bt_result", None) # Einzel-Ergebnis löschen
            return

    with st.spinner(f"Lade Kursdaten für {ticker_input}…"):
        try:
            end = datetime.today().date()
            start = end - timedelta(days=period_options[period_sel])
            df = yf.download(ticker_input, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Robustes Handling von Duplikaten
            df = df.loc[:, ~df.columns.duplicated()]
            
            if df.empty or len(df) < 60:
                st.error(f"Zu wenige Daten für **{ticker_input}**. Bitte anderen Ticker oder längeren Zeitraum wählen.")
                return
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")
            return

    with st.spinner("Strategie wird simuliert…"):
        try:
            if strategy == "Sparplan (DCA)":
                result = run_backtest_dca(close, monthly_amount, commission_pct=commission)
            else:
                if strategy == "RSI Überverkauft/Überkauft":
                    entries, exits, _ = strategy_rsi_oversold(close, *rsi_args)
                elif strategy == "Golden Cross (SMA 20/50)":
                    entries, exits, _, _ = strategy_golden_cross(close, *sma_args)
                elif strategy == "EMA Crossover":
                    entries, exits, _, _ = strategy_ema_cross(close, *ema_args)
                elif strategy == "MACD Crossover":
                    entries, exits, _, _ = strategy_macd_cross(close)
                elif strategy == "Bollinger Bands Mean Reversion":
                    entries, exits, _, _, _ = strategy_bollinger_bands(close, *bb_args)
                elif strategy == "TTM Squeeze Pro Enhanced":
                    # ttm_args format: (length, bb_mult, kc_high, kc_mid, kc_low, exit_days, agg_exit)
                    ent, ex_c, ex_a, _, _, _, _, _ = strategy_ttm_squeeze_pro(high, low, close, *ttm_args[:6])
                    agg_exit = ttm_args[6]
                    exits = ex_a if agg_exit else ex_c
                    entries = ent
                elif strategy == "Donchian Breakout (Turtle)":
                    entries, exits, _, _ = strategy_donchian_breakout(close, *donchian_args)
                elif strategy == "Buy & Hold (Einmalanlage)":
                    entries, exits = strategy_buy_and_hold(close)

                result = run_backtest(df, entries, exits, float(initial_capital), 
                                     take_profit_pct=take_profit,
                                     commission_pct=commission)
                
            st.session_state["bt_result"] = result
            st.session_state["bt_close"] = close
            st.session_state["bt_run_ticker"] = ticker_input
            st.session_state["bt_run_strategy"] = strategy
        except Exception as e:
            st.error(f"Backtest-Fehler: {e}")
            logger.exception(e)


# ============================================================================
# DISPLAY
# ============================================================================

def display_backtesting():
    st.markdown("## Backtesting")
    st.caption("Strategie-Tests auf historischen Kursdaten · Vektorisierte Simulation")

    tab_single, tab_multi = st.tabs(["Einzel-Ticker", "Universum-Backtest"])

    with tab_single:
        display_single_ticker_backtesting()

    with tab_multi:
        display_portfolio_backtesting()


def display_single_ticker_backtesting():
    """Bestehende Einzel-Ticker Backtest-Logik (verschoben in Unterfunktion)."""

    # --- Einstellungen ---
    col1, col2, col3 = st.columns(3)
    with col1:
        user_input = st.text_input("Name oder Ticker", value="AAPL", key="bt_ticker")
        ticker_input = get_ticker(user_input)
        st.caption(f"Ausgewähltes Symbol: **{ticker_input}**")
    with col2:
        period_options = {
            "6 Monate": 180, "1 Jahr": 365, "2 Jahre": 730,
            "3 Jahre": 1095, "5 Jahre": 1825, "10 Jahre": 3650,
            "15 Jahre": 5475, "20 Jahre": 7300
        }
        period_sel = st.selectbox("Zeitraum", list(period_options.keys()),
                                  index=2, key="bt_period")
    with col3:
        initial_capital = st.number_input("Startkapital ($)", min_value=1000,
                                          max_value=10_000_000, value=10_000,
                                          step=1000, key="bt_capital")

    strategy_options = [
        "RSI Überverkauft/Überkauft",
        "Golden Cross (SMA 20/50)",
        "EMA Crossover",
        "MACD Crossover",
        "Bollinger Bands Mean Reversion",
        "TTM Squeeze Pro Enhanced",
        "Donchian Breakout (Turtle)",
        "Buy & Hold (Einmalanlage)",
        "Sparplan (DCA)",
    ]
    strategy = st.selectbox("Strategie", strategy_options, key="bt_strategy")

    # --- Universum Auswahl (für Multi-Ticker) ---
    with st.expander(" Universum Filter (Multi-Ticker)"):
        # Pre-selection from session state (if jumping from screener)
        def_mode = st.session_state.get("bt_universe_mode", False)
        def_choice = st.session_state.get("bt_universe_choice", "DAX 40")
        
        universe_mode = st.toggle("Multi-Ticker Modus aktivieren", value=def_mode, help="Backtestet die Strategie auf einem ganzen Universum.")
        
        # New: Added 'Screener Ergebnisse'
        universe_options = ["DAX 40", "S&P 500 (Full)", "Nasdaq 100 (Full)", "Screener Ergebnisse"]
        if def_choice not in universe_options:
            def_choice = "DAX 40"
            
        universe_choice = st.selectbox("Ziel-Universum", universe_options, 
                                       index=universe_options.index(def_choice))
        st.caption("Hinweis: Multi-Ticker Backtests laden Daten für viele Aktien. Dies kann länger dauern.")

    # Strategie-Parameter
    with st.expander(" Strategie-Parameter"):
        take_profit = st.slider("Take-Profit (%) [0 = Deaktiviert]", 0.0, 100.0, 0.0, step=0.5)
        commission = st.slider("Provision / Gebühr (%)", 0.0, 5.0, 0.1, step=0.05, 
                               help="Handels-Kommission pro Transaktion (Kauf und Verkauf).")
        st.markdown("---")
        if strategy == "RSI Überverkauft/Überkauft":
            rsi_period = st.slider("RSI Periode", 5, 30, 14)
            rsi_buy = st.slider("RSI Kauf (Überverkauft)", 10, 45, 30)
            rsi_sell = st.slider("RSI Verkauf (Überkauft)", 55, 90, 70)
        elif strategy in ["Golden Cross (SMA 20/50)"]:
            sma_fast = st.slider("SMA Schnell", 5, 50, 20)
            sma_slow = st.slider("SMA Langsam", 20, 200, 50)
        elif strategy == "EMA Crossover":
            ema_fast = st.slider("EMA Schnell", 5, 20, 8)
            ema_slow = st.slider("EMA Langsam", 20, 100, 21)
        elif strategy == "Bollinger Bands Mean Reversion":
            bb_window = st.slider("BB Fenster", 10, 50, 20)
            bb_std = st.slider("BB Std-Abweichung", 1.0, 3.5, 2.0, step=0.1)
        elif strategy == "TTM Squeeze Pro Enhanced":
            ttm_length = st.slider("TTM Squeeze Länge", 10, 50, 20)
            ttm_bb_mult = st.slider("Bollinger Band Std. Abw.", 1.0, 4.0, 2.0, step=0.1)
            ttm_agg_exit = st.toggle("Aggressiver Ausstieg (Momentum Cooling)", True, 
                                     help="Verkauft sobald das Momentum nachlässt, statt auf Null-Durchgang zu warten.")
            if ttm_agg_exit:
                ttm_exit_days = st.slider("Bestätigungstage (Ausstieg)", 1, 10, 2, 
                                          help="Anzahl aufeinanderfolgender Tage, an denen das Momentum sinken muss.")
            else:
                ttm_exit_days = 1
            st.markdown("**Keltner Channels**")
            ttm_kc_high = st.slider("Keltner Channel #1 (High)", 0.5, 3.0, 1.0, step=0.1)
            ttm_kc_mid = st.slider("Keltner Channel #2 (Mid)", 0.5, 3.0, 1.5, step=0.1)
            ttm_kc_low = st.slider("Keltner Channel #3 (Low - Signalgeber)", 0.5, 4.0, 2.0, step=0.1)
        elif strategy == "Donchian Breakout (Turtle)":
            entry_window = st.slider("Breakout Fenster (Einstieg)", 10, 50, 20)
            exit_window = st.slider("Breakout Fenster (Ausstieg)", 5, 30, 10)
        elif strategy == "Buy & Hold (Einmalanlage)":
            st.info("Kauft am ersten Tag und hält das gesamte Kapital bis zum Ende (Einmalanlage).")
        elif strategy == "Sparplan (DCA)":
            monthly_amount = st.slider("Monatliche Sparrate ($)", 50, 2000, 200, step=50)
            st.info("Kauft am ersten Handelstag jedes Monats für den gewählten Betrag.")

        # --- Optimierung ---
        st.markdown("---")
        if strategy not in ["Buy & Hold (Einmalanlage)", "Sparplan (DCA)"]:
            if st.button("Parameter optimieren (Einzel-Ticker)", use_container_width=True, help="Sucht die profitabelste Kombination für den aktuellen Ticker."):
                st.session_state["run_optimization"] = True
            
            if st.button("Universe-Optimierung (Multi-Ticker)", use_container_width=True, help="Sucht die besten Parameter für das gesamte Universum."):
                st.session_state["run_universe_optimization"] = True
                st.session_state["bt_opt_universe"] = universe_choice

    if st.button("Backtest starten", type="primary", key="run_bt") or st.session_state.get("bt_run_immediately", False):
        if st.session_state.get("bt_run_immediately"):
            st.session_state["bt_run_immediately"] = False
            
        st.session_state["run_optimization"] = False # Normalen Lauf triggern
        st.session_state.pop("bt_opt_result", None)
        st.session_state.pop("bt_multi_result", None)
        st.session_state.pop("mc_result", None) # Altes Monte Carlo löschen
        st.session_state["mc_expanded_state"] = False # Expander beim neuen Backtest schließen
        run_backtest_with_params(ticker_input, period_sel, period_options, strategy, initial_capital, take_profit, commission, 
                                 rsi_args=(rsi_period, rsi_buy, rsi_sell) if strategy == "RSI Überverkauft/Überkauft" else None,
                                 sma_args=(sma_fast, sma_slow) if strategy == "Golden Cross (SMA 20/50)" else None,
                                 ema_args=(ema_fast, ema_slow) if strategy == "EMA Crossover" else None,
                                 bb_args=(bb_window, bb_std) if strategy == "Bollinger Bands Mean Reversion" else None,
                                 ttm_args=(ttm_length, ttm_bb_mult, ttm_kc_high, ttm_kc_mid, ttm_kc_low, ttm_exit_days, ttm_agg_exit) if strategy == "TTM Squeeze Pro Enhanced" else None,
                                 donchian_args=(entry_window, exit_window) if strategy == "Donchian Breakout (Turtle)" else None,
                                 monthly_amount=monthly_amount if strategy == "Sparplan (DCA)" else 200,
                                 universe_mode=universe_mode, universe_choice=universe_choice)

    if st.session_state.get("run_optimization"):
        st.session_state["run_optimization"] = False
        with st.spinner(f"Optimiere {strategy} für {ticker_input}..."):
            try:
                # Daten laden
                end = datetime.today().date()
                start = end - timedelta(days=period_options[period_sel])
                df_opt = yf.download(ticker_input, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
                if isinstance(df_opt.columns, pd.MultiIndex):
                    df_opt.columns = df_opt.columns.get_level_values(0)
                
                if not df_opt.empty:
                    opt_res = optimize_parameters(df_opt, strategy, float(initial_capital), commission)
                    st.session_state["bt_opt_result"] = opt_res
                    st.success(f"Optimierung abgeschlossen! Beste Rendite: **{opt_res['metrics']['net_profit_pct']:+.1f}%**")
                else:
                    st.error("Keine Daten für Optimierung gefunden.")
            except Exception as e:
                st.error(f"Optimierungsfehler: {e}")
    
    if st.session_state.get("run_universe_optimization"):
        st.session_state["run_universe_optimization"] = False
        sel_u = st.session_state.get("bt_opt_universe", "DAX 40")
        
        if sel_u == "Screener Ergebnisse":
            screener_res = st.session_state.get("screener_results", [])
            universe_tickers = [r["ticker"] for r in screener_res]
        else:
            universe_tickers = get_full_universe(sel_u)

        if not universe_tickers:
            st.error(f"Universum {sel_u} konnte nicht geladen werden.")
        else:
            with st.spinner(f"Optimiere {strategy} für {sel_u} Universum ({len(universe_tickers)} Aktien)..."):
                try:
                    res_u = optimize_multi_ticker(universe_tickers, strategy, period_options[period_sel], float(initial_capital), commission)
                    if "error" in res_u:
                        st.error(res_u["error"])
                    else:
                        st.session_state["bt_opt_result"] = res_u
                        st.success(f"Universe-Optimierung fertig! Beste Ø-Rendite: **{res_u['metrics']['avg_profit']:+.1f}%**")
                except Exception as e:
                    st.error(f"Universe-Optimierungsfehler: {e}")
                    logger.exception(e)

    # Best-Case Anzeige
    if "bt_opt_result" in st.session_state:
        res = st.session_state["bt_opt_result"]
        params = res["params"]
        
        with st.expander("**Beste Parameter gefunden**", expanded=True):
            cols = st.columns(len(params))
            for i, (k, v) in enumerate(params.items()):
                cols[i].metric(k.replace("_", " ").title(), f"{v}")
            
            profit_val = res["metrics"].get("net_profit_pct") or res["metrics"].get("avg_profit") or 0.0
            trades_val = res["metrics"].get("total_trades") or 0
            
            st.info(f"Diese Kombination erzielte **{profit_val:+.1f}%** Rendite "
                    f"bei **{trades_val}** Trades (getestet: {res['total_tested']} Kombis).")

            # ITEM #6: Overfitting-Warnung
            st.warning(
                "⚠️ **Overfitting-Warnung:** Der Grid Search optimiert auf dem gesamten Trainingszeitraum. "
                "Die gefundenen Parameter könnten auf historischen Daten überfit sein. "
                "Validiere sie bitte auf einem anderen (neueren) Zeitraum, bevor du sie live einsetzt.",
                icon="⚠️"
            )

            if st.button("Diese Parameter übernehmen & Backtest starten", type="primary"):
                # Parametern in Session State oder als Trigger für neuen Lauf nutzen
                # Einfachster Weg: Direkt neue Session State Variablen setzen, die die Slider überschreiben würden
                # Aber hier führen wir den Backtest einfach direkt aus:
                st.session_state.pop("bt_result", None)
                # (Wir müssten hier den Backtest mit den neuen Params triggern)
                st.session_state["bt_trigger_opt_apply"] = True

    if st.session_state.get("bt_trigger_opt_apply"):
        st.session_state["bt_trigger_opt_apply"] = False
        res = st.session_state["bt_opt_result"]
        p = res["params"]
        
        # Mapping der Parameter für run_backtest_with_params
        rsi_args = (int(p['rsi_period']), int(p['rsi_buy']), int(p['rsi_sell'])) if 'rsi_period' in p else None
        sma_args = (int(p['sma_fast']), int(p['sma_slow'])) if 'sma_fast' in p else None
        ema_args = (int(p['ema_fast']), int(p['ema_slow'])) if 'ema_fast' in p else None
        bb_args = (int(p['bb_window']), float(p['bb_std'])) if 'bb_window' in p else None
        ttm_args = (int(p['ttm_length']), 2.0, 1.0, float(p['ttm_kc_mid']), 2.0, int(p.get('ttm_exit_days', 2)), p['agg_exit']) if 'ttm_length' in p else None
        donchian_args = (int(p['entry_window']), int(p['exit_window'])) if 'entry_window' in p else None
        
        run_backtest_with_params(ticker_input, period_sel, period_options, strategy, initial_capital, 
                                 float(p.get('tp', 0)), commission,
                                 rsi_args=rsi_args, sma_args=sma_args, ema_args=ema_args,
                                 bb_args=bb_args, ttm_args=ttm_args, donchian_args=donchian_args,
                                 monthly_amount=monthly_amount if strategy == "Sparplan (DCA)" else 200,
                                 universe_mode=universe_mode, universe_choice=universe_choice)

    # --- Ergebnisse anzeigen (Multi-Ticker) ---
    if "bt_multi_result" in st.session_state:
        res_m = st.session_state["bt_multi_result"]
        if not res_m:
            st.warning("Keine Ergebnisse für Multi-Ticker Backtest.")
        else:
            m = res_m["metrics"]
            st.markdown("---")
            st.markdown(f"Aggregierte Ergebnisse ({m['num_tickers']} Aktien)")
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Ø Rendite %", f"{m['avg_profit']:+.1f}%")
            mc2.metric("Ø Win Rate", f"{m['avg_win_rate']:.1f}%")
            mc3.metric("Ø Max Drawdown", f"{m['avg_drawdown']:.1f}%")
            mc4.metric("Trades gesamt", m["total_trades"])
            
            # Einzelticker-Tabelle
            with st.expander("Einzelergebnisse der Aktien"):
                rows = []
                for r in sorted(res_m["results"], key=lambda x: x["metrics"]["net_profit_pct"], reverse=True):
                    rm = r["metrics"]
                    rows.append({
                        "Ticker": r["ticker"],
                        "Profit %": f"{rm['net_profit_pct']:+.1f}%",
                        "Drawdown": f"{rm['max_drawdown']:.1f}%",
                        "Win Rate": f"{rm['win_rate']:.1f}%",
                        "Trades": rm["total_trades"]
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Ergebnisse anzeigen ---
    if "bt_result" in st.session_state:
        result = st.session_state["bt_result"]
        close = st.session_state["bt_close"]
        m = result["metrics"]
        ticker_sym = st.session_state.get("bt_run_ticker", "")
        strat_name = st.session_state.get("bt_run_strategy", "")

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

        # Erweiterte Metriken: CAGR + Sortino
        col9, col10, col11, _ = st.columns(4)
        col9.metric("CAGR", f"{m.get('cagr', 0):+.1f}%",
                    help="Compound Annual Growth Rate — jährliche Wachstumsrate")
        col10.metric("Sortino Ratio", f"{m.get('sortino_ratio', 0):.3f}",
                     help="Wie Sharpe, aber bewertet nur negative Volatilität")
        col11.metric("Profit Factor", f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞")

        # Profit Factor
        pf = m["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        total_comm = m.get("total_commission", 0.0)
        st.markdown(f"**Profit Factor:** `{pf_str}` · "
                    f"**Gezahlte Provisionen:** `${total_comm:,.2f}` · "
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
                
                exit_d = str(t.get("exit_date", ""))[:10] if t.get("exit_date") else "-"
                exit_p = f"{t.get('exit_price'):.2f}" if t.get("exit_price") is not None else "-"
                
                trade_rows.append({
                    "Einstieg": str(t.get("entry_date", ""))[:10],
                    "Kauf": f"{t.get('entry_price', 0):.2f}",
                    "Ausstieg": exit_d,
                    "Verkauf": exit_p,
                    "Grund": t.get("exit_reason", ""),
                    "PnL ($)": f"{pnl:+.2f}",
                    "PnL (%)": f"{t.get('pnl_pct', 0):+.1f}%",
                    "Status": " Offen" if t.get("open") else ("✓" if pnl > 0 else "✗"),
                })
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

            # CSV Export
            csv = pd.DataFrame(trade_rows).to_csv(index=False).encode("utf-8")
            st.download_button("Trade-Liste als CSV", csv,
                               f"backtest_{ticker_sym}.csv", "text/csv")

        # Monte Carlo Simulation (Item #8)
        if result["trades"] and len(result["trades"]) >= 5:
            st.markdown("---")
            # Wir erzwingen, dass der Expander nach der Simulation offen bleibt
            mc_expanded = st.session_state.get("mc_expanded_state", False) or "mc_result" in st.session_state
            with st.expander(" Monte Carlo Simulation", expanded=mc_expanded):
                n_sims = st.slider("Anzahl Simulationen", 500, 10000, 5000, step=500, key="mc_n_sims")
                st.markdown(
                    f"Simuliert **{n_sims:,}** zufällige Trade-Kombinationen (Bootstrapping), "
                    "um die Robustheit der Strategie gegenüber Sequenz-Risiken zu prüfen."
                )
                if st.button("Monte Carlo starten", type="primary", key="run_mc"):
                    with st.spinner(f"{n_sims} Simulationen laufen..."):
                        mc = run_monte_carlo_simulation(
                            result["trades"], m["initial_capital"], n_sims=n_sims
                        )
                    st.session_state["mc_result"] = mc
                    st.session_state["mc_expanded_state"] = True
                    st.rerun() # Sofortiger Rerun, damit show_mc oben beim nächsten Durchlauf True ist

                mc = st.session_state.get("mc_result")
                if mc:
                    _display_monte_carlo(mc, m["initial_capital"])


# ============================================================================
# MONTE CARLO SIMULATION (Item #8)
# ============================================================================

def run_monte_carlo_simulation(trades: list, initial_capital: float,
                               n_sims: int = 10000) -> dict:
    """
    Führt eine Monte Carlo Simulation durch (Bootstrapping).
    Zieht zufällig Trades aus der Historie (mit Zurücklegen) und simuliert 
    das Wachstum des Kapitals durch Compounding (prozentual).

    Args:
        trades: Liste der abgeschlossenen Trades aus run_backtest()
        initial_capital: Startkapital
        n_sims: Anzahl Simulationen (Standard: 1000)
    Returns:
        dict mit Perzentil-Kurven, Final-Equity-Verteilung und Schlüsseldaten
    """
    # Wir bevorzugen net_pnl_pct (inkl. Gebühren) für realistisches Compounding
    ret_list = [t["net_pnl_pct"] for t in trades if "net_pnl_pct" in t]
    
    # Ausweichen auf pnl_pct falls net_pnl_pct nicht da ist (Legacy Support)
    if not ret_list:
        ret_list = [t["pnl_pct"] for t in trades if "pnl_pct" in t]

    if len(ret_list) < 2:
        return {}

    n_trades = len(ret_list)
    ret_arr = np.array(ret_list) / 100.0  # In Dezimal umwandeln (z.B. 0.05 für 5%)
    rng = np.random.default_rng()  # Kein fixer Seed mehr für echte Varianz pro User-Klick (oder weglassen)

    # n_sims x n_trades Matrix mit zufällig gezogenen Renditen (mit Zurücklegen = Bootstrapping)
    # Damit erhalten wir eine echte Verteilung der Endkapitalien.
    sim_returns = rng.choice(ret_arr, size=(n_sims, n_trades), replace=True)

    # Equity Curves berechnen (initial_capital * cumprod(1 + r))
    # Wir fügen vorne eine Spalte mit 1.0 ein für den Startpunkt
    multipliers = np.hstack([np.ones((n_sims, 1)), 1.0 + sim_returns])
    equity_matrix = initial_capital * np.cumprod(multipliers, axis=1) # (n_sims, n_trades+1)

    # Perzentile berechnen
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_curves = {p: np.percentile(equity_matrix, p, axis=0) for p in percentiles}

    final_equities = equity_matrix[:, -1]
    prob_profit = float(np.mean(final_equities > initial_capital)) * 100

    # Maximaler Drawdown pro Simulation
    max_dds = []
    for sim in equity_matrix:
        peak = np.maximum.accumulate(sim)
        # Division durch peak, wobei wir Nullen vermeiden (sollte bei Equity nicht vorkommen)
        dd = (sim - peak) / np.where(peak > 0, peak, 1.0) * 100
        max_dds.append(dd.min())
    max_dds = np.array(max_dds)

    return {
        "pct_curves": pct_curves,
        "final_equities": final_equities,
        "prob_profit": prob_profit,
        "max_drawdowns": max_dds,
        "n_sims": n_sims,
        "n_trades": n_trades,
        "initial_capital": initial_capital,
    }


def _display_monte_carlo(mc: dict, initial_capital: float):
    """Zeigt die Monte Carlo Ergebnisse als interaktiven Plotly-Chart."""
    if not mc:
        return

    pct = mc["pct_curves"]
    n = mc["n_trades"] + 1  # inkl. Startpunkt
    x = list(range(n))
    finals = mc["final_equities"]
    ic = mc["initial_capital"]
    prob = mc["prob_profit"]
    max_dds = mc["max_drawdowns"]

    # --- Schlüssel-Metriken ---
    mc1, mc2, mc3, mc4 = st.columns(4)
    median_e = float(np.percentile(finals, 50))
    worst_e  = float(np.percentile(finals, 5))
    best_e   = float(np.percentile(finals, 95))
    med_dd   = float(np.percentile(mc["max_drawdowns"], 50))

    m_color = "normal" if median_e >= ic else "inverse"
    mc1.metric("Median Endkapital", f"${median_e:,.0f}",
               delta=f"{(median_e/ic-1)*100:+.1f}%", delta_color=m_color)
    mc2.metric("Worst Case (5%)",   f"${worst_e:,.0f}",
               delta=f"{(worst_e/ic-1)*100:+.1f}%", delta_color="inverse" if worst_e < ic else "normal")
    mc3.metric("Best Case (95%)",   f"${best_e:,.0f}",
               delta=f"{(best_e/ic-1)*100:+.1f}%", delta_color="normal")
    mc4.metric("Gewinn-WS", f"{prob:.1f}%",
               help="Anteil der Simulationen, die profitabel enden")

    # --- Fan-Chart ---
    fig = go.Figure()

    # Konfidenz-Bänder (Füllflächen)
    band_configs = [
        (5,  95, "rgba(38,166,154,0.08)",  "5–95% Konfidenz"),
        (10, 90, "rgba(38,166,154,0.12)", "10–90% Konfidenz"),
        (25, 75, "rgba(38,166,154,0.20)", "25–75% Konfidenz (IQR)"),
    ]
    for lo, hi, color, name in band_configs:
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(pct[hi]) + list(pct[lo])[::-1],
            fill="toself", fillcolor=color,
            line=dict(width=0), name=name, showlegend=True,
            hoverinfo="skip"
        ))

    # Median-Linie
    fig.add_trace(go.Scatter(
        x=x, y=pct[50],
        mode="lines", name="Median (50%)",
        line=dict(color="#26a69a", width=2.5)
    ))
    # 5th / 95th Perzentil
    fig.add_trace(go.Scatter(
        x=x, y=pct[5],
        mode="lines", name="5. Perzentil (Worst)",
        line=dict(color="#ef5350", width=1.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pct[95],
        mode="lines", name="95. Perzentil (Best)",
        line=dict(color="#66bb6a", width=1.5, dash="dot")
    ))
    # Startkapital-Linie
    fig.add_hline(
        y=ic, line_dash="dash", line_color="#888",
        annotation_text="Startkapital", annotation_position="bottom right"
    )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
        height=380, margin=dict(l=0, r=10, t=30, b=0),
        title=f"Monte Carlo Fan-Chart ({mc['n_sims']} Simulationen, {mc['n_trades']} Trades)",
        xaxis_title="Trade #", yaxis_title="Kapital ($)",
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(gridcolor="#1e2230"),
        xaxis=dict(gridcolor="#1e2230"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Histogramm der Endkapitalien ---
    col_hist, col_dd = st.columns(2)

    with col_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=finals, nbinsx=50,
            marker_color="#26a69a", opacity=0.8,
            name="Endkapital-Verteilung"
        ))
        fig_hist.add_vline(x=ic, line_dash="dash", line_color="#888",
                           annotation_text="Startkapital")
        fig_hist.add_vline(x=float(np.percentile(finals, 50)),
                           line_color="#26a69a", line_width=2,
                           annotation_text="Median")
        fig_hist.update_layout(
            template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
            height=260, margin=dict(l=0, r=0, t=30, b=0),
            title="Verteilung Endkapital",
            xaxis=dict(gridcolor="#1e2230"),
            yaxis=dict(gridcolor="#1e2230"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_dd:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Histogram(
            x=mc["max_drawdowns"], nbinsx=50,
            marker_color="#ef5350", opacity=0.8,
            name="Max Drawdown-Verteilung"
        ))
        fig_dd.add_vline(x=float(np.percentile(max_dds, 50)),
                         line_color="#ef5350", line_width=2,
                         annotation_text=f"Median {med_dd:.1f}%")
        fig_dd.update_layout(
            template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
            height=260, margin=dict(l=0, r=0, t=30, b=0),
            title="Verteilung Max Drawdown",
            xaxis_title="Max Drawdown (%)",
            xaxis=dict(gridcolor="#1e2230"),
            yaxis=dict(gridcolor="#1e2230"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    st.caption(
        f"Methode: {mc['n_sims']} zufällige Permutationen der {mc['n_trades']} Trades · "
        "Gleiche Trade-Ergebnisse, zufällige Reihenfolge · Seed: 42 (reproduzierbar)"
    )


# ============================================================================
# PORTFOLIO BACKTESTING UI & PLOTS
# ============================================================================

def display_portfolio_backtesting():
    st.subheader("Realistisches Portfolio-Management")
    st.info("Dieser Modus simuliert ein gemeinsames Kapital, das auf mehrere Aktien verteilt wird. "
            "Exits geben Kapital frei, das am selben Tag für neue Einträge (zum Open) genutzt werden kann.")

    # --- Einstellungen ---
    col1, col2 = st.columns([2, 1])
    with col1:
        custom_tickers = st.text_area("Ticker-Liste (kommagetrennt)", value="AAPL, MSFT, TSLA, NVDA, AMD, AMZN, META, GOOGL",
                                     help="Gib die Symbole deiner Portfoliowerte ein, getrennt durch Kommata.")
        tickers = [s.strip().upper() for s in custom_tickers.split(",") if s.strip()]
    with col2:
        period_options = {
            "6 Monate": 180, "1 Jahr": 365, "2 Jahre": 730,
            "3 Jahre": 1095, "5 Jahre": 1825, "10 Jahre": 3650,
        }
        period_sel = st.selectbox("Zeitraum", list(period_options.keys()), index=2, key="bt_port_period")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        initial_capital = st.number_input("Portfolio-Startkapital ($)", min_value=1000, value=50000, step=5000, key="bt_port_cap")
    with c2:
        max_positions = st.slider("Max. offene Positionen", 1, 20, 5, key="bt_port_max_pos")
    with c3:
        cap_per_trade = st.slider("Kapital pro Trade (%)", 1, 100, 20, key="bt_port_cap_pct", 
                                  help="Maximaler Anteil des Portfolios, der in einen einzelnen Titel investiert wird.")

    strategy_options = [
        "RSI Überverkauft/Überkauft", "Golden Cross (SMA 20/50)", "EMA Crossover", 
        "MACD Crossover", "Bollinger Bands Mean Reversion", "TTM Squeeze Pro Enhanced", 
        "Donchian Breakout (Turtle)", "Buy & Hold (Einmalanlage)"
    ]
    strategy = st.selectbox("Strategie für Portfolio", strategy_options, key="bt_port_strat")

    with st.expander(" Portfolio-Strategie-Parameter"):
        # Parameter-Inputs (ähnlich wie Einzel-Ticker, aber mit eigenen Keys)
        tg_tp = st.slider("Take-Profit (%)", 0.0, 100.0, 0.0, step=0.5, key="bt_port_tp")
        tg_comm = st.slider("Gebühr (%)", 0.0, 5.0, 0.1, step=0.05, key="bt_port_comm")
        
        rsi_args, sma_args, ema_args, bb_args, ttm_args, donchian_args = None, None, None, None, None, None
        
        if strategy == "RSI Überverkauft/Überkauft":
            rsi_period = st.slider("RSI Periode", 5, 30, 14, key="bt_port_rsi_p")
            rsi_buy = st.slider("RSI Kauf", 10, 45, 30, key="bt_port_rsi_b")
            rsi_sell = st.slider("RSI Verkauf", 55, 90, 70, key="bt_port_rsi_s")
            rsi_args = (rsi_period, rsi_buy, rsi_sell)
        elif strategy == "Golden Cross (SMA 20/50)":
            sma_fast = st.slider("SMA Schnell", 5, 50, 20, key="bt_port_sma_f")
            sma_slow = st.slider("SMA Langsam", 20, 200, 50, key="bt_port_sma_s")
            sma_args = (sma_fast, sma_slow)
        elif strategy == "EMA Crossover":
            ema_fast = st.slider("EMA Schnell", 5, 20, 8, key="bt_port_ema_f")
            ema_slow = st.slider("EMA Langsam", 20, 100, 21, key="bt_port_ema_s")
            ema_args = (ema_fast, ema_slow)
        elif strategy == "Bollinger Bands Mean Reversion":
            bb_window = st.slider("BB Fenster", 10, 50, 20, key="bt_port_bb_w")
            bb_std = st.slider("BB Std", 1.0, 3.5, 2.0, step=0.1, key="bt_port_bb_d")
            bb_args = (bb_window, bb_std)
        elif strategy == "TTM Squeeze Pro Enhanced":
            ttm_l = st.slider("Länge", 10, 50, 20, key="bt_port_ttm_l")
            ttm_bb = st.slider("BB Mult", 1.0, 4.0, 2.0, key="bt_port_ttm_b")
            ttm_agg = st.toggle("Aggres. Exit", True, key="bt_port_ttm_a")
            ttm_ex = st.slider("Bestätigung", 1, 10, 2, key="bt_port_ttm_e") if ttm_agg else 1
            ttm_kc_m = st.slider("KC Mid", 0.5, 3.0, 1.5, key="bt_port_ttm_k")
            ttm_args = (ttm_l, ttm_bb, 1.0, ttm_kc_m, 2.0, ttm_ex, ttm_agg)
        elif strategy == "Donchian Breakout (Turtle)":
            d_en = st.slider("Einstieg", 10, 50, 20, key="bt_port_d_en")
            d_ex = st.slider("Ausstieg", 5, 30, 10, key="bt_port_d_ex")
            donchian_args = (d_en, d_ex)

    if st.button("Portfolio-Backtest starten", type="primary"):
        res = run_portfolio_backtest(
            tickers, strategy, period_options[period_sel], float(initial_capital),
            max_positions=max_positions, capital_per_trade_pct=cap_per_trade,
            commission_pct=tg_comm, take_profit_pct=tg_tp,
            rsi_args=rsi_args, sma_args=sma_args,
            ema_args=ema_args, bb_args=bb_args, ttm_args=ttm_args, 
            donchian_args=donchian_args
        )
        if "error" in res:
            st.error(res["error"])
        else:
            st.session_state["bt_port_result"] = res
            st.session_state["bt_port_strategy_name"] = strategy
            st.success("Portfolio-Backtest abgeschlossen!")

    # --- Ergebnisse anzeigen ---
    if "bt_port_result" in st.session_state:
        res = st.session_state["bt_port_result"]
        m = res["metrics"]
        strat_name = st.session_state["bt_port_strategy_name"]
        
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Return", f"{m['net_profit_pct']:+.1f}%")
        c2.metric("Max Drawdown", f"{m['max_drawdown']:.1f}%")
        c3.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.3f}")
        c4.metric("Trades gesamt", m["total_trades"])
        
        # 1. Haupt-Equity Curve
        st.markdown("### Portfolio Equity Curve")
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=res["equity_curve"].index, y=res["equity_curve"].values,
            mode='lines', name='Gesamtportfolio',
            line=dict(color='#26a69a', width=3),
            fill='tozeroy', fillcolor='rgba(38,166,154,0.1)'
        ))
        fig_port.update_layout(template='plotly_dark', paper_bgcolor='#131722', plot_bgcolor='#131722', height=400, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig_port, use_container_width=True)
        
        # 2. Trade-Tabelle & Ticker-Details
        st.markdown("### Ticker-Details & Einzelverlauf")
        
        # Gruppierte Trade Stats
        if res["trades"]:
            df_trades = pd.DataFrame(res["trades"])
            stats = df_trades.groupby("ticker")["pnl_pct"].agg(["sum", "count", "mean"]).reset_index()
            stats.columns = ["Ticker", "Gesamt %", "Trades", "Ø %"]
            st.dataframe(stats.sort_values("Gesamt %", ascending=False), use_container_width=True, hide_index=True)
            
            # Ticker Expanders
            for ticker in sorted(res["all_data"].keys()):
                with st.expander(f"Details: {ticker}"):
                    df_ticker = res["all_data"][ticker]
                    # Trades für diesen Ticker filtern
                    t_ticker = [t for t in res["trades"] if t["ticker"] == ticker]
                    
                    # Chart mit Markern
                    fig_ticker = plot_equity_curve(
                        equity_curve=pd.Series(dtype=float), # Platzhalter
                        close=df_ticker["Close"].squeeze(),
                        trades=t_ticker, ticker=ticker, strategy_name=strat_name
                    )
                    # Wir nutzen plot_equity_curve nur für den unteren Teil (Price + Markers)
                    # Aber plot_equity_curve erwartet eine Equity Curve... 
                    # Wir passen plot_equity_curve leicht an oder nutzen einen spezifischen Plot
                    st.plotly_chart(fig_ticker, use_container_width=True)
                    
                    # Kurze Zusammenfassung
                    t_win = [t for t in t_ticker if t.get("pnl", 0) > 0]
                    wr = len(t_win)/len(t_ticker)*100 if t_ticker else 0
                    st.caption(f"**Trades:** {len(t_ticker)} · **Win Rate:** {wr:.1f}% · **Profit:** {sum(t['pnl'] for t in t_ticker):+.2f}$")

            # CSV Export
            csv_data = pd.DataFrame(res["trades"]).to_csv(index=False).encode("utf-8")
            st.download_button("Alle Portfolio-Trades als CSV", csv_data, "portfolio_trades.csv", "text/csv")


# ============================================================================
# PORTFOLIO BACKTESTING (NEU)
# ============================================================================

def run_portfolio_backtest(tickers: list, strategy: str, period_days: int, initial_capital: float,
                          max_positions: int = 5, capital_per_trade_pct: float = 20.0,
                          commission_pct: float = 0.1, take_profit_pct: float = 0.0,
                          rsi_args=None, sma_args=None, ema_args=None, 
                          bb_args=None, ttm_args=None, donchian_args=None) -> dict:
    """
    Echter Portfolio-Backtest mit gemeinsamem Kapital und Positions-Limits.
    """
    end = datetime.today().date()
    start = end - timedelta(days=period_days)
    
    # 1. Daten laden
    all_data = {}
    with st.spinner(f"Lade Daten für {len(tickers)} Ticker..."):
        for sym in tickers:
            try:
                sym = sym.strip().upper()
                df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                if not df.empty and len(df) > 10:
                    all_data[sym] = df
            except Exception:
                continue

    if not all_data:
        return {"error": "Keine Kursdaten für die gewählten Ticker gefunden."}

    # 2. Gemeinsamen Index aufbauen
    all_dates = pd.DatetimeIndex([])
    for df in all_data.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    # 3. Signale vorausberechnen
    signals = {}
    for sym, df in all_data.items():
        try:
            cl = df["Close"].squeeze()
            hi = df["High"].squeeze()
            lo = df["Low"].squeeze()
            ent, ext = _get_signals_internal(strategy, cl, hi, lo, rsi_args, sma_args, ema_args, bb_args, ttm_args, donchian_args)
            
            # Auf gemeinsamen Index reindexieren
            signals[sym] = {
                "entries": ent.reindex(all_dates, fill_value=False),
                "exits": ext.reindex(all_dates, fill_value=False)
            }
        except Exception:
            continue

    # 4. Simulation
    cash = initial_capital
    positions = {} # sym: {shares, entry_price, entry_date, cost}
    equity_curve = pd.Series(index=all_dates, dtype=float)
    trades = []
    
    comm_rate = commission_pct / 100.0

    for i, date in enumerate(all_dates):
        if i == 0:
            equity_curve.iloc[0] = cash
            continue
        
        prev_date = all_dates[i-1]
        
        intraday_freed_cash = 0.0

        # 1. EXITS verarbeiten (Signal GESTERN -> Verkauf HEUTE OPEN)
        for sym in list(positions.keys()):
            sig_exit = signals[sym]["exits"].loc[prev_date]
            if sig_exit:
                df_sym = all_data[sym]
                if date in df_sym.index:
                    exit_price = float(df_sym.loc[date, "Open"])
                    pos = positions.pop(sym)
                    
                    comm_exit = pos["shares"] * exit_price * comm_rate
                    proceeds = (pos["shares"] * exit_price) - comm_exit
                    cash += proceeds
                    
                    pnl = proceeds - pos["cost"]
                    net_pct = (proceeds / pos["cost"] - 1) * 100 if pos["cost"] > 0 else 0
                    trades.append({
                        "ticker": sym,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": (exit_price / pos["entry_price"] - 1) * 100,
                        "net_pnl_pct": net_pct,
                        "reason": "Signal"
                    })

        # 2. ENTRIES verarbeiten (Signal GESTERN -> Kauf HEUTE OPEN)
        if len(positions) < max_positions:
            # Dynamische Berechnung basierend auf Portfolio-Wert vom Vortag
            current_portfolio_value = equity_curve.loc[prev_date]
            cap_per_trade_val = current_portfolio_value * (capital_per_trade_pct / 100.0)
            
            for sym, sig in signals.items():
                if sym in positions: continue
                if len(positions) >= max_positions: break
                
                if sig["entries"].loc[prev_date]:
                    df_sym = all_data[sym]
                    if date in df_sym.index:
                        entry_price = float(df_sym.loc[date, "Open"])
                        if entry_price <= 0: continue
                        
                        available = min(cash, cap_per_trade_val)
                        if available < (initial_capital * 0.01): continue # Mindestbetrag
                        
                        comm_entry = available * comm_rate
                        shares = (available - comm_entry) / entry_price
                        cost = (shares * entry_price) + comm_entry
                        
                        cash -= cost
                        positions[sym] = {
                            "shares": shares,
                            "entry_price": entry_price,
                            "entry_date": date,
                            "cost": cost
                        }

        # 3. INTRADAY TAKE PROFIT (Check heutiges High für alle offenen Positionen)
        if take_profit_pct > 0:
            for sym in list(positions.keys()):
                df_sym = all_data[sym]
                if date in df_sym.index:
                    high_p = float(df_sym.loc[date, "High"])
                    open_p = float(df_sym.loc[date, "Open"])
                    pos = positions[sym]
                    tp_price = pos["entry_price"] * (1 + (take_profit_pct / 100))
                    
                    if high_p >= tp_price:
                        # Same-day limits checking
                        is_same_day = (pos["entry_date"] == date)
                        exit_price = max(open_p, tp_price) if not is_same_day else tp_price
                        
                        pos = positions.pop(sym)
                        
                        comm_exit = pos["shares"] * exit_price * comm_rate
                        proceeds = (pos["shares"] * exit_price) - comm_exit
                        # Erst morgen verfügbar machen!
                        intraday_freed_cash += proceeds
                        
                        pnl = proceeds - pos["cost"]
                        net_pct = (proceeds / pos["cost"] - 1) * 100 if pos["cost"] > 0 else 0
                        trades.append({
                            "ticker": sym,
                            "entry_date": pos["entry_date"],
                            "exit_date": date,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "pnl_pct": (exit_price / pos["entry_price"] - 1) * 100,
                            "net_pnl_pct": net_pct,
                            "reason": "Take-Profit"
                        })
                        
        # Cash vom TP wird am Ende des Tages für den nächsten Tag (i+1) verfügbar
        cash += intraday_freed_cash

        # 4. Equity berechnen (Cash + Marktwert zum Tages-Close)
        current_pos_value = 0.0
        for sym, pos in positions.items():
            df_sym = all_data[sym]
            if date in df_sym.index:
                price = float(df_sym.loc[date, "Close"])
            else:
                price = pos["entry_price"]
            current_pos_value += pos["shares"] * price
            
        equity_curve.loc[date] = cash + current_pos_value

    # Offene Positionen schließen für Metriken (ohne Mark-to-Market Gebühren)
    for sym, pos in positions.items():
        last_price = float(all_data[sym]["Close"].iloc[-1])
        proceeds = pos["shares"] * last_price
        pnl = proceeds - pos["cost"]
        net_pct = (proceeds / pos["cost"] - 1) * 100 if pos["cost"] > 0 else 0
        trades.append({
            "ticker": sym,
            "entry_date": pos["entry_date"],
            "exit_date": all_dates[-1],
            "entry_price": pos["entry_price"],
            "exit_price": last_price,
            "pnl": pnl,
            "pnl_pct": (last_price / pos["entry_price"] - 1) * 100,
            "reason": "Offen",
            "open": True
        })

    metrics = _compute_portfolio_metrics(equity_curve, trades, initial_capital)
    
    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": metrics,
        "all_data": all_data,
        "signals": signals
    }


def _compute_portfolio_metrics(equity: pd.Series, trades: list, initial_capital: float) -> dict:
    """Berechnet Performance-Kennzahlen für das Portfolio."""
    if equity.empty: return {}

    final_equity = equity.iloc[-1]
    net_profit = final_equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    returns = equity.pct_change().dropna()
    rf = 0.04
    ann_return = returns.mean() * 252 if len(returns) > 1 else 0.0

    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (ann_return - rf) / (returns.std() * np.sqrt(252))

    # Sortino Ratio
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0.0
    sortino = (ann_return - rf) / downside_vol if downside_vol > 0 else 0.0

    # CAGR
    trading_days = len(equity.dropna())
    years = trading_days / 252
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0.1 else net_profit_pct

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0.0

    completed_trades = [t for t in trades if not t.get("open")]
    win_trades = [t for t in completed_trades if t["pnl"] > 0]
    win_rate = len(win_trades) / len(completed_trades) * 100 if completed_trades else 0

    return {
        "net_profit": net_profit,
        "net_profit_pct": net_profit_pct,
        "cagr": round(cagr, 2),
        "max_drawdown": max_drawdown,
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "total_trades": len(trades),
        "win_rate": win_rate,
        "initial_capital": initial_capital,
        "final_equity": final_equity
    }


def compare_all_strategies(df: pd.DataFrame, initial_capital: float,
                           commission: float = 0.1) -> dict:
    """
    Führt alle Strategien auf demselben Ticker aus und gibt
    Equity Curves + Metriken für den Vergleich zurück.
    """
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    strategies = [
        ("RSI (14, 30/70)",       lambda: strategy_rsi_oversold(close)[:2]),
        ("Golden Cross (20/50)",  lambda: strategy_golden_cross(close)[:2]),
        ("EMA Cross (8/21)",      lambda: strategy_ema_cross(close)[:2]),
        ("MACD Crossover",        lambda: strategy_macd_cross(close)[:2]),
        ("Bollinger Bands",       lambda: strategy_bollinger_bands(close)[:2]),
        ("TTM Squeeze Pro",       lambda: strategy_ttm_squeeze_pro(high, low, close)[:2]),
        ("Donchian (20/10)",      lambda: strategy_donchian_breakout(close)[:2]),
        ("Buy & Hold",            lambda: strategy_buy_and_hold(close)),
    ]

    results = {}
    for name, sig_fn in strategies:
        try:
            entries, exits = sig_fn()
            res = run_backtest(df, entries, exits, initial_capital, commission_pct=commission)
            results[name] = {
                "equity": res["equity_curve"],
                "metrics": res["metrics"],
                "trades": res["trades"]
            }
        except Exception as e:
            logger.debug(f"compare_all_strategies: {name} fehlgeschlagen: {e}")
    return results
