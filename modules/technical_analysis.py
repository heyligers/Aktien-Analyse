# ============================================================================
# modules/technical_analysis.py — Indikatoren, Muster, Fibonacci
# ============================================================================

import pandas as pd
import json


def detect_candlestick_patterns(df):
    """Erkennt gängige Candlestick-Muster. Gibt LWC-Marker-Liste zurück."""
    markers = []
    if len(df) < 3:
        return markers
    for i in range(2, len(df)):
        ts = df.index[i]
        t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(ts.value // 1e9)
        o, h, l, c = [float(df.iloc[i][x]) for x in ['Open', 'High', 'Low', 'Close']]
        body = abs(c - o)
        full_range = h - l
        if full_range == 0:
            continue
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        if body / full_range < 0.1:
            markers.append({"time": t, "position": "aboveBar", "color": "#ffeb3b",
                            "shape": "circle", "text": "Doji"})
            continue

        if body > 0 and lower_shadow > 2 * body and upper_shadow < body * 0.3:
            markers.append({"time": t, "position": "belowBar", "color": "#26a69a",
                            "shape": "arrowUp", "text": "Hammer"})
            continue

        if body > 0 and upper_shadow > 2 * body and lower_shadow < body * 0.3:
            markers.append({"time": t, "position": "aboveBar", "color": "#ef5350",
                            "shape": "arrowDown", "text": "Shooting Star"})
            continue

        prev_o, prev_c = float(df.iloc[i-1]['Open']), float(df.iloc[i-1]['Close'])
        if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
            markers.append({"time": t, "position": "belowBar", "color": "#26a69a",
                            "shape": "arrowUp", "text": "Bull. Engulf."})
        elif prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
            markers.append({"time": t, "position": "aboveBar", "color": "#ef5350",
                            "shape": "arrowDown", "text": "Bear. Engulf."})
    return markers


def calculate_pivot_points(df):
    """Klassische Pivot Points aus der letzten Kerze."""
    h, l, c = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
    p = (h + l + c) / 3
    return {'P': p, 'R1': 2*p - l, 'R2': p + (h - l), 'R3': 2*p + (h - 2*l),
            'S1': 2*p - h, 'S2': p - (h - l), 'S3': 2*p - (2*h - l)}


def calculate_fibonacci(df, lookback=60):
    """Fibonacci Retracements vom Hoch zum Tief der letzten N Kerzen."""
    window = df.tail(min(lookback, len(df)))
    high = float(window['High'].max())
    low = float(window['Low'].min())
    diff = high - low
    if diff < 0.001:
        return {}
    return {
        '0% (Hoch)': high,
        '23,6%': high - 0.236 * diff,
        '38,2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61,8%': high - 0.618 * diff,
        '78,6%': high - 0.786 * diff,
        '100% (Tief)': low,
    }


def sr_fib_to_lwc_lines(pivot_data, fib_data, df):
    """Erzeugt JavaScript-Code für horizontale S/R und Fibonacci Linien im LWC-Chart."""
    js = ""
    t_start = int(df.index[0].timestamp()) if hasattr(df.index[0], 'timestamp') else int(df.index[0].value // 1e9)
    t_end = int(df.index[-1].timestamp()) if hasattr(df.index[-1], 'timestamp') else int(df.index[-1].value // 1e9)

    if pivot_data:
        colors = {'P': '#ffffff', 'R1': '#ef5350', 'R2': '#c62828', 'R3': '#b71c1c',
                  'S1': '#26a69a', 'S2': '#00796b', 'S3': '#004d40'}
        for label, val in pivot_data.items():
            col = colors.get(label, '#888')
            js += f"""
            var sr_{label} = mainChart.addLineSeries({{
                color:'{col}', lineWidth:1, lineStyle:2, title:'{label}: {val:.2f}',
                lastValueVisible:true, priceLineVisible:false
            }});
            sr_{label}.setData([{{time:{t_start},value:{val:.4f}}},{{time:{t_end},value:{val:.4f}}}]);"""

    if fib_data:
        fib_colors = ['#ffeb3b', '#ff9800', '#ff5722', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5']
        for i, (label, val) in enumerate(fib_data.items()):
            col = fib_colors[i % len(fib_colors)]
            safe = label.replace('%', 'pct').replace(' ', '').replace('(', '').replace(')', '').replace(',', '')
            js += f"""
            var fib_{safe} = mainChart.addLineSeries({{
                color:'{col}', lineWidth:1, lineStyle:1, title:'Fib {label}: {val:.2f}',
                lastValueVisible:true, priceLineVisible:false
            }});
            fib_{safe}.setData([{{time:{t_start},value:{val:.4f}}},{{time:{t_end},value:{val:.4f}}}]);"""
    return js
