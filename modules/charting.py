# ============================================================================
# modules/charting.py — LightweightCharts HTML-Chart-Builder
# ============================================================================

import json
import pandas as pd


def build_lwc_html(df, sma_on, ema_on, bb_on, vwap_on,
                   rsi_on, macd_on, stoch_on,
                   atr_on, obv_on, willr_on, ich_on, interval,
                   cdl_markers=None, sr_fib_js=""):

    def to_lwc_candles(df):
        rows = []
        for ts, row in df.iterrows():
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(ts.value // 1e9)
            rows.append({"time": t, "open": round(float(row["Open"]), 4),
                         "high": round(float(row["High"]), 4),
                         "low": round(float(row["Low"]), 4),
                         "close": round(float(row["Close"]), 4)})
        return rows

    def to_lwc_line(df, col):
        rows = []
        for ts, val in df[col].items():
            if pd.isna(val): continue
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(ts.value // 1e9)
            rows.append({"time": t, "value": round(float(val), 4)})
        return rows

    def to_lwc_volume(df):
        rows = []
        for ts, row in df.iterrows():
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(ts.value // 1e9)
            color = "rgba(0,200,90,0.6)" if float(row["Close"]) >= float(row["Open"]) else "rgba(220,50,50,0.6)"
            rows.append({"time": t, "value": float(row["Volume"]), "color": color})
        return rows

    candle_data = json.dumps(to_lwc_candles(df))
    volume_data = json.dumps(to_lwc_volume(df))
    markers_data = json.dumps(cdl_markers or [])

    # --- Overlays ---
    overlays_js = ""
    if sma_on:
        for col in [c for c in df.columns if "SMA" in c]:
            d = json.dumps(to_lwc_line(df, col))
            color = "#f0a500" if "20" in col else "#2196f3"
            overlays_js += f"""
            var s_{col} = mainChart.addLineSeries({{color:'{color}',lineWidth:1.5,title:'{col}'}});
            s_{col}.setData({d});"""
    if ema_on:
        for col in [c for c in df.columns if "EMA" in c]:
            d = json.dumps(to_lwc_line(df, col))
            color = "#ff6b35" if "20" in col else "#9c27b0"
            overlays_js += f"""
            var e_{col} = mainChart.addLineSeries({{color:'{color}',lineWidth:1.5,lineStyle:1,title:'{col}'}});
            e_{col}.setData({d});"""
    if bb_on:
        bbu_cols = [c for c in df.columns if "BBU" in c]
        bbl_cols = [c for c in df.columns if "BBL" in c]
        if bbu_cols and bbl_cols:
            du = json.dumps(to_lwc_line(df, bbu_cols[0]))
            dl = json.dumps(to_lwc_line(df, bbl_cols[0]))
            overlays_js += f"""
            var sBBU = mainChart.addLineSeries({{color:'rgba(150,150,150,0.7)',lineWidth:1,lineStyle:2,title:'BB Up'}});
            sBBU.setData({du});
            var sBBL = mainChart.addLineSeries({{color:'rgba(150,150,150,0.7)',lineWidth:1,lineStyle:2,title:'BB Low'}});
            sBBL.setData({dl});"""
    if vwap_on and "VWAP" in df.columns:
        dv = json.dumps(to_lwc_line(df, "VWAP"))
        overlays_js += f"""
            var sVWAP = mainChart.addLineSeries({{color:'#00bcd4',lineWidth:1.5,lineStyle:3,title:'VWAP'}});
            sVWAP.setData({dv});"""
    if ich_on:
        ich_map = {
            'ITS_9': ('#26a69a', 'Tenkan'), 'IKS_26': ('#ef5350', 'Kijun'),
            'ISA_9': ('rgba(38,166,154,0.25)', 'Span A'),
            'ISB_26': ('rgba(239,83,80,0.25)', 'Span B'),
            'ICS_26': ('#b0bec5', 'Chikou'),
        }
        for col_key, (color, label) in ich_map.items():
            matching = [c for c in df.columns if col_key in c]
            if matching:
                di = json.dumps(to_lwc_line(df, matching[0]))
                overlays_js += f"""
            var ich_{col_key} = mainChart.addLineSeries({{color:'{color}',lineWidth:1,title:'{label}'}});
            ich_{col_key}.setData({di});"""

    # --- Oszillatoren ---
    osc_panels_js = ""
    if rsi_on:
        r_cols = [c for c in df.columns if "RSI" in c]
        if r_cols:
            dr = json.dumps(to_lwc_line(df, r_cols[0]))
            osc_panels_js += f"""
            var rsiChart = LightweightCharts.createChart(document.getElementById('rsi_panel'), {{
                ...panelOpts, height: 120,
            }});
            rsiChart.applyOptions({{...chartOpts, rightPriceScale:{{visible:true,scaleMargins:{{top:0.05,bottom:0.05}}}}}});
            syncCharts.push(rsiChart);
            var rsiSeries = rsiChart.addLineSeries({{color:'#9c27b0',lineWidth:1.5,title:'RSI'}});
            rsiSeries.setData({dr});
            rsiChart.addLineSeries({{color:'rgba(255,60,60,0.5)',lineWidth:1,lineStyle:2}}).setData(
                {dr}.map(d => ({{time:d.time,value:70}})));
            rsiChart.addLineSeries({{color:'rgba(60,200,60,0.5)',lineWidth:1,lineStyle:2}}).setData(
                {dr}.map(d => ({{time:d.time,value:30}})));"""
    if macd_on:
        m_cols = [c for c in df.columns if "MACD_" in c]
        s_cols = [c for c in df.columns if "MACDs" in c]
        if m_cols and s_cols:
            dm = json.dumps(to_lwc_line(df, m_cols[0]))
            ds = json.dumps(to_lwc_line(df, s_cols[0]))
            osc_panels_js += f"""
            var macdChart = LightweightCharts.createChart(document.getElementById('macd_panel'), {{
                ...panelOpts, height: 120,
            }});
            macdChart.applyOptions({{...chartOpts, rightPriceScale:{{visible:true,scaleMargins:{{top:0.1,bottom:0.1}}}}}});
            syncCharts.push(macdChart);
            var macdSeries = macdChart.addLineSeries({{color:'#2196f3',lineWidth:1.5,title:'MACD'}});
            var signalSeries = macdChart.addLineSeries({{color:'#ff9800',lineWidth:1.5,title:'Signal'}});
            macdSeries.setData({dm});
            signalSeries.setData({ds});"""
    if stoch_on:
        k_cols = [c for c in df.columns if "STOCHk" in c]
        d_cols = [c for c in df.columns if "STOCHd" in c]
        if k_cols and d_cols:
            dk = json.dumps(to_lwc_line(df, k_cols[0]))
            dd_s = json.dumps(to_lwc_line(df, d_cols[0]))
            osc_panels_js += f"""
            var stochChart = LightweightCharts.createChart(document.getElementById('stoch_panel'), {{
                ...panelOpts, height: 120,
            }});
            stochChart.applyOptions({{...chartOpts, rightPriceScale:{{visible:true,scaleMargins:{{top:0.05,bottom:0.05}}}}}});
            syncCharts.push(stochChart);
            var stochK = stochChart.addLineSeries({{color:'#29b6f6',lineWidth:1.5,title:'%K'}});
            var stochD = stochChart.addLineSeries({{color:'#ff7043',lineWidth:1.5,title:'%D'}});
            stochK.setData({dk});
            stochD.setData({dd_s});
            stochChart.addLineSeries({{color:'rgba(255,60,60,0.4)',lineWidth:1,lineStyle:2}}).setData(
                {dk}.map(d => ({{time:d.time,value:80}})));
            stochChart.addLineSeries({{color:'rgba(60,200,60,0.4)',lineWidth:1,lineStyle:2}}).setData(
                {dk}.map(d => ({{time:d.time,value:20}})));"""
    if atr_on:
        atr_cols = [c for c in df.columns if "ATR" in c]
        if atr_cols:
            da = json.dumps(to_lwc_line(df, atr_cols[0]))
            osc_panels_js += f"""
            var atrChart = LightweightCharts.createChart(document.getElementById('atr_panel'), {{
                ...panelOpts, height: 100,
            }});
            atrChart.applyOptions({{...chartOpts, rightPriceScale:{{visible:true,scaleMargins:{{top:0.05,bottom:0.05}}}}}});
            syncCharts.push(atrChart);
            var atrSeries = atrChart.addLineSeries({{color:'#ff9800',lineWidth:1.5,title:'ATR(14)'}});
            atrSeries.setData({da});"""
    if obv_on:
        obv_cols = [c for c in df.columns if "OBV" in c]
        if obv_cols:
            do_d = json.dumps(to_lwc_line(df, obv_cols[0]))
            osc_panels_js += f"""
            var obvChart = LightweightCharts.createChart(document.getElementById('obv_panel'), {{
                ...panelOpts, height: 100,
            }});
            obvChart.applyOptions({{...chartOpts, rightPriceScale:{{visible:true,scaleMargins:{{top:0.1,bottom:0.1}}}}}});
            syncCharts.push(obvChart);
            var obvSeries = obvChart.addLineSeries({{color:'#ab47bc',lineWidth:1.5,title:'OBV'}});
            obvSeries.setData({do_d});"""
    if willr_on:
        wr_cols = [c for c in df.columns if "WILLR" in c]
        if wr_cols:
            dw = json.dumps(to_lwc_line(df, wr_cols[0]))
            osc_panels_js += f"""
            var willrChart = LightweightCharts.createChart(document.getElementById('willr_panel'), {{
                ...panelOpts, height: 100,
            }});
            willrChart.applyOptions({{...chartOpts, rightPriceScale:{{visible:true,scaleMargins:{{top:0.05,bottom:0.05}}}}}});
            syncCharts.push(willrChart);
            var willrSeries = willrChart.addLineSeries({{color:'#ef9a9a',lineWidth:1.5,title:'Williams %R'}});
            willrSeries.setData({dw});
            willrChart.addLineSeries({{color:'rgba(255,60,60,0.4)',lineWidth:1,lineStyle:2}}).setData(
                {dw}.map(d => ({{time:d.time,value:-20}})));
            willrChart.addLineSeries({{color:'rgba(60,200,60,0.4)',lineWidth:1,lineStyle:2}}).setData(
                {dw}.map(d => ({{time:d.time,value:-80}})));"""

    osc_divs = ""
    if rsi_on:   osc_divs += '<div id="rsi_panel"   style="width:100%;border-top:1px solid #2a2a3a;"></div>'
    if macd_on:  osc_divs += '<div id="macd_panel"  style="width:100%;border-top:1px solid #2a2a3a;"></div>'
    if stoch_on: osc_divs += '<div id="stoch_panel" style="width:100%;border-top:1px solid #2a2a3a;"></div>'
    if atr_on:   osc_divs += '<div id="atr_panel"   style="width:100%;border-top:1px solid #2a2a3a;"></div>'
    if obv_on:   osc_divs += '<div id="obv_panel"   style="width:100%;border-top:1px solid #2a2a3a;"></div>'
    if willr_on: osc_divs += '<div id="willr_panel" style="width:100%;border-top:1px solid #2a2a3a;"></div>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:#131722; font-family:'Segoe UI',system-ui,sans-serif; overflow:hidden; }}
  #fs_container {{ width:100%; display:flex; flex-direction:column; }}
  #fs_container.pseudo-fs {{ position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:99999; background:#131722; }}
  #toolbar {{ display:flex; align-items:center; gap:2px; padding:3px 8px; background:#1e222d; border-bottom:1px solid #2a2e39; flex-shrink:0; user-select:none; }}
  .tb {{ background:none; border:1px solid transparent; color:#787b86; padding:4px 7px; border-radius:4px; cursor:pointer; font-size:13px; display:flex; align-items:center; justify-content:center; min-width:30px; height:26px; transition:all .15s; }}
  .tb:hover {{ background:#2a2e39; color:#d1d4dc; border-color:#363a45; }}
  .tb.active {{ background:#2962FF; color:#fff; border-color:#2962FF; }}
  .tsep {{ width:1px; height:18px; background:#363a45; margin:0 3px; }}
  .tlbl {{ font-size:10px; color:#787b86; padding:0 4px; }}
  #wrapper {{ width:100%; display:flex; flex-direction:column; flex:1; min-height:0; }}
  #chart_area {{ position:relative; flex-shrink:0; }}
  #main_panel {{ width:100%; height:520px; }}
  #draw_canvas {{ position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:5; }}
  #draw_canvas.drawing {{ cursor:crosshair; }}
  #vol_panel {{ width:100%; height:100px; border-top:1px solid #2a2a3a; }}
  .legend {{ position:absolute; top:6px; left:8px; z-index:10; color:#d1d4dc; font-size:12px; pointer-events:none; }}
  #crosshair_info {{ padding:4px 8px; background:rgba(19,23,34,0.85); border-radius:4px; display:inline-block; }}
  #ruler_info {{ position:absolute; z-index:20; background:rgba(30,34,45,0.95); border:1px solid #2962FF; border-radius:6px; padding:8px 12px; color:#d1d4dc; font-size:11px; pointer-events:none; display:none; box-shadow:0 4px 12px rgba(0,0,0,0.5); white-space:nowrap; }}
  #draw_hint {{ position:absolute; bottom:4px; left:50%; transform:translateX(-50%); background:rgba(41,98,255,0.9); color:#fff; padding:3px 12px; border-radius:4px; font-size:11px; pointer-events:none; z-index:15; display:none; }}
</style></head><body>
<div id="fs_container">
  <div id="toolbar">
    <button class="tb active" data-tool="cursor" title="Cursor (Esc)">↖</button>
    <div class="tsep"></div>
    <button class="tb" data-tool="trendline" title="Trendlinie">╱</button>
    <button class="tb" data-tool="hline" title="Horizontale Linie">━</button>
    <button class="tb" data-tool="rect" title="Rechteck">▭</button>
    <button class="tb" data-tool="fib_draw" title="Fibonacci">Fib</button>
    <button class="tb" data-tool="circle" title="Ellipse">◯</button>
    <button class="tb" data-tool="text" title="Text">T</button>
    <button class="tb" data-tool="pitchfork" title="Pitchfork">⑂</button>
    <div class="tsep"></div>
    <button class="tb" data-tool="ruler" title="Lineal / Messung">📏</button>
    <div class="tsep"></div>
    <button class="tb" id="btn_undo" title="Letzte löschen">↺</button>
    <button class="tb" id="btn_clear" title="Alle löschen">×</button>
    <div style="flex:1"></div>
    <span class="tlbl" id="draw_count"></span>
    <button class="tb" id="btn_fs" title="Vollbild (F)">⊞</button>
  </div>
  <div id="wrapper">
    <div id="chart_area">
      <div id="main_panel"></div>
      <canvas id="draw_canvas"></canvas>
      <div class="legend"><span id="crosshair_info"></span></div>
      <div id="ruler_info"></div>
      <div id="draw_hint"></div>
    </div>
    <div id="vol_panel"></div>
    {osc_divs}
  </div>
</div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
var chartOpts = {{
    autoSize: true,
    layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
    grid:   {{ vertLines: {{ color: '#1e2230' }}, horzLines: {{ color: '#1e2230' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{ timeVisible: true, secondsVisible: false, borderColor: '#2a2a3a', rightOffset: 5 }},
    rightPriceScale: {{ borderColor: '#2a2a3a' }},
    handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true }},
    handleScale:  {{ mouseWheel: true, pinch: true, axisPressedMouseMove: {{ time: true, price: true }} }},
}};
var panelOpts = {{
    autoSize: true,
    layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
    grid:   {{ vertLines: {{ color: '#1e2230' }}, horzLines: {{ color: '#1e2230' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{ visible: false }},
    rightPriceScale: {{ borderColor: '#2a2a3a' }},
    handleScroll: {{ mouseWheel: true, pressedMouseMove: true }},
    handleScale:  {{ mouseWheel: true, pinch: true, axisPressedMouseMove: {{ time: true, price: true }} }},
}};
var mainChart = LightweightCharts.createChart(document.getElementById('main_panel'), {{
    ...chartOpts, height: 520,
}});
mainChart.applyOptions({{ rightPriceScale: {{ scaleMargins: {{ top: 0.05, bottom: 0.1 }} }} }});
var candleSeries = mainChart.addCandlestickSeries({{
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});
candleSeries.setData({candle_data});
var markers = {markers_data};
if (markers && markers.length > 0) {{
    markers.sort(function(a,b) {{ return a.time - b.time; }});
    candleSeries.setMarkers(markers);
}}
{overlays_js}
{sr_fib_js}
var volChart = LightweightCharts.createChart(document.getElementById('vol_panel'), {{
    ...panelOpts, height: 100,
}});
volChart.applyOptions({{ rightPriceScale: {{ visible: true, scaleMargins: {{ top: 0.1, bottom: 0 }} }} }});
var volSeries = volChart.addHistogramSeries({{ priceFormat: {{ type: 'volume' }}, title: 'Vol' }});
volSeries.setData({volume_data});
var syncCharts = [volChart];
{osc_panels_js}
function syncTimeRange(srcChart, charts) {{
    srcChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
        if (range === null) return;
        charts.forEach(c => {{ if (c !== srcChart) c.timeScale().setVisibleLogicalRange(range); }});
    }});
}}
syncTimeRange(mainChart, syncCharts);
syncCharts.forEach(c => syncTimeRange(c, [mainChart, ...syncCharts.filter(x => x !== c)]));
mainChart.timeScale().fitContent();

// ===== DRAWING TOOLS =====
var DS = {{ tool:'cursor', pts:[], mx:null, my:null, mt:null, mp:null, drawings:[], canvas:document.getElementById('draw_canvas'), ctx:null }};
DS.ctx = DS.canvas.getContext('2d');
var candleDataArr = {candle_data};
var hintEl = document.getElementById('draw_hint');
var rulerEl = document.getElementById('ruler_info');
var countEl = document.getElementById('draw_count');
function resizeCanvas() {{ var r = document.getElementById('main_panel'); DS.canvas.width = r.clientWidth; DS.canvas.height = r.clientHeight; }}
resizeCanvas();
new ResizeObserver(function(){{ resizeCanvas(); renderAll(); }}).observe(document.getElementById('main_panel'));
function t2x(t) {{ var c = mainChart.timeScale().timeToCoordinate(t); return c === null ? -9999 : c; }}
function p2y(p) {{ var c = candleSeries.priceToCoordinate(p); return c === null ? -9999 : c; }}
function y2p(y) {{ return candleSeries.coordinateToPrice(y); }}
var toolBtns = document.querySelectorAll('.tb[data-tool]');
toolBtns.forEach(function(b) {{ b.addEventListener('click', function() {{ setTool(b.getAttribute('data-tool')); }}); }});
function setTool(t) {{
  DS.tool = t; DS.pts = [];
  toolBtns.forEach(function(b){{ b.classList.toggle('active', b.getAttribute('data-tool')===t); }});
  DS.canvas.classList.toggle('drawing', t !== 'cursor');
  document.getElementById('main_panel').style.cursor = t === 'cursor' ? 'default' : 'crosshair';
  var interact = t === 'cursor';
  mainChart.applyOptions({{ handleScroll: interact ? {{ mouseWheel:true, pressedMouseMove:true, horzTouchDrag:true, vertTouchDrag:true }} : false, handleScale: interact ? {{ mouseWheel:true, pinch:true, axisPressedMouseMove:{{ time:true, price:true }} }} : false }});
  var hints = {{ trendline:'Klick: Start → Ende', hline:'Klick: Preisniveau', rect:'Klick: Ecke 1 → 2', fib_draw:'Klick: Hoch → Tief', circle:'Klick: Ecke 1 → 2', text:'Klick: Position', pitchfork:'Klick: 3× Punkte', ruler:'Klick: Start → Ende' }};
  hintEl.style.display = hints[t] ? 'block' : 'none';
  hintEl.textContent = hints[t] || '';
  rulerEl.style.display = 'none';
  renderAll();
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') {{ if (isFS) exitFS(); else setTool('cursor'); }}
  if (e.key === 'f' || e.key === 'F') toggleFS();
  if ((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) undoDrawing();
}});
mainChart.subscribeCrosshairMove(function(param) {{
  if (param.point && param.time) {{
    DS.mt = param.time; DS.mp = y2p(param.point.y);
    DS.mx = param.point.x; DS.my = param.point.y;
    if (DS.tool !== 'cursor' && DS.pts.length > 0) renderAll();
    if (DS.tool === 'ruler' && DS.pts.length === 1) showRulerLive();
  }}
  if (param.seriesData && param.seriesData.size > 0) {{
    var bar = param.seriesData.get(candleSeries);
    if (bar) document.getElementById('crosshair_info').innerHTML = 'O: '+bar.open.toFixed(2)+' H: '+bar.high.toFixed(2)+' L: '+bar.low.toFixed(2)+' C: <b>'+bar.close.toFixed(2)+'</b>';
  }}
}});
mainChart.subscribeClick(function(param) {{
  if (DS.tool === 'cursor' || !param.time || !param.point) return;
  var price = y2p(param.point.y); var time = param.time;
  DS.pts.push({{time:time, price:price}});
  var needed = {{hline:1, text:1, trendline:2, rect:2, fib_draw:2, circle:2, ruler:2, pitchfork:3}};
  var n = needed[DS.tool] || 2;
  if (DS.tool === 'text' && DS.pts.length === 1) {{
    var txt = prompt('Text eingeben:','');
    if (txt) DS.drawings.push({{type:'text',points:[...DS.pts],color:'#d1d4dc',text:txt}});
    DS.pts = []; renderAll(); updateCount(); return;
  }}
  if (DS.pts.length >= n) {{
    DS.drawings.push({{type:DS.tool, points:[...DS.pts], color: DS.tool==='ruler'?'#787b86':'#2962FF'}});
    DS.pts = [];
    if (DS.tool === 'ruler') showRulerFinal(DS.drawings[DS.drawings.length-1]);
    renderAll(); updateCount();
  }}
}});

function updateCount() {{ countEl.textContent = DS.drawings.length > 0 ? DS.drawings.length + ' Obj.' : ''; }}
document.getElementById('btn_undo').addEventListener('click', undoDrawing);
document.getElementById('btn_clear').addEventListener('click', function() {{ DS.drawings = []; DS.pts = []; rulerEl.style.display='none'; renderAll(); updateCount(); }});
function undoDrawing() {{ DS.drawings.pop(); DS.pts = []; rulerEl.style.display = 'none'; renderAll(); updateCount(); }}
mainChart.timeScale().subscribeVisibleLogicalRangeChange(function() {{ renderAll(); }});
function renderAll() {{
  resizeCanvas();
  var ctx = DS.ctx; var W = DS.canvas.width; var H = DS.canvas.height;
  ctx.clearRect(0,0,W,H);
  DS.drawings.forEach(function(d) {{ renderOne(ctx,d,W,H,false); }});
  if (DS.pts.length > 0 && DS.mt !== null) {{
    var preview = {{type:DS.tool, points:[...DS.pts, {{time:DS.mt,price:DS.mp}}], color:'rgba(41,98,255,0.6)', text:''}};
    renderOne(ctx,preview,W,H,true);
  }}
}}
function renderOne(ctx,d,W,H,preview) {{
  var pts = d.points.map(function(p) {{ return {{x:t2x(p.time), y:p2y(p.price)}}; }});
  if (pts.some(function(p){{ return p.x < -5000 || p.y < -5000; }})) return;
  ctx.save();
  ctx.strokeStyle = d.color || '#2962FF'; ctx.fillStyle = d.color || '#2962FF';
  ctx.lineWidth = preview ? 1 : 1.5; ctx.setLineDash(preview ? [5,4] : []); ctx.globalAlpha = preview ? 0.7 : 1;
  switch(d.type) {{
    case 'trendline':
      if (pts.length < 2) break;
      ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y); ctx.lineTo(pts[1].x,pts[1].y); ctx.stroke();
      dot(ctx,pts[0]); dot(ctx,pts[1]); break;
    case 'hline':
      ctx.beginPath(); ctx.moveTo(0,pts[0].y); ctx.lineTo(W,pts[0].y); ctx.stroke();
      ctx.font='11px sans-serif'; ctx.fillText(d.points[0].price.toFixed(2),4,pts[0].y-4); break;
    case 'rect':
      if (pts.length < 2) break;
      var rx=Math.min(pts[0].x,pts[1].x), ry=Math.min(pts[0].y,pts[1].y);
      var rw=Math.abs(pts[1].x-pts[0].x), rh=Math.abs(pts[1].y-pts[0].y);
      ctx.globalAlpha = preview ? 0.1 : 0.12; ctx.fillRect(rx,ry,rw,rh);
      ctx.globalAlpha = preview ? 0.7 : 0.8; ctx.strokeRect(rx,ry,rw,rh); break;
    case 'fib_draw':
      if (pts.length < 2) break;
      var hp = Math.max(d.points[0].price,d.points[1].price);
      var lp = Math.min(d.points[0].price,d.points[1].price);
      var diff = hp - lp; if (diff < 0.001) break;
      var xl = Math.min(pts[0].x,pts[1].x), xr = Math.max(pts[0].x,pts[1].x);
      var fibLvls = [0,0.236,0.382,0.5,0.618,0.786,1];
      var fibCols = ['#FF6D00','#FFD600','#00E676','#2979FF','#D500F9','#F50057','#FF6D00'];
      fibLvls.forEach(function(lv,i) {{
        var pr = hp - lv*diff; var fy = p2y(pr); if (fy < -5000) return;
        ctx.strokeStyle = fibCols[i]; ctx.globalAlpha = 0.7; ctx.setLineDash([4,2]);
        ctx.beginPath(); ctx.moveTo(xl,fy); ctx.lineTo(xr,fy); ctx.stroke();
        ctx.fillStyle = fibCols[i]; ctx.font='10px sans-serif';
        ctx.fillText((lv*100).toFixed(1)+'% ('+pr.toFixed(2)+')',xl+4,fy-3);
        ctx.setLineDash(preview?[5,4]:[]);
      }}); break;
    case 'circle':
      if (pts.length < 2) break;
      var ecx=(pts[0].x+pts[1].x)/2, ecy=(pts[0].y+pts[1].y)/2;
      var erx=Math.abs(pts[1].x-pts[0].x)/2, ery=Math.abs(pts[1].y-pts[0].y)/2;
      if (erx < 1) erx = 1; if (ery < 1) ery = 1;
      ctx.beginPath(); ctx.ellipse(ecx,ecy,erx,ery,0,0,2*Math.PI);
      ctx.globalAlpha=0.1; ctx.fill(); ctx.globalAlpha=preview?0.7:0.8; ctx.stroke(); break;
    case 'text':
      ctx.fillStyle = d.color || '#d1d4dc'; ctx.font = 'bold 13px sans-serif';
      ctx.fillText(d.text||'', pts[0].x, pts[0].y); break;
    case 'pitchfork':
      if (pts.length < 3) break;
      var mid = {{x:(pts[1].x+pts[2].x)/2, y:(pts[1].y+pts[2].y)/2}};
      var ddx = mid.x - pts[0].x, ddy = mid.y - pts[0].y;
      ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y); ctx.lineTo(mid.x+ddx*2,mid.y+ddy*2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pts[1].x,pts[1].y); ctx.lineTo(pts[1].x+ddx*3,pts[1].y+ddy*3); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pts[2].x,pts[2].y); ctx.lineTo(pts[2].x+ddx*3,pts[2].y+ddy*3); ctx.stroke();
      dot(ctx,pts[0]); dot(ctx,pts[1]); dot(ctx,pts[2]); break;
    case 'ruler':
      if (pts.length < 2) break;
      ctx.setLineDash([6,3]); ctx.strokeStyle='#787b86';
      ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y); ctx.lineTo(pts[1].x,pts[1].y); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle='#787b86'; dot(ctx,pts[0]); dot(ctx,pts[1]); break;
  }}
  ctx.restore();
}}
function dot(ctx,p) {{ ctx.beginPath(); ctx.arc(p.x,p.y,3,0,2*Math.PI); ctx.fill(); }}
function showRulerLive() {{
  if (DS.pts.length < 1 || DS.mt === null) return;
  showRulerData(DS.pts[0], {{time:DS.mt, price:DS.mp}}, DS.mx, DS.my);
}}
function showRulerFinal(d) {{
  if (d.points.length < 2) return;
  var p2 = d.points[1]; var px = t2x(p2.time); var py = p2y(p2.price);
  showRulerData(d.points[0], p2, px, py);
}}
function showRulerData(p1, p2, mx, my) {{
  var dp = p2.price - p1.price; var dpPct = (dp / p1.price * 100);
  var t1 = typeof p1.time === 'object' ? new Date(p1.time.year, p1.time.month-1, p1.time.day) : new Date(p1.time*1000);
  var t2 = typeof p2.time === 'object' ? new Date(p2.time.year, p2.time.month-1, p2.time.day) : new Date(p2.time*1000);
  var diffMs = Math.abs(t2-t1); var diffDays = Math.round(diffMs/86400000);
  var diffH = Math.round(diffMs/3600000);
  var timeStr = diffDays > 0 ? diffDays+' Tage' : diffH+' Std';
  var bars = 0;
  if (candleDataArr.length > 0) {{
    var st = Math.min(p1.time, p2.time); var et = Math.max(p1.time, p2.time);
    bars = candleDataArr.filter(function(c){{ return c.time >= st && c.time <= et; }}).length;
  }}
  var arrow = dp >= 0 ? '▲' : '▼';
  var col = dp >= 0 ? '#26a69a' : '#ef5350';
  rulerEl.innerHTML = '<div style="color:'+col+';font-weight:bold;margin-bottom:3px;">'+arrow+' '+dp.toFixed(2)+' ('+dpPct.toFixed(2)+'%)</div><div> '+timeStr+' · '+bars+' Kerzen</div>';
  rulerEl.style.display = 'block';
  rulerEl.style.left = Math.min(mx+15, DS.canvas.width-180)+'px';
  rulerEl.style.top = Math.max(my-50, 5)+'px';
}}

// ===== FULLSCREEN =====
var fsContainer = document.getElementById('fs_container');
var isFS = false;
var savedIframeStyles = null;
document.getElementById('btn_fs').addEventListener('click', toggleFS);
function toggleFS() {{ if (!isFS) {{ enterFS(); }} else {{ exitFS(); }} }}
function enterFS() {{
  try {{
    var iframe = window.frameElement;
    if (iframe) {{
      savedIframeStyles = {{
        position: iframe.style.position, top: iframe.style.top, left: iframe.style.left,
        width: iframe.style.width, height: iframe.style.height, zIndex: iframe.style.zIndex,
        maxWidth: iframe.style.maxWidth, border: iframe.style.border
      }};
      iframe.style.position = 'fixed'; iframe.style.top = '0'; iframe.style.left = '0';
      iframe.style.width = '100vw'; iframe.style.height = '100vh';
      iframe.style.zIndex = '999999'; iframe.style.maxWidth = '100vw'; iframe.style.border = 'none';
    }}
  }} catch(e) {{}}
  fsContainer.classList.add('pseudo-fs');
  document.body.style.overflow = 'hidden';
  isFS = true;
  setTimeout(resizeChartsFS, 80);
}}
function exitFS() {{
  try {{
    var iframe = window.frameElement;
    if (iframe && savedIframeStyles) {{
      Object.keys(savedIframeStyles).forEach(function(k) {{ iframe.style[k] = savedIframeStyles[k] || ''; }});
    }}
  }} catch(e) {{}}
  fsContainer.classList.remove('pseudo-fs');
  document.body.style.overflow = '';
  isFS = false;
  setTimeout(resizeChartsFS, 80);
}}
function resizeChartsFS() {{
  if (isFS) {{
    var tbH = document.getElementById('toolbar').offsetHeight;
    var screenH = window.innerHeight || document.documentElement.clientHeight;
    var oscH = 0;
    syncCharts.forEach(function(){{ oscH += 120; }});
    var volH = 100;
    var mainH = screenH - tbH - volH - oscH - 4;
    if (mainH < 300) mainH = 300;
    document.getElementById('main_panel').style.height = mainH + 'px';
  }} else {{
    document.getElementById('main_panel').style.height = '520px';
  }}
  mainChart.timeScale().fitContent();
  resizeCanvas(); renderAll();
}}
</script></body></html>"""
    return html
