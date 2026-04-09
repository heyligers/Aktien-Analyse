import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go

from modules.data_api import load_data, get_ticker, get_ticker_info
from modules.ml_forecasting import train_and_predict_trend_sklearn, train_sklearn_classification

def run():
    st.set_page_config(page_title="KI Forecast", page_icon="🔮", layout="wide")

    
    st.markdown("<h1>Prognose (Machine Learning)</h1>", unsafe_allow_html=True)
    st.markdown("Nutze KI und statistische Modelle aus Scikit-Learn, um zukünftige Kursbewegungen abzuschätzen.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Aktie auswählen")
        query = st.text_input("Ticker oder Name", "NVDA")
        ticker = get_ticker(query)
        info = get_ticker_info(ticker)
        st.write(f"**{info['name']}** ({ticker})")
        
        st.markdown("---")
        st.subheader("Modell Konfiguration")
        future_days = st.slider("Prognose-Tage (Trend)", 10, 90, 30, 5, help="Gibt an, wie viele Tage der Regressions-Trendkanal in die Zukunft gezeichnet werden soll.")
        forward_days = st.selectbox("Ziel-Horizont (Muster)", [1, 3, 5, 10, 20], index=2, help="Hier legst du fest, ob die aktuelle Situation (anhand von Indikatoren) nach X Tagen historisch betrachtet eher zu steigenden oder fallenden Kursen geführt hat.")
        
        run_btn = st.button("KI-Prognose starten", use_container_width=True, type="primary")
        
    with col2:
        if run_btn:
            with st.spinner(f"Lade historische Daten (10 Jahre) für {ticker}..."):
                # 10 Jahre Historie für ein robusteres ML-Modell
                end = datetime.datetime.today().date()
                start = end - datetime.timedelta(days=3650)
                df = load_data(ticker, start, end, "1d")
                
            if df.empty or len(df) < 200:
                st.error("Nicht genug historische Daten gefunden.")
                return
                
            tab_trend, tab_muster = st.tabs(["Zeitreihen-Trend (Regression)", "Mustererkennung (Klassifikation)"])
            
            with tab_trend:
                st.markdown("### Scikit-Learn Trend Forecast")
                st.info("Dieses Modell nutzt eine *Polynomiale Ridge Regression*, um den mittelfristigen Trend aus dem Chartverlauf zu projizieren.")
                
                with st.spinner("Trainiere Regressions-Modell..."):
                    # Für den rein visuellen, kurzfristigen Trend schneiden wir die Daten auf 
                    # das letzte Trading-Jahr (ca. 250 Tage) ab. 10 Jahre Historie würden 
                    # den kurzfristigen 30-Tage Trend mathematisch komplett verzerren (Hyperbel-Effekt).
                    df_trend = df.tail(250)
                    future_df, hist_df = train_and_predict_trend_sklearn(df_trend, future_days)
                    
                    fig = go.Figure()
                    
                    # Historischer Kurs (nur das letzte Jahr für bessere Sichtbarkeit)
                    fig.add_trace(go.Scatter(
                        x=df_trend.index, y=df_trend['Close'],
                        mode='lines', name='Historischer Kurs',
                        line=dict(color='rgba(255, 255, 255, 0.7)', width=2)
                    ))
                    
                    # Modell Fit (Historische Trendlinie)
                    fig.add_trace(go.Scatter(
                        x=hist_df['ds'], y=hist_df['yhat'],
                        mode='lines', name='Trend (Fit)',
                        line=dict(color='rgba(0, 176, 255, 0.4)', width=1)
                    ))
                    
                    # Risiko-Korridor der Zukunft
                    fig.add_trace(go.Scatter(
                        name='Untere Grenze',
                        x=future_df['ds'], y=future_df['yhat_lower'],
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        name='Risiko-Korridor',
                        x=future_df['ds'], y=future_df['yhat_upper'],
                        mode='lines', line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(0, 176, 255, 0.15)', showlegend=True
                    ))
                    
                    # Simulierter Realistischer Pfad
                    fig.add_trace(go.Scatter(
                        x=future_df['ds'], y=future_df['synthetic_path'],
                        mode='lines', name='Wahrscheinlicher Verlauf',
                        line=dict(color='rgba(255, 235, 59, 1.0)', width=2)
                    ))
                    
                    # Glatte Trend-Prognose
                    fig.add_trace(go.Scatter(
                        x=future_df['ds'], y=future_df['yhat'],
                        mode='lines', name='Regressions-Mittellinie',
                        line=dict(color='rgb(0, 176, 255)', width=2, dash='dot')
                    ))
                    
                    # Markierung für den Zielkurs am Ende
                    last_day = future_df.iloc[-1]
                    fig.add_trace(go.Scatter(
                        x=[last_day['ds']], y=[last_day['yhat']],
                        mode='markers+text', name='Zielkurs',
                        marker=dict(color='rgb(0, 176, 255)', size=10),
                        text=[f"{last_day['yhat']:.2f}"],
                        textposition="top right",
                        textfont=dict(color='rgb(0, 176, 255)', size=14)
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        title=f"Trend Projektion für {future_days} Tage",
                        xaxis_title="Datum",
                        yaxis_title="Kurs",
                        hovermode="x unified",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            with tab_muster:
                st.markdown("### Random Forest Indikator-Analyse")
                st.info(f"Der **Random Forest Classifier** bewertet anhand der technischen Indikatoren, deren Steigung und des S&P500 die Aufwärts-Wahrscheinlichkeit nach **{forward_days} Tagen**.")
                
                with st.spinner("Trainiere Random Forest..."):
                    result = train_sklearn_classification(df, forward_days)
                    
                    if result[0] is None:
                        st.error("Nicht genug Daten, um die Indikatoren präzise zu berechnen.")
                    else:
                        prob, accuracy, importances, features, bt_metrics = result
                        prob_down, prob_up = prob[0], prob[1]
                        
                        st.markdown(f"#### Historische Test-Genauigkeit (Out-of-Sample): **{accuracy*100:.1f} %**")
                        if accuracy < 0.5:
                            st.warning("⚠️ Vorsicht: Das Modell schneidet auf den ungesehenen Testdaten schlechter als ein Münzwurf ab. Die Indikatoren haben hier scheinbar keine Prognosekraft.")
                        
                        cols = st.columns(2)
                        with cols[0]:
                            color = "rgb(0, 200, 83)" if prob_up > 0.5 else "white"
                            st.markdown(f"<h3 style='color: {color}; text-align: center'>Aufwärts</h3>", unsafe_allow_html=True)
                            st.markdown(f"<h1 style='color: {color}; text-align: center'>{prob_up*100:.1f} %</h1>", unsafe_allow_html=True)
                        with cols[1]:
                            color = "rgb(255, 82, 82)" if prob_down > 0.5 else "white"
                            st.markdown(f"<h3 style='color: {color}; text-align: center'>Abwärts</h3>", unsafe_allow_html=True)
                            st.markdown(f"<h1 style='color: {color}; text-align: center'>{prob_down*100:.1f} %</h1>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("### Realitäts-Check: ML Backtest (inkl. Slippage)")
                        st.info("Im maschinellen Lernen reicht eine hohe Trefferquote oft nicht aus, weil jede ausgeführte Transaktion Gebühren und Spread kostet (hier 0.3% simuliert).")
                        
                        net_profit = bt_metrics['net_profit_pct']
                        bh_profit = bt_metrics['buy_and_hold_pct']
                        is_profitable = net_profit > 0
                        beats_market = net_profit > bh_profit
                        
                        metric_cols = st.columns(3)
                        metric_cols[0].metric(f"Netto Rendite ({bt_metrics['total_test_days']} Tage)", f"{net_profit:+.2f} %", "Nach Gebühren" if is_profitable else "-Nach Gebühren")
                        metric_cols[1].metric("Buy & Hold (Benchmark)", f"{bh_profit:+.2f} %", None)
                        metric_cols[2].metric("Ausgeführte Trades", bt_metrics['trades_taken'], None)
                        
                        if is_profitable and beats_market:
                            st.success("✅ **Profitabel!** Das Modell konnte den Buy & Hold Ansatz auf den ungesehenen Testdaten ( Out-of-Sample) inklusive realer Kosten schlagen.")
                        elif is_profitable:
                            st.warning("⚠️ **Profitabel, aber schwächer als Buy & Hold.** Die Taktik macht zwar Gewinn, aber stures Halten wäre durch weniger Ordergebühren besser gewesen.")
                        else:
                            st.error("🛑 **Verlustgeschäft.** Durch Slippage und/oder falsche Prognosen verbrennt das Modell im Live-Test Geld. Die Signale sollten nicht blind gehandelt werden.")
                            
                        st.markdown("---")
                        # Balkendiagramm für Indikator-Relevanz
                        feat_df = pd.DataFrame({'Feature': features, 'Wichtigkeit': importances})
                        feat_df = feat_df.sort_values(by='Wichtigkeit', ascending=True).tail(10)
                        
                        fig_feat = go.Figure(go.Bar(
                            x=feat_df['Wichtigkeit'],
                            y=feat_df['Feature'],
                            orientation='h',
                            marker_color='rgba(0, 176, 255, 0.7)'
                        ))
                        fig_feat.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=350, title="Top 10 Entscheidungs-Faktoren (Feature Importance)"
                        )
                        st.plotly_chart(fig_feat, use_container_width=True)

if __name__ == "__main__":
    run()
