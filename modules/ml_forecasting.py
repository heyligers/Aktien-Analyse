import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

def train_and_predict_trend_sklearn(df, future_days=30):
    """Trainiert eine polynomiale Regression (Scikit-Learn) für einen visuellen Trend-Forecast."""
    df_copy = df.copy()
    df_copy.dropna(subset=['Close'], inplace=True)
    
    df_copy['seq'] = np.arange(len(df_copy))
    X = df_copy[['seq']].values
    y = df_copy['Close'].values
    
    from sklearn.preprocessing import StandardScaler
    # Lineare Regression (Grad 1) verhindert utopische U-Kurven und Abstürze
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=1), Ridge(alpha=100.0))
    model.fit(X, y)
    
    df_copy['yhat'] = model.predict(X)
    
    last_seq = X[-1][0]
    future_seqs = np.arange(last_seq + 1, last_seq + 1 + future_days).reshape(-1, 1)
    future_preds = model.predict(future_seqs)
    
    residuals = y - df_copy['yhat']
    std_error = np.std(residuals)
    
    volatility_expansion = np.linspace(1.2, 3.0, future_days)
    
    last_date = df_copy.index[-1]
    future_dates = []
    days_added = 0
    current_date = last_date
    while days_added < future_days:
        current_date += datetime.timedelta(days=1)
        if current_date.weekday() < 5: 
            future_dates.append(current_date)
            days_added += 1
            
    np.random.seed(42)
    recent_swings = np.diff(y[-60:]) if len(y) >= 60 else np.diff(y)
    raw_noise = np.random.choice(recent_swings, size=future_days)
    smooth_noise = pd.Series(raw_noise).rolling(window=3, min_periods=1).mean().values
    
    # Shift the whole trendline to start at the last known price to prevent large jumps
    gap = y[-1] - future_preds[0]
    adjusted_preds = future_preds + gap
    
    synthetic_path = adjusted_preds + (smooth_noise * 1.5)
    synthetic_path[0] = y[-1]
    
    future_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': adjusted_preds,
        'synthetic_path': synthetic_path,
        'yhat_lower': adjusted_preds - (std_error * volatility_expansion),
        'yhat_upper': adjusted_preds + (std_error * volatility_expansion)
    })
    
    hist_df = pd.DataFrame({
        'ds': df_copy.index,
        'yhat': df_copy['yhat']
    })
    
    return future_df, hist_df

def train_sklearn_classification(df, forward_days=5, transaction_cost=0.003):
    """
    Trainiert einen Gradient Boosting / Random Forest Classifier mit stationären Makro- und Lagged-Features.
    Beinhaltet einen Out-of-Sample Backtest inkl. Slippage.
    """
    df_feat = df.copy()
    original_cols = list(df_feat.columns)
    
    # 1. Stationäre Technische Indikatoren generieren (Oszillatoren, keine absoluten Preise!)
    df_feat.ta.rsi(length=14, append=True)
    df_feat.ta.macd(append=True)
    df_feat.ta.atr(length=14, append=True)
    df_feat.ta.adx(length=14, append=True)
    df_feat.ta.cci(length=14, append=True)
    df_feat.ta.roc(length=10, append=True)
    
    # Distanz-Features (Prozentualer Abstand zu Moving Averages statt absoluter Preis)
    df_feat['Dist_SMA20'] = df_feat['Close'] / df_feat['Close'].rolling(20).mean() - 1
    df_feat['Dist_SMA50'] = df_feat['Close'] / df_feat['Close'].rolling(50).mean() - 1
    df_feat['Dist_SMA200'] = df_feat['Close'] / df_feat['Close'].rolling(200).mean() - 1
    
    # Bollinger Band Width (Volatilitäts-Indikator, stationär)
    sma20 = df_feat['Close'].rolling(20).mean()
    std20 = df_feat['Close'].rolling(20).std()
    df_feat['BB_Width'] = (std20 * 4) / sma20
        
    new_ta_cols = [c for c in df_feat.columns if c not in original_cols]
    
    # 2. Log-Returns statt absoluter Preise für Lags
    # Verwenden von natürlichem Logarithmus für symmetrische Returns
    df_feat['LogRet_1d'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
    df_feat['LogRet_3d'] = np.log(df_feat['Close'] / df_feat['Close'].shift(3))
    df_feat['LogRet_5d'] = np.log(df_feat['Close'] / df_feat['Close'].shift(5))
    
    # Ausgewählte Indikatoren in die Vergangenheit shiften
    target_lag_cols = [c for c in new_ta_cols if 'RSI' in c or 'MACD' in c or 'ADX' in c]
    for col in target_lag_cols:
        df_feat[f"{col}_Lag1"] = df_feat[col].shift(1)
        df_feat[f"{col}_Lag2"] = df_feat[col].shift(2)
        
    # 3. Macro Features (Der breite Markt: S&P 500)
    try:
        start_dt = df_feat.index[0]
        end_dt = df_feat.index[-1] + datetime.timedelta(days=1)
        spy = yf.download('^GSPC', start=start_dt, end=end_dt, progress=False)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            # LogReturns für den S&P 500
            df_feat['Market_Ret_1d'] = np.log(spy['Close'] / spy['Close'].shift(1))
            df_feat['Market_Ret_5d'] = np.log(spy['Close'] / spy['Close'].shift(5))
    except Exception as e:
        pass 
        
    df_feat.dropna(inplace=True)
    
    if len(df_feat) < 100:
        return None, 0, None, None, None
        
    df_feat['Future_Close'] = df_feat['Close'].shift(-forward_days)
    # Log Return für das Target-Fenster
    df_feat['Future_Return'] = df_feat['Future_Close'] / df_feat['Close'] - 1.0
    df_feat['Target'] = np.where(df_feat['Future_Close'] > df_feat['Close'], 1, 0)
    
    # Training Set ohne die letzten offenen Tage
    train_df = df_feat.iloc[:-forward_days].copy()
    current_features = df_feat.iloc[[-1]].copy()
    
    # Sicherstellen, dass absolut KEINE unbereinigten, nicht-stationären Preisdaten im Feature-Set landen
    ignore_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Future_Close', 'Future_Return', 'Target']
    features = [c for c in train_df.columns if c not in ignore_cols]
    
    X = train_df[features]
    y = train_df['Target']
    
    # CHRONOLOGISCHER SPLIT (Walk-forward Validation statt Data Leakage)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Random Forest: Mit stärkerer Regularisierung (Overfitting verhindern)
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=6,             # Statt 10 -> Weniger Auswendiglernen
        min_samples_leaf=10,     # Statt 2 -> Robustere Blätter
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # ----------------------------------------------------
    # OOS Backtester Simulation (Mit Slippage & Gebühren)
    # ----------------------------------------------------
    test_idx = X_test.index
    # Realer Kursgewinn/Verlust, den ein Trader im Zeitraum erzielt hätte (über forward_days)
    test_returns = train_df.loc[test_idx, 'Future_Return'].values
    
    # Um unrealistische Zinseszins-Effekte (+3000%) durch überlappende Trades zu verhindern,
    # handeln wir sequenziell: Wenn wir kaufen, blockieren wir neue Trades für "forward_days" Tage.
    net_returns_on_trades = []
    cooldown = 0
    
    for i in range(len(y_pred)):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        if y_pred[i] == 1: # Modells prognostiziert Up-Trend
            # Return nach Transaktionskosten (Slippage + Gebühren)
            ret = test_returns[i] - transaction_cost
            net_returns_on_trades.append(ret)
            # Kapital ist nun für forward_days gebunden (wir können es nicht morgen direkt wieder neu investieren)
            cooldown = forward_days - 1
            
    # Berechnung des kumulierten Netto-Gewinns auf das Startkapital (sequenzielle Aufzinsung)
    total_net_profit_pct = (np.prod([1 + r for r in net_returns_on_trades]) - 1) * 100 if len(net_returns_on_trades) > 0 else 0.0
    
    # Buy and Hold Benchmark für den EXAKT GLEICHEN Test-Zeitraum
    start_price = train_df.loc[test_idx[0], 'Close']
    end_price = train_df.loc[test_idx[-1], 'Future_Close']
    bh_profit_pct = (end_price / start_price - 1) * 100
    
    backtest_metrics = {
        "net_profit_pct": total_net_profit_pct,
        "buy_and_hold_pct": bh_profit_pct,
        "trades_taken": len(net_returns_on_trades),
        "total_test_days": len(X_test)
    }
    
    # Aktuelle Prognose
    X_current = current_features[features]
    prob = model.predict_proba(X_current)[0]
    
    return prob, accuracy, model.feature_importances_, features, backtest_metrics
