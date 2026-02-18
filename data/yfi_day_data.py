import yfinance as yf
import pandas as pd
import numpy as np
import os

SAVE_DIR = "E:/b/pj2/data"
TARGET_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'NVDA', 'AMD', 'META', 'PLTR', 'TSLA']
START_DATE = "2015-01-01"
END_DATE = "2025-12-01"

MACRO_TICKERS = {
    'VIX': '^VIX',
    'TNX': '^TNX',
    'DXY': 'DX-Y.NYB',
    'QQQ': 'QQQ'
}

os.makedirs(SAVE_DIR, exist_ok=True)


def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist


print(f"🌍 시장 지표 수집 중... ({START_DATE} ~ {END_DATE})")
macro_df = pd.DataFrame()
available_macro_cols = []

for name, ticker in MACRO_TICKERS.items():
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
        if df.empty:
            print(f"⚠️ {name} 수집 실패: 데이터가 비어 있습니다.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col_to_use = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        series = df[col_to_use]
        if series.dropna().empty:
            print(f"⚠️ {name} 수집 실패: 유효한 종가 데이터가 없습니다.")
            continue
        macro_df[name] = series
        available_macro_cols.append(name)
    except Exception as e:
        print(f"⚠️ {name} 수집 실패: {e}")

if not macro_df.empty:
    macro_df.index = macro_df.index.tz_localize(None)
else:
    print("⚠️ 시장 지표 데이터를 찾을 수 없습니다.")


print("\n🚀 개별 종목 데이터셋 생성 시작...")

for ticker in TARGET_TICKERS:
    print(f"[{ticker}] 처리 중...", end=" ")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
        if df.empty:
            print("❌ 데이터 없음")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None)
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], _ = calculate_macd(df['Close'])

        if not macro_df.empty and available_macro_cols:
            df = df.join(macro_df[available_macro_cols], how='left')
            df[available_macro_cols] = df[available_macro_cols].ffill().bfill()

        df['DayOfWeek'] = df.index.dayofweek

        required_columns = ['Close', 'Volume', 'MA20', 'RSI', 'MACD', 'MACD_Signal']
        if available_macro_cols:
            required_columns += available_macro_cols
        original_len = len(df)
        df.dropna(subset=required_columns, inplace=True)

        if len(df) == 0:
            print(f"⚠️ 데이터가 모두 삭제되었습니다. (병합 문제 가능성) 원본: {original_len}행")
            continue

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        file_path = f"{SAVE_DIR}/{ticker}_daily_dataset.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ 완료 ({len(df)}행)")
    except Exception as e:
        print(f"❌ 에러: {e}")

print("\n🏁 지정된 기간의 데이터 수집 완료.")