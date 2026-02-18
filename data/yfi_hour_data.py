import yfinance as yf
import pandas as pd
import os
import tempfile

SAVE_DIR = "E:/b/pj2/data"
TARGET_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'NVDA', 'AMD', 'META', 'PLTR', 'TSLA']
START_DATE = "2024-01-01"
END_DATE = "2025-12-12"
INTERVAL = "1h"
OUTPUT_TIMEZONE = "America/New_York"  # 모든 시계열을 뉴욕 시간으로 통일

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


print(f"⏰ 시간별 시장 지표 수집 중... ({START_DATE} ~ {END_DATE}, interval={INTERVAL})")
macro_series = {}
available_macro_cols = []

for name, ticker in MACRO_TICKERS.items():
    try:
        # TNX(미국 10년물)는 yfinance가 인트라데이(1h)를 제공하지 않을 수 있으므로
        # 일간(1d)으로 받아서 이후 시간 인덱스에 동일값을 채웁니다.
        use_interval = '1d' if name == 'TNX' else INTERVAL
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval=use_interval,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            print(f"⚠️ {name} 수집 실패: 데이터가 비어 있습니다.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col_to_use = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        series = df[col_to_use].copy()
        # TNX는 일간으로 받아왔으므로 인덱스를 날짜(시간 00:00)로 유지합니다.
        if series.dropna().empty:
            print(f"⚠️ {name} 수집 실패: 유효한 종가 데이터가 없습니다.")
            continue
        index = pd.to_datetime(series.index)
        if getattr(index, "tz", None) is None:
            index = index.tz_localize('UTC')
        else:
            index = index.tz_convert('UTC')
        index = index.tz_convert(OUTPUT_TIMEZONE)
        series.index = index.tz_localize(None)
        series.sort_index(inplace=True)

        macro_series[name] = series
        available_macro_cols.append(name)
    except Exception as e:
        print(f"⚠️ {name} 수집 실패: {e}")

if available_macro_cols:
    joined_indicators = ", ".join(available_macro_cols)
    print(f"✅ 수집된 시장 지표: {joined_indicators}")
else:
    print("⚠️ 시장 지표 데이터를 찾을 수 없습니다.")


print("\n🚀 개별 종목 시간별 데이터셋 생성 시작...")

for ticker in TARGET_TICKERS:
    print(f"[{ticker}] 처리 중...", end=" ")
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            print("❌ 데이터 없음")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        index = pd.to_datetime(df.index)
        if getattr(index, "tz", None) is None:
            index = index.tz_localize('UTC')
        else:
            index = index.tz_convert('UTC')
        index = index.tz_convert(OUTPUT_TIMEZONE)
        df.index = index.tz_localize(None)
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], _ = calculate_macd(df['Close'])

        valid_macro_cols = []
        if available_macro_cols:
            macro_to_join = pd.DataFrame(index=df.index)
            for name in available_macro_cols:
                series = macro_series.get(name)
                if series is None:
                    continue
                aligned = series.reindex(df.index, method='ffill')
                if aligned.dropna().empty:
                    continue
                # 보간 전후로 양 끝단이 비어 있지 않도록 보완합니다.
                aligned = aligned.ffill().bfill()
                macro_to_join[name] = aligned
                valid_macro_cols.append(name)

            if valid_macro_cols:
                df = df.join(macro_to_join[valid_macro_cols], how='left')
                df[valid_macro_cols] = df[valid_macro_cols].ffill().bfill()

        df['DayOfWeek'] = df.index.dayofweek
        df['Hour'] = df.index.hour

        required_columns = ['Close', 'Volume', 'MA20', 'RSI', 'MACD', 'MACD_Signal']
        if valid_macro_cols:
            required_columns += valid_macro_cols
        original_len = len(df)
        df.dropna(subset=required_columns, inplace=True)

        if len(df) == 0:
            print(f"⚠️ 데이터가 모두 삭제되었습니다. (병합 문제 가능성) 원본: {original_len}행")
            continue

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Datetime'}, inplace=True)
        file_path = f"{SAVE_DIR}/{ticker}_hourly_dataset.csv"
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(prefix=f"{ticker}_", suffix="_hourly_tmp.csv", dir=SAVE_DIR)
            os.close(fd)
            df.to_csv(temp_path, index=False)
            os.replace(temp_path, file_path)
        except PermissionError:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            print("❌ 에러: 대상 CSV가 열려 있어 저장할 수 없습니다. 파일을 닫고 다시 실행하세요.")
            continue
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        print(f"✅ 완료 ({len(df)}행)")
    except Exception as e:
        print(f"❌ 에러: {e}")

print("\n🏁 지정된 기간의 시간별 데이터 수집 완료.")
