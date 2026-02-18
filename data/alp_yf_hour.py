import os
import tempfile

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# ==========================================
# [1] 설정 영역
# ==========================================
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("❌ .env 파일에서 ALPACA_API_KEY 또는 ALPACA_SECRET_KEY를 찾을 수 없습니다.")

# 기간 및 기본 설정
START_DATE = "2016-01-01"
END_DATE = "2025-12-01"
INTERVAL = "1h"
OUTPUT_TIMEZONE = "America/New_York"  # 모든 시계열을 뉴욕 시간으로 통일

SAVE_DIR = "E:/b/pj2/data"
TARGET_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'NVDA', 'AMD', 'META', 'PLTR', 'TSLA']

# 거시지표는 yfinance 일봉으로만 수집
MACRO_TICKERS = {
    'VIX': '^VIX',
    'TNX': '^TNX',
    'DXY': 'DX-Y.NYB'
}

# 시장 ETF는 Alpaca에서 시세를 받아 각 파일 열로만 추가
MARKET_ETFS = ['QQQ', 'XLK']

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# [2] 보조 함수 (RSI, MACD, ATR, VWAP)
# ==========================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (변동성 지표)"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP: 거래량 가중 평균 가격 (세션 전체 기준 누적)"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    cum_vp = (typical_price * df['Volume']).cumsum()
    cum_vol = df['Volume'].cumsum()
    vwap = cum_vp / cum_vol
    return vwap


# ==========================================
# [3] 거시지표(yfinance, 일봉) 수집
# ==========================================

def fetch_macro_series():
    print(f"⏰ 일별 거시 시장 지표 수집 중... ({START_DATE} ~ {END_DATE})")
    macro_series = {}
    available_macro_cols = []

    for name, ticker in MACRO_TICKERS.items():
        try:
            # 거시지표는 모두 일봉(1d)으로 수집 후, 이후 시간 인덱스에 채워서 사용
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval='1d',
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
        joined = ", ".join(available_macro_cols)
        print(f"✅ 수집된 거시 지표: {joined}")
    else:
        print("⚠️ 거시 지표 데이터를 찾을 수 없습니다.")

    return macro_series, available_macro_cols


# ==========================================
# [4] ETF (QQQ, XLK) 시세 수집 (Alpaca)
# ==========================================

def fetch_market_etfs(api: tradeapi.REST):
    etf_series = {}
    for symbol in MARKET_ETFS:
        try:
            bars = api.get_bars(
                symbol=symbol,
                timeframe=TimeFrame.Hour,
                start=START_DATE,
                end=END_DATE,
                adjustment='raw'
            ).df

            if bars.empty:
                print(f"⚠️ {symbol} 데이터 없음 (Alpaca)")
                continue

            idx = pd.to_datetime(bars.index)
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize('UTC')
            else:
                idx = idx.tz_convert('UTC')
            idx = idx.tz_convert(OUTPUT_TIMEZONE)
            bars.index = idx.tz_localize(None)

            etf_series[symbol] = bars['close'].sort_index().copy()
            print(f"✅ {symbol} 시세 수집 완료 (Alpaca)")
        except Exception as e:
            print(f"⚠️ {symbol} 수집 실패 (Alpaca): {e}")
    return etf_series


# ==========================================
# [5] Alpaca 시세 + 거시/기술/ETF 지표 결합
# ==========================================

def build_alpaca_yf_hourly_dataset():
    print(f"🚀 Alpaca 시세 + yfinance 거시 + 기술지표 + ETF(QQQ, XLK) 결합 데이터 생성 ({START_DATE} ~ {END_DATE}, {INTERVAL})")

    api = tradeapi.REST(
        key_id=API_KEY,
        secret_key=SECRET_KEY,
        base_url='https://paper-api.alpaca.markets',
        api_version='v2'
    )

    macro_series, available_macro_cols = fetch_macro_series()
    etf_series = fetch_market_etfs(api)

    print("\n🚀 개별 종목 시간별 데이터셋 생성 시작...")

    for ticker in TARGET_TICKERS:
        print(f"[{ticker}] 처리 중...", end=" ")
        try:
            bars = api.get_bars(
                symbol=ticker,
                timeframe=TimeFrame.Hour,
                start=START_DATE,
                end=END_DATE,
                adjustment='raw'
            ).df

            if bars.empty:
                print("❌ 데이터 없음")
                continue

            # 인덱스(시간) 처리 및 뉴욕 시간으로 변환
            index = pd.to_datetime(bars.index)
            if getattr(index, "tz", None) is None:
                index = index.tz_localize('UTC')
            else:
                index = index.tz_convert('UTC')
            index = index.tz_convert(OUTPUT_TIMEZONE)
            bars.index = index.tz_localize(None)

            # 기본 컬럼 정리 (Open, High, Low, Close, Volume)
            bars = bars[['open', 'high', 'low', 'close', 'volume']]
            bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            df = bars.sort_index().copy()

            # --- 기술적 지표 추가 ---
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['MACD_Signal'], _ = calculate_macd(df['Close'])
            df['ATR'] = calculate_atr(df)
            df['VWAP'] = calculate_vwap(df)

            # --- 거시지표 결합 (일봉을 시간축에 맞춰 채우기) ---
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
                    aligned = aligned.ffill().bfill()
                    macro_to_join[name] = aligned
                    valid_macro_cols.append(name)

                if valid_macro_cols:
                    df = df.join(macro_to_join[valid_macro_cols], how='left')
                    df[valid_macro_cols] = df[valid_macro_cols].ffill().bfill()

            # --- ETF(QQQ, XLK) 시세 결합 ---
            etf_added_cols = []
            if etf_series:
                for symbol, series in etf_series.items():
                    aligned = series.reindex(df.index, method='ffill')
                    if aligned.dropna().empty:
                        continue
                    aligned = aligned.ffill().bfill()
                    df[symbol] = aligned
                    etf_added_cols.append(symbol)

            # 시간 파생 변수
            df['DayOfWeek'] = df.index.dayofweek
            df['Hour'] = df.index.hour

            # 필수 컬럼 기준으로 NaN 제거
            required_columns = ['Close', 'Volume', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'VWAP']
            if valid_macro_cols:
                required_columns += valid_macro_cols
            if etf_added_cols:
                required_columns += etf_added_cols

            original_len = len(df)
            df.dropna(subset=required_columns, inplace=True)

            if len(df) == 0:
                print(f"⚠️ 데이터가 모두 삭제되었습니다. (병합 문제 가능성) 원본: {original_len}행")
                continue

            # 인덱스를 Datetime 컬럼으로 변환
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Datetime'}, inplace=True)

            # 파일 저장 (기존 경로 + 새로운 파일명 패턴)
            file_path = f"{SAVE_DIR}/{ticker}_hourly_alp_yf_dataset_v2.csv"
            temp_path = None
            try:
                fd, temp_path = tempfile.mkstemp(prefix=f"{ticker}_", suffix="_hourly_alp_yf_v2_tmp.csv", dir=SAVE_DIR)
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

    print("\n🏁 지정된 기간의 Alpaca+거시+기술+ETF 시간별 데이터 수집 완료.")


if __name__ == "__main__":
    build_alpaca_yf_hourly_dataset()
