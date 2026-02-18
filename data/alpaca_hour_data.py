import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import os
from dotenv import load_dotenv # ⬅️ 추가: .env 파일을 로드하는 라이브러리
from datetime import datetime

# ==========================================
# [1] 설정 영역
# ==========================================
# .env 파일 로드
# 이 함수는 스크립트가 실행되는 폴더에서 .env 파일을 찾습니다.
load_dotenv()

# 1. Alpaca API 키 설정 (🔑 .env 파일에서 불러오기)
# 환경 변수에 키가 없으면 에러를 발생시킵니다.
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("❌ .env 파일에서 ALPACA_API_KEY 또는 ALPACA_SECRET_KEY를 찾을 수 없습니다.")

# 2. 수집할 기간 설정 (YYYY-MM-DD)
START_DATE = "2016-01-01"  
END_DATE = "2025-12-01"

# 3. 대상 종목 리스트
TARGET_TICKERS = ['AAPL', 'TSLA', 'NVDA', 'PLTR', 'AMZN', 'GOOGL', 'AMD', 'META']

# 4. 저장 경로
SAVE_DIR = "E:/b/pj2/data"

# ==========================================
# [2] API 연결 및 데이터 수집
# ==========================================
def fetch_alpaca_data():
    print(f"🚀 Alpaca API로 데이터 수집을 시작합니다. ({START_DATE} ~ {END_DATE})")
    
    # 저장 폴더 생성
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # API 연결 (키가 .env에서 자동으로 로드됩니다)
    api = tradeapi.REST(
        key_id=API_KEY,
        secret_key=SECRET_KEY,
        base_url='https://paper-api.alpaca.markets',
        api_version='v2'
    )

    for ticker in TARGET_TICKERS:
        print(f"[{ticker}] 다운로드 중...", end=" ")
        
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

            # --- 데이터 전처리 부분은 이전과 동일 ---
            bars.index = bars.index.tz_convert(None)
            bars = bars[['open', 'high', 'low', 'close', 'volume']]
            bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            bars.reset_index(inplace=True)
            bars.rename(columns={'timestamp': 'Datetime'}, inplace=True)
            # --- ---

            file_path = f"{SAVE_DIR}/{ticker}_hourly_alpaca.csv"
            bars.to_csv(file_path, index=False)
            
            print(f"✅ 완료 ({len(bars)} rows)")

        except Exception as e:
            print(f"❌ 에러 발생: {e}")

    print("\n🏁 모든 종목 데이터 수집이 완료되었습니다.")

if __name__ == "__main__":
    fetch_alpaca_data()