import requests
import json
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# ==========================================
# [1] API 키 로드
# ==========================================
KEY_FILE_PATH = "key.json"

try:
    with open(KEY_FILE_PATH, 'r', encoding='utf-8') as f:
        secrets = json.load(f)
    APP_KEY = secrets['APP_KEY']
    APP_SECRET = secrets['APP_SECRET']
    print(f"✅ API 키 파일 로드 완료")

except Exception as e:
    print(f"❌ 키 파일 오류: {e}")
    exit()

# ==========================================
# [2] 설정
# ==========================================
IS_VIRTUAL = False  # 실전 서버 (데이터 조회용)
TICKERS = ['AAPL', 'TSLA', 'NVDA', 'PLTR', 'AMZN', 'GOOGL']

if IS_VIRTUAL:
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
else:
    URL_BASE = "https://openapi.koreainvestment.com:9443"

EXCHANGE_MAP = {
    'AAPL': 'NAS', 'TSLA': 'NAS', 'NVDA': 'NAS',
    'PLTR': 'NAS', 'AMZN': 'NAS', 'GOOGL': 'NAS',
    'MSFT': 'NAS'
}

# ==========================================
# [3] 토큰 캐시 설정 (token.txt)
# ==========================================
TOKEN_FILE = "token.txt"
TOKEN_EXPIRE_SECONDS = 24 * 60 * 60  # 24시간


def get_access_token():
    """
    1) token.txt 에 저장된 토큰이 있으면 불러와서 만료 여부 확인
    2) 없거나 만료되었으면 새 토큰 발급 후 token.txt에 저장
    """
    token_info = None

    # 1) token.txt에서 기존 토큰 로드
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                token_info = json.load(f)
        except Exception as e:
            print(f"⚠️ token.txt 읽기 오류: {e}")
            token_info = None

    # 2) 저장된 토큰이 있고, 형식이 정상이라면 만료 체크
    if token_info:
        access_token = token_info.get("access_token")
        issued_at = token_info.get("issued_at")

        if access_token and issued_at:
            elapsed = time.time() - issued_at
            if elapsed < TOKEN_EXPIRE_SECONDS:
                print(f"✅ 저장된 토큰 사용 (경과 시간: {elapsed / 3600:.2f}시간)")
                return access_token
            else:
                print("⏳ 저장된 토큰 만료됨 → 새 토큰 발급 시도")
        else:
            print("⚠️ token.txt 내용이 이상함 → 새 토큰 발급")

    else:
        print("📌 token.txt 없음 → 새 토큰 발급")

    # 3) 새 토큰 발급
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    url = f"{URL_BASE}/oauth2/tokenP"

    res = requests.post(url, headers=headers, data=json.dumps(body))
    print("🔑 토큰 발급 응답 상태:", res, res.text)

    # HTTP 에러 시 예외
    res.raise_for_status()

    data = res.json()
    access_token = data.get("access_token")

    if not access_token:
        raise RuntimeError(f"❌ 토큰 발급 실패: {data}")

    # 4) token.txt에 저장
    try:
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "access_token": access_token,
                    "issued_at": time.time()
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        print("💾 token.txt 저장 완료")
    except Exception as e:
        print(f"⚠️ token.txt 저장 실패: {e}")

    return access_token


# ==========================================
# [4] 해외주식 기간별 일봉 조회 함수
#      (TR: HHDFS76240000, 범위 조회용)
# ==========================================
def get_overseas_ohlcv_range(token, ticker, start_day, end_day,
                             timeframe="D", adj_price=True):
    """
    해외 주식 기간별 시세 (일/주/월봉) 범위 조회
    - start_day, end_day: 'YYYYMMDD' 문자열
    - timeframe: 'D'(일), 'W'(주), 'M'(월)
    """
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "HHDFS76240000",  # 해외주식 기간별 시세
        "custtype": "P"
    }

    timeframe_lookup = {"D": "0", "W": "1", "M": "2"}

    # end_day가 없으면 오늘 날짜로
    if not end_day:
        end_day = datetime.now().strftime("%Y%m%d")

    all_rows = []

    params = {
        "AUTH": "",
        "EXCD": EXCHANGE_MAP.get(ticker, "NAS"),
        "SYMB": ticker,
        "GUBN": timeframe_lookup.get(timeframe, "0"),  # 0=일, 1=주, 2=월
        "BYMD": end_day,                               # 이 날짜 기준 과거로 조회
        "MODP": 1 if adj_price else 0,                 # 1=수정주가
    }

    while True:
        res = requests.get(
            f"{URL_BASE}/uapi/overseas-price/v1/quotations/dailyprice",
            headers=headers,
            params=params,
            timeout=10,
        )

        if res.status_code != 200:
            print(f"❌ HTTP 에러: {res.status_code} / {res.text}")
            break

        data = res.json()

        if data.get("rt_cd") != "0":
            print("⚠️ API 에러:", data.get("msg1"))
            break

        # 보통 output2가 리스트(일자별 데이터)
        rows = data.get("output2") or data.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]

        if not rows:
            break

        all_rows += rows

        # 응답의 마지막 날짜 확인 (더 과거로 갈 기준)
        last_date = rows[-1].get("xymd")
        if not last_date:
            break

        # start_day 도달/지나면 종료
        if last_date <= start_day:
            break

        # 더 과거 데이터 요청
        params["BYMD"] = last_date
        time.sleep(0.05)  # 너무 빠르게 연속 호출 방지

    # start_day ~ end_day 범위로 필터
    filtered = []
    for r in all_rows:
        d = r.get("xymd")
        if d is None:
            continue
        if start_day <= d <= end_day:
            filtered.append(r)

    # 날짜 최신순으로 왔을 가능성 있으니 정렬
    filtered.sort(key=lambda x: x.get("xymd"))

    return filtered


# ==========================================
# [4b] 해외주식 분(분봉) 조회 함수 (예시)
# ==========================================
def get_overseas_ohlcv_minute(token, ticker, start_dt, end_dt,
                              interval_minutes=1, adj_price=True):
    """
    해외 주식 분봉(예시 구현)
    - start_dt, end_dt: 'YYYYMMDDHHMM' 또는 datetime
    - interval_minutes: 분봉 간격 (1, 5, 15 등)

    NOTE: 실제 한국투자 API의 분봉 엔드포인트 및 파라미터 이름은
    문서와 다를 수 있습니다. 아래는 기본적인 호출 구조 예시이며,
    응답 필드명 또는 URL을 실제 값으로 바꾸어 사용하세요.
    """
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "custtype": "P"
    }

    # 분봉용 엔드포인트(필요시 실제 엔드포인트로 수정)
    endpoint = f"{URL_BASE}/uapi/overseas-price/v1/quotations/minuteprice"

    # 문자열 입력을 허용: datetime이면 변환
    def to_str(dt):
        if isinstance(dt, datetime):
            return dt.strftime("%Y%m%d%H%M")
        return str(dt)

    start_s = to_str(start_dt)
    end_s = to_str(end_dt)

    params = {
        "AUTH": "",
        "EXCD": EXCHANGE_MAP.get(ticker, "NAS"),
        "SYMB": ticker,
        # 아래 파라미터명은 예시입니다. 실제 문서에 맞게 수정하세요.
        "INTERVAL": interval_minutes,
        "STTM": start_s,  # start time
        "EDTM": end_s,    # end time
        "MODP": 1 if adj_price else 0,
    }

    try:
        res = requests.get(endpoint, headers=headers, params=params, timeout=15)
    except Exception as e:
        print(f"❌ 요청 실패: {e}")
        return []

    if res.status_code != 200:
        print(f"❌ HTTP 에러: {res.status_code} / {res.text}")
        return []

    data = res.json()

    # 응답 코드 확인 (API별 필드명이 다를 수 있음)
    if data.get("rt_cd") and data.get("rt_cd") != "0":
        print("⚠️ API 에러:", data.get("msg1") or data.get("message"))
        return []

    rows = data.get("output2") or data.get("output") or []
    if isinstance(rows, dict):
        rows = [rows]

    # 일부 API는 역순으로 내려올 수 있으니 정렬
    try:
        # xymdhm 또는 time 등의 필드명일 수 있음 — 우선 'xymdhm' 사용
        rows.sort(key=lambda x: x.get("xymdhm") or x.get("time") or "")
    except Exception:
        pass

    # 간단히 반환
    return rows


# ==========================================
# [5] 메인 실행
# ==========================================
if __name__ == "__main__":
    # 저장 경로
    save_dir = "E:/b/pj2/data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ✅ 조회 기간 설정 (최근 1년)
    today = datetime.now()
    end_day = today.strftime("%Y%m%d")
    start_day = (today - timedelta(days=365)).strftime("%Y%m%d")

    print(f"📅 조회 기간: {start_day} ~ {end_day}")

    try:
        token = get_access_token()
        print("✅ 최종 사용 토큰:", token)
        print("✅ 토큰 준비 완료, 일봉 데이터 수집 시작...\n")
        # ----------------------
        # 분봉 수집: 최근 1일치 분봉(예시)
        # ----------------------
        print("✅ 토큰 준비 완료, 분봉 데이터 수집 시작...\n")

        for ticker in TICKERS:
            print(f"📡 [{ticker}] 분봉 수집 중...", end=" ")

            # 최근 N일을 분봉으로 가져오려면 시작/종료 시각을 설정
            days = 1
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)

            # 문자열 포맷 YYYYMMDDHHMM 사용
            start_s = start_dt.strftime('%Y%m%d%H%M')
            end_s = end_dt.strftime('%Y%m%d%H%M')

            records = get_overseas_ohlcv_minute(
                token,
                ticker,
                start_dt=start_s,
                end_dt=end_s,
                interval_minutes=1,
                adj_price=True,
            )

            if not records:
                print("⚠️ 데이터 없음 (엔드포인트/파라미터 확인 필요)")
                time.sleep(0.5)
                continue

            df = pd.DataFrame(records)

            # 분봉 응답의 시간 필드명은 API마다 다릅니다. 가능한 키 후보를 확인
            time_keys = ['xymdhm', 'time', 'tr_time', 'timestamp']
            datetime_col = None
            for k in time_keys:
                if k in df.columns:
                    datetime_col = k
                    break

            if not datetime_col:
                # 가능한 숫자 필드 중 길이로 추정
                for c in df.columns:
                    sample = str(df[c].iat[0]) if len(df) > 0 else ''
                    if len(sample) >= 12 and sample.isdigit():
                        datetime_col = c
                        break

            if not datetime_col:
                print(f"\n⚠️ [{ticker}] 시간 필드 찾기 실패 -> 컬럼: {list(df.columns)}")
                time.sleep(0.5)
                continue

            # 표준 컬럼 매핑
            # 가능한 가격/거래량 필드명 후보
            col_map = {
                'open': None, 'high': None, 'low': None, 'clos': None, 'tvol': None,
                'openp': None, 'highp': None, 'lowp': None, 'close': None, 'volume': None
            }
            for c in df.columns:
                lc = c.lower()
                if 'open' in lc and col_map['open'] is None:
                    col_map['open'] = c
                if 'high' in lc and col_map['high'] is None:
                    col_map['high'] = c
                if 'low' in lc and col_map['low'] is None:
                    col_map['low'] = c
                if ('clos' in lc or 'close' in lc) and col_map['clos'] is None:
                    col_map['clos'] = c
                if 'vol' in lc or 'tvol' in lc:
                    if col_map['tvol'] is None:
                        col_map['tvol'] = c

            # 필요한 값이 없는 경우 경고
            if not col_map['clos']:
                print(f"\n⚠️ [{ticker}] 종가 컬럼 없음 -> 컬럼: {list(df.columns)}")
                time.sleep(0.5)
                continue

            # DataFrame 구성
            df2 = pd.DataFrame()
            df2['DateTime'] = df[datetime_col]

            def parse_dt(x):
                try:
                    # YYYYMMDDHHMM 숫자 형태
                    if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit() and len(str(x)) >= 12):
                        return pd.to_datetime(str(x), format='%Y%m%d%H%M')
                    return pd.to_datetime(x)
                except Exception:
                    return pd.NaT

            df2['DateTime'] = df2['DateTime'].apply(parse_dt)

            df2['Open'] = df[col_map['open']] if col_map['open'] in df.columns else None
            df2['High'] = df[col_map['high']] if col_map['high'] in df.columns else None
            df2['Low'] = df[col_map['low']] if col_map['low'] in df.columns else None
            # 종가은 clos 또는 close
            close_col = col_map['clos'] or col_map['close'] if 'close' in col_map else col_map['clos']
            df2['Close'] = df[close_col] if close_col in df.columns else None
            vol_col = col_map['tvol'] if col_map['tvol'] in df.columns else ('volume' if 'volume' in df.columns else None)
            df2['Volume'] = df[vol_col] if vol_col and vol_col in df.columns else None

            # CSV 저장
            s = start_dt.strftime('%Y%m%d')
            e = end_dt.strftime('%Y%m%d')
            filename = f"{save_dir}/{ticker}_minute_{s}_{e}.csv"
            df2.to_csv(filename, index=False)
            print(f"✅ 저장 완료 ({len(df2)}개 분봉 레코드)")

            time.sleep(0.5)

        print("\n🚀 모든 작업 완료")

    except Exception as e:
        print(f"\n❌ 전체 실행 중 오류 발생: {e}")
