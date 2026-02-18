# 주가 예측을 위한 시계열 모델 및 강화학습(2025)
* LSTM 가격 예측과 PPO 강화학습을 결합한 자동 트레이딩 프로젝트.
* 미국 주식의 방향을 예측하고 매수, 매도를 진행하는 에이전트를 개발하고자 프로젝트 진행.
* 주가에 영향을 줄 수 있는 날짜, 시간, 경제지표, 기술적지표 데이터를 활용해 시계열 모델(lstm)을 학습.
* 학습한 시계열 모델(lstm)을 바탕으로 매수,매도를 진행하는 강화학습(ppo)을 진행해 검증기간 수익률 확인.

## [주요 기능]

**1. 다양한 데이터 수집 파이프라인**
- **시계열 데이터**: yfinance, Alpaca API, 한국투자증권 API를 통한 일봉/시간봉 수집
- **뉴스 및 펀더멘털**: DuckDuckGo 뉴스, Finviz 증권사 의견, 내부자 거래 정보
- **거시 지표**: VIX, 미국 10년물(TNX), 달러지수(DXY), QQQ, XLK 등
- **기술적 지표**: RSI, MACD, ATR, VWAP, 이동평균 등 자동 계산

**2. LSTM 가격 예측 모델**
- PyTorch 기반 시계열 예측
- 60개 시퀀스 학습 → 다음 시점 예측
- 로그 수익률 예측으로 Data Leakage 방지
- MinMaxScaler 정규화 및 Train/Test Split

**3. PPO 강화학습 매매 에이전트**
- Stable-Baselines3 기반 PPO 알고리즘
- LSTM 예측값을 State로 활용
- 행동 공간: 매도(0), 보유(1), 매수(2)
- 포트폴리오 가치 최대화 목표

**4. AI 요약 에이전트**
- Qwen LLM을 활용한 종목 분석 자동화
- 뉴스, 펀더멘털, 투자의견을 JSON 형태로 요약
- Ollama 로컬 서버 호환

---

## Directory Structure
```text
Stock-Price-Prediction-with-LSTM-and-PPO/
│
├── agent/                      # 뉴스/펀더멘털 수집 에이전트
│ ├── agent_summary.py          # LLM 기반 종목 분석 요약 (Qwen)
│ ├── agent_data.py             # 기본 데이터 수집 테스트
│ └── agent_data_Selenium.py    # Selenium 기반 확장 크롤링
│
├── data/                       # 시계열 데이터 수집 스크립트
│ ├── yfi_day_data.py           # yfinance 일봉
│ ├── yfi_hour_data.py          # yfinance 시간봉
│ ├── alpaca_hour_data.py       # Alpaca API 시간봉
│ ├── alp_yf_hour.py            # Alpaca + yfinance 혼합 (ATR, VWAP 포함)
│ └── han_api_data_day.py       # 한국투자증권 API 일봉
│
├── lstm.py                     # LSTM 주가 예측 모델
├── ppo.py                      # PPO 강화학습 매매 에이전트
└── README.md

```
---
## Flowchart
```text
[1. 데이터 수집]
   ├─ data/yfi_day_data.py        -> 일봉 CSV 생성
   ├─ data/yfi_hour_data.py       -> 시간봉 CSV 생성
   └─ data/alp_yf_hour.py         -> 고급 시간봉 CSV 생성
              ↓
[2. 피처 엔지니어링]
   - RSI, MACD, ATR, VWAP 계산
   - 거시지표 병합 (VIX, TNX, DXY, QQQ, XLK)
   - 시간 특성 추가 (DayOfWeek, Hour)
              ↓
[3. LSTM 학습 및 예측]
   - lstm.py 실행
   - 과거 60개 시점 → 다음 1개 시점 예측
   - 모델 저장: models/{ticker}_lstm_model.pth
   - Scaler 저장: models/{ticker}_scaler_X.pkl, scaler_y.pkl
              ↓
[4. PPO 강화학습]
   - ppo.py 실행
   - LSTM 예측값을 State에 포함
   - 매매 행동 학습 (매도/보유/매수)
   - PPO 모델 저장: ppo_models/{ticker}_ppo_model.zip
              ↓
[5. 백테스팅 및 평가]
   - 검증 데이터로 시뮬레이션
   - 수익률, 샤프비율, MDD 계산
   - 자산 곡선 시각화

[뉴스 분석 (옵션)]
   - agent/agent_summary.py 실행
   - 실시간 뉴스 + 펀더멘털 수집
   - Qwen LLM으로 투자 인사이트 생성

```

---
**[성능 지표]**
**LSTM 평가**
- RMSE (Root Mean Squared Error): 평균 예측 오차
- MAE (Mean Absolute Error): 절대 오차
- Directional Accuracy: 방향 예측 정확도 (상승/하락)

*PPO 백테스팅**
- 수익률: (최종 자산 - 초기 자본) / 초기 자본 × 100
- 샤프 비율: 위험 대비 수익률 (Sharpe Ratio)
- MDD: 최대 낙폭 (Maximum Drawdown)
- 거래 횟수: 총 매수/매도 실행 횟수
- 승률: 수익 거래 비율
