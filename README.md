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
