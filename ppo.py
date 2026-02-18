import pandas as pd
import numpy as np
import torch
import joblib
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ==========================================
# [1] 설정 영역
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PPO_MODEL_DIR = os.path.join(BASE_DIR, "ppo_models")

if not os.path.exists(PPO_MODEL_DIR):
    os.makedirs(PPO_MODEL_DIR)

TICKERS = ['AAPL', 'AMD', 'AMZN', 'GOOGL', 'META', 'NVDA', 'PLTR', 'TSLA']
SEQ_LENGTH = 60
INITIAL_BALANCE = 10000 
TRANSACTION_FEE = 0  # 수수료

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature 리스트 (code/3/lstm.py 와 동일해야 함)
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI', 'MACD', 'MACD_Signal', 'MA20',
    'ATR', 'VWAP', 'VIX', 'TNX', 'DXY', 'QQQ', 'XLK',
    'DayOfWeek', 'Hour'
]

# ==========================================
# [2] LSTM 클래스 (구조 동일)
# ==========================================
class StockLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# [3] 주식 거래 환경 (Gym)
# ==========================================
class StockTradingEnv(gym.Env):
    def __init__(self, df, lstm_model, scaler_X, scaler_y, initial_balance=10000, transaction_fee=0.0005):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.lstm_model = lstm_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Action: 0=매도, 1=보유, 2=매수
        self.action_space = spaces.Discrete(3)
        
        # Observation Space 정의
        # 1. LSTM예측수익률, 2. 현재수익률(전봉대비), 3. VWAP괴리율, 4. RSI/100, 
        # 5. MACD, 6. ATR/Close(변동성비율), 7. 심리지수, 8. 공포탐욕/100, 9. 보유비율, 10. 현금비율
        self.obs_dim = 10 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Feature 컬럼 인덱싱 준비 (LSTM 학습 시 사용한 스케일러 정보 우선)
        if hasattr(self.scaler_X, "feature_names_in_"):
            self.feature_cols = list(self.scaler_X.feature_names_in_)
        else:
            self.feature_cols = [f for f in FEATURES if f in self.df.columns]
        
        self.current_step = SEQ_LENGTH
        self.balance = initial_balance
        self.shares_held = 0
        self.total_assets = initial_balance
        self.max_assets = initial_balance
        self.trades = []
        self.asset_history = [initial_balance]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = SEQ_LENGTH
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_assets = self.initial_balance
        self.max_assets = self.initial_balance
        self.trades = []
        self.asset_history = [self.initial_balance]
        return self._get_observation(), {}

    def _get_lstm_prediction(self):
        """LSTM을 이용해 '다음 타임스텝의 예상 수익률' 예측"""
        if self.current_step < SEQ_LENGTH:
            return 0.0
        
        # LSTM 입력 데이터 추출 (SEQ_LENGTH 만큼)
        sequence = self.df[self.feature_cols].iloc[self.current_step - SEQ_LENGTH:self.current_step].values
        # 스케일링
        sequence_scaled = self.scaler_X.transform(sequence)
        
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            # 예측된 스케일된 수익률
            pred_scaled = self.lstm_model(seq_tensor).cpu().numpy()[0, 0]
        
        # 원래 수익률 스케일로 복원
        pred_return = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        return pred_return

    def _get_observation(self):
        # 현재 데이터 가져오기
        row = self.df.iloc[self.current_step]
        prev_close = self.df.iloc[self.current_step - 1]['Close']
        
        # 1. LSTM 예측 (예상 수익률)
        predicted_return = self._get_lstm_prediction()
        
        # 2. 현재 변동률 (전봉 대비)
        current_return = (row['Close'] - prev_close) / prev_close
        
        # 3. VWAP 괴리율 (현재가가 VWAP보다 얼마나 높냐/낮냐)
        vwap_diff = (row['Close'] - row['VWAP']) / row['VWAP'] if 'VWAP' in row else 0
        
        # 4. 기타 지표 정규화
        rsi_norm = row['RSI'] / 100.0 if 'RSI' in row else 0.5
        macd_val = row['MACD'] if 'MACD' in row else 0
        atr_ratio = (row['ATR'] / row['Close']) if 'ATR' in row else 0 # 가격 대비 변동성
        sentiment = row['News_Sentiment'] if 'News_Sentiment' in row else 0
        fear_greed = row['Fear_Greed_Index'] / 100.0 if 'Fear_Greed_Index' in row else 0.5
        
        # 5. 포트폴리오 상태 (정규화)
        total_val = self.balance + self.shares_held * row['Close']
        shares_ratio = (self.shares_held * row['Close']) / total_val # 자산 중 주식 비중 (0~1)
        cash_ratio = self.balance / total_val # 자산 중 현금 비중 (0~1)

        obs = np.array([
            predicted_return,
            current_return,
            vwap_diff,
            rsi_norm,
            macd_val,
            atr_ratio,
            sentiment,
            fear_greed,
            shares_ratio,
            cash_ratio
        ], dtype=np.float32)
        
        # NaN 방지
        return np.nan_to_num(obs)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        current_price = row['Close']
        prev_assets = self.total_assets

        # 기본적으로 현재 가격 기준 자산 재계산
        self.total_assets = self.balance + self.shares_held * current_price

        # 행동 수행
        if action == 0:  # 매도
            if self.shares_held > 0:
                assets_before = prev_assets
                shares_to_sell = self.shares_held
                sell_amount = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.balance += sell_amount
                self.shares_held = 0
                self.total_assets = self.balance  # 전량 매도 후 자산은 현금만

                profit = self.total_assets - assets_before
                profit_rate = (profit / assets_before) * 100 if assets_before != 0 else 0.0

                self.trades.append({
                    'step': self.current_step,
                    'timestamp': row['timestamp'] if 'timestamp' in self.df.columns else None,
                    'action': 'SELL',
                    'price': float(current_price),
                    'shares_traded': int(shares_to_sell),
                    'assets_before': float(assets_before),
                    'assets_after': float(self.total_assets),
                    'profit': float(profit),
                    'profit_rate': float(profit_rate)
                })
        elif action == 2:  # 매수
            if self.balance > current_price:
                assets_before = prev_assets
                # 수수료가 0이면 단순 계산
                if self.transaction_fee == 0:
                    max_shares = int(self.balance / current_price)
                    cost = max_shares * current_price
                else:
                    max_shares = int(self.balance / (current_price * (1 + self.transaction_fee)))
                    cost = max_shares * current_price * (1 + self.transaction_fee)
                
                if max_shares > 0:
                    self.balance -= cost
                    self.shares_held += max_shares
                    self.total_assets = self.balance + self.shares_held * current_price

                    profit = self.total_assets - assets_before
                    profit_rate = (profit / assets_before) * 100 if assets_before != 0 else 0.0

                    self.trades.append({
                        'step': self.current_step,
                        'timestamp': row['timestamp'] if 'timestamp' in self.df.columns else None,
                        'action': 'BUY',
                        'price': float(current_price),
                        'shares_traded': int(max_shares),
                        'assets_before': float(assets_before),
                        'assets_after': float(self.total_assets),
                        'profit': float(profit),
                        'profit_rate': float(profit_rate)
                    })
        else:
            # HOLD인 경우에도 현재 가격 기준 자산만 갱신
            self.total_assets = self.balance + self.shares_held * current_price

        # 보상 계산: (현재 자산 - 이전 자산) / 이전 자산 * 100 (퍼센트 단위 보상)
        reward = ((self.total_assets - prev_assets) / prev_assets) * 100

        # 자산 히스토리 저장 (성능 지표 계산용)
        self.asset_history.append(self.total_assets)
        
        # 거래 활성화: HOLD 페널티 및 거래 보너스
        if action == 1:  # HOLD
            reward -= 0.02  # HOLD 페널티 강화
        elif action == 0:  # SELL
            reward += 0.05  # 매도 시 보너스 (손익실현 장려)
        elif action == 2:  # BUY
            reward += 0.03  # 매수 시 보너스

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # 에피소드 종료 시 보유 주식 강제 매도 (손익 실현)
        if done and self.shares_held > 0:
            final_price = self.df.iloc[self.current_step - 1]['Close']
            assets_before_final = self.total_assets
            sell_amount = self.shares_held * final_price * (1 - self.transaction_fee)
            self.balance += sell_amount
            self.shares_held = 0
            self.total_assets = self.balance
            
            profit_final = self.total_assets - assets_before_final
            profit_rate_final = (profit_final / assets_before_final) * 100 if assets_before_final != 0 else 0.0
            
            self.trades.append({
                'step': self.current_step - 1,
                'timestamp': self.df.iloc[self.current_step - 1]['timestamp'] if 'timestamp' in self.df.columns else None,
                'action': 'SELL',
                'price': float(final_price),
                'shares_traded': int(self.shares_held) if self.shares_held > 0 else 0,
                'assets_before': float(assets_before_final),
                'assets_after': float(self.total_assets),
                'profit': float(profit_final),
                'profit_rate': float(profit_rate_final)
            })
        
        return self._get_observation(), reward, done, truncated, {'total_assets': self.total_assets}

# ==========================================
# [4] PPO 학습 함수
# ==========================================
def train_ppo(ticker):
    print(f"\n🤖 [{ticker}] PPO 강화학습 시작...")
    
    # 데이터 로드
    file_path = os.path.join(DATA_DIR, f"{ticker}_hourly_dataset.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, f"{ticker}_hourly_alp_yf_dataset_v2.csv") # 파일명 주의
		
    df = pd.read_csv(file_path)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # LSTM에서 사용했던 스케일러 로드 (feature 개수/순서 동기화용)
    scaler_X = joblib.load(f"{MODEL_DIR}/{ticker}_scaler_X.pkl")
    scaler_y = joblib.load(f"{MODEL_DIR}/{ticker}_scaler_y.pkl")

    # LSTM 모델 로드: 스케일러 기준 입력 차원 사용
    if hasattr(scaler_X, "feature_names_in_"):
        feature_cols = list(scaler_X.feature_names_in_)
    else:
        feature_cols = [f for f in FEATURES if f in df.columns]

    lstm_model = StockLSTM(input_size=scaler_X.n_features_in_)
    lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/{ticker}_lstm.pth", map_location=device))
    lstm_model.to(device)
    lstm_model.eval()

    # 학습/검증 분리
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # 학습/검증 데이터 크기 출력
    print(f"📊 학습 데이터: {len(train_df)}행, 검증 데이터: {len(val_df)}행")
    
    # 환경 생성
    train_env = DummyVecEnv([lambda: StockTradingEnv(train_df, lstm_model, scaler_X, scaler_y, INITIAL_BALANCE, TRANSACTION_FEE)])
    val_env = StockTradingEnv(val_df, lstm_model, scaler_X, scaler_y, INITIAL_BALANCE, TRANSACTION_FEE)
    
    # 모델 정의 및 학습
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, batch_size=64, n_steps=2048)
    model.learn(total_timesteps=30000) # 학습 횟수 조절 가능

    # 모델 저장
    model.save(f"{PPO_MODEL_DIR}/{ticker}_ppo")
    print(f"✅ PPO 모델 저장 완료: {ticker}")

    # 검증
    print(f"\n🔍 [{ticker}] 검증 시작...")
    obs, _ = val_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = val_env.step(action)
        total_reward += reward
        if truncated:
            break

    final_assets = info['total_assets']
    profit = final_assets - INITIAL_BALANCE
    profit_rate = (profit / INITIAL_BALANCE) * 100

    # ===== 성능 지표 계산 =====
    # 자산 히스토리 기반 샤프비율 / 최대 낙폭
    sharpe_ratio = None
    max_drawdown = None
    if hasattr(val_env, "asset_history") and len(val_env.asset_history) > 1:
        equity = np.array(val_env.asset_history, dtype=float)
        returns = np.diff(equity) / equity[:-1]
        if np.std(returns) > 0:
            sharpe_ratio = float(np.mean(returns) / np.std(returns))
        else:
            sharpe_ratio = 0.0

        running_max = np.maximum.accumulate(equity)
        drawdowns = equity / running_max - 1.0
        max_drawdown = float(drawdowns.min())  # 음수값 (예: -0.25 = -25%)

    # 거래 로그 기반 승률 (매수->매도 쌍으로 계산)
    win_rate = None
    if len(val_env.trades) > 0:
        trades_df = pd.DataFrame(val_env.trades)
        # SELL 기준 승률 계산
        sell_trades = trades_df[trades_df["action"] == "SELL"]
        if len(sell_trades) > 0:
            # 각 SELL 거래의 profit_rate로 승패 판단
            wins = (sell_trades["profit_rate"] > 0).sum()
            win_rate = float(wins / len(sell_trades) * 100.0)
        else:
            win_rate = 0.0  # SELL이 없으면 승률 0%
    else:
        win_rate = 0.0  # 거래가 없으면 승률 0%

    print(f"\n{'='*50}")
    print(f"[{ticker}] 검증 결과")
    print(f"{'='*50}")
    print(f"초기 자본: ${INITIAL_BALANCE:,.2f}")
    print(f"최종 자산: ${final_assets:,.2f}")
    print(f"수익: ${profit:,.2f} ({profit_rate:.2f}%)")
    print(f"총 보상: {total_reward:.4f}")
    print(f"거래 횟수: {len(val_env.trades)}")
    if sharpe_ratio is not None:
        print(f"샤프비율: {sharpe_ratio:.4f}")
    if max_drawdown is not None:
        print(f"최대 낙폭(MDD): {max_drawdown*100:.2f}%")
    if win_rate is not None:
        print(f"승률(SELL 기준): {win_rate:.2f}%")
    print(f"{'='*50}\n")

    # 검증 매매 로그를 CSV로 저장
    if len(val_env.trades) > 0:
        trades_df = pd.DataFrame(val_env.trades)
        trades_path = os.path.join(PPO_MODEL_DIR, f"{ticker}_ppo_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"\n💾 {ticker} 검증 매매 로그 CSV 저장 완료: {trades_path}")

    return {
        'ticker': ticker,
        'initial': INITIAL_BALANCE,
        'final': final_assets,
        'profit': profit,
        'profit_rate': profit_rate,
        'total_reward': total_reward,
        'trades': len(val_env.trades),
        'sharpe_ratio': sharpe_ratio if sharpe_ratio is not None else 0,
        'max_drawdown': max_drawdown if max_drawdown is not None else 0,
        'win_rate': win_rate  # 이미 0.0으로 초기화되어 있음
    }

if __name__ == "__main__":
    print("="*60)
    print("PPO 강화학습 기반 주식 트레이딩")
    print("="*60)

    results = []

    for ticker in TICKERS:
        try:
            result = train_ppo(ticker)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # 전체 결과 요약
    if results:
        print("\n" + "="*100)
        print("📊 전체 종목별 성능지표 비교")
        print("="*100)
        
        # 데이터프레임으로 변환하여 테이블 형식으로 출력
        results_df = pd.DataFrame(results)
        
        # 컬럼 포맷팅
        summary_table = pd.DataFrame({
            '종목': results_df['ticker'],
            '초기자본': ['$' + f"{r:,.0f}" for r in results_df['initial']],
            '최종자산': ['$' + f"{r:,.2f}" for r in results_df['final']],
            '수익': ['$' + f"{r:,.2f}" for r in results_df['profit']],
            '수익률': [f"{r:.2f}%" for r in results_df['profit_rate']],
            '거래횟수': results_df['trades'].astype(int),
            '샤프비율': [f"{r:.4f}" for r in results_df['sharpe_ratio']],
            'MDD': [f"{r*100:.2f}%" for r in results_df['max_drawdown']],
            '승률': [f"{r:.2f}%" for r in results_df['win_rate']]
        })
        
        print(summary_table.to_string(index=False))
        
        print("\n" + "="*100)
        print("📈 최종 요약 통계")
        print("="*100)
        print(f"총 종목 수: {len(results)}")
        print(f"평균 수익률: {results_df['profit_rate'].mean():.2f}%")
        print(f"최고 수익률: {results_df['profit_rate'].max():.2f}% ({results_df.loc[results_df['profit_rate'].idxmax(), 'ticker']})")
        print(f"최저 수익률: {results_df['profit_rate'].min():.2f}% ({results_df.loc[results_df['profit_rate'].idxmin(), 'ticker']})")
        print(f"평균 거래횟수: {results_df['trades'].mean():.1f}회")
        print(f"평균 샤프비율: {results_df['sharpe_ratio'].mean():.4f}")
        print(f"평균 MDD: {(results_df['max_drawdown'].mean()*100):.2f}%")
        print(f"평균 승률: {results_df['win_rate'].mean():.2f}%")
        print("="*100)

        # 검증 결과 CSV 저장
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(PPO_MODEL_DIR, "ppo_train_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 검증 결과 CSV 저장 완료: {summary_path}")