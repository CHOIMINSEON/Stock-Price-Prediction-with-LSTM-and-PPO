import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error # 추가된 라이브러리
import matplotlib.pyplot as plt # 시각화 라이브러리
import joblib
import os

# ==========================================
# [1] 설정 영역
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results") # 결과 이미지 저장 폴더

for path in [MODEL_DIR, RESULT_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# 사용할 종목 리스트
TICKERS = ['AAPL', 'AMD', 'AMZN', 'GOOGL', 'META', 'NVDA', 'PLTR', 'TSLA']

# 모델 하이퍼파라미터
SEQ_LENGTH = 60       
HIDDEN_SIZE = 64      
NUM_LAYERS = 2        
EPOCHS = 50           
BATCH_SIZE = 32       
LEARNING_RATE = 0.001 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 학습 장치: {device}")

# 학습에 사용할 변수들
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'RSI', 'MACD', 'MACD_Signal', 'MA20', 
    'ATR', 'VWAP', 'VIX', 'TNX', 'DXY', 'QQQ', 'XLK',
    'DayOfWeek', 'Hour'
]

# ==========================================
# [2] LSTM 모델 정의
# ==========================================
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

# ==========================================
# [3] 데이터 전처리 함수
# ==========================================
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length][3] # Index 3 is 'Close' (Target)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# [4] 학습 및 평가 메인 로직
# ==========================================
def train_lstm(ticker):
    print(f"\n[{ticker}] 데이터 로딩 중...")
    file_path = os.path.join(DATA_DIR, f"{ticker}_hourly_alp_yf_dataset_v2.csv")
    
    if not os.path.exists(file_path):
        print(f"파일 없음: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # 시간순 정렬
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    # 필요한 컬럼만 선택
    available_features = [f for f in FEATURES if f in df.columns]
    data = df[available_features].values
    
    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler() # 종가(Close)만 따로 스케일링 (역변환 위해)

    scaled_data = scaler_X.fit_transform(data)
    
    # Target(Close) Scaling 별도 저장
    close_idx = available_features.index('Close')
    scaler_y.fit(df[['Close']]) 

    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # Train/Val Split (8:2)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # Tensor 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 모델 초기화
    model = StockLSTM(input_size=len(available_features), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 루프
    best_loss = float('inf')
    patience = 0
    
    print(f"[{ticker}] 학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        avg_train_loss = train_loss / len(train_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")

        # Early Stopping check
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience = 0
            torch.save(model.state_dict(), f"{MODEL_DIR}/{ticker}_lstm.pth")
        else:
            patience += 1
            if patience > 5:
                print("Early Stopping triggered.")
                break
            
    # 스케일러 저장
    joblib.dump(scaler_X, f"{MODEL_DIR}/{ticker}_scaler_X.pkl")
    joblib.dump(scaler_y, f"{MODEL_DIR}/{ticker}_scaler_y.pkl") # y 스케일러도 저장

    # ==========================================
    # [5] (New) 상세 평가 및 시각화
    # ==========================================
    evaluate_model(model, X_val_tensor, y_val_tensor, scaler_y, ticker)

def evaluate_model(model, X_val, y_val, scaler_y, ticker):
    model.eval()
    with torch.no_grad():
        predictions = model(X_val).cpu().numpy()
        actuals = y_val.cpu().numpy()

    # 1. 역변환 (0~1 -> 실제 달러 가격)
    pred_price = scaler_y.inverse_transform(predictions)
    actual_price = scaler_y.inverse_transform(actuals)

    # 2. 오차 계산 (RMSE, MAE)
    mse = mean_squared_error(actual_price, pred_price)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_price, pred_price)

    # 3. 방향 정확도 (Directional Accuracy)
    # t시점의 가격이 t-1시점보다 올랐는지 내렸는지 부호가 같으면 정답
    # 실제 변동폭: Actual[t] - Actual[t-1]
    # 예측 변동폭: Pred[t] - Actual[t-1] (주의: 예측값과 '이전 실제값'을 비교해야 함)
    
    # 데이터가 시계열이므로 i번째 데이터의 '전날 가격'은 i-1번째 실제 가격임.
    # 하지만 X_val 데이터셋 구성 상, y_val[i]는 i번째 시퀀스의 타겟임.
    # 단순화를 위해 전체 배열에서의 연속성을 가정하고 계산 (1번째 인덱스부터 비교)
    
    actual_diff = actual_price[1:] - actual_price[:-1]
    pred_diff = pred_price[1:] - actual_price[:-1] 
    
    # 부호가 같으면 True (오름/내림 맞춤)
    correct_direction = np.sign(actual_diff) == np.sign(pred_diff)
    accuracy = np.mean(correct_direction) * 100

    print(f"\n📊 [{ticker}] 최종 성능 평가")
    print(f" - RMSE (평균 오차): ${rmse:.4f}")
    print(f" - MAE  (절대 오차): ${mae:.4f}")
    print(f" - Directional Accuracy (방향 정확도): {accuracy:.2f}%")

    # 4. 시각화 (최근 100시간)
    plt.figure(figsize=(12, 6))
    
    # 전체 기간 중 마지막 100개만 시각화
    vis_len = 100
    plt.plot(actual_price[-vis_len:], label='Actual Price', color='blue', alpha=0.6)
    plt.plot(pred_price[-vis_len:], label='Predicted Price', color='red', linestyle='--')
    
    plt.title(f"{ticker} LSTM Prediction (Last {vis_len} Hours)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    
    save_path = f"{RESULT_DIR}/{ticker}_prediction.png"
    plt.savefig(save_path)
    plt.close()
    print(f" - 그래프 저장 완료: {save_path}\n")

if __name__ == "__main__":
    for ticker in TICKERS:
        train_lstm(ticker)