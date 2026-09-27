import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("1. 讀取處理好的資料...")
df = pd.read_csv('processed_data.csv')

# 確保資料按期別排序 (由舊到新，這樣「上一期」推「下一期」才有邏輯)
df = df.sort_values('期別').reset_index(drop=True)

# 準備我們的 X (特徵/題目) 和 Y (目標/答案)
# 我們挑選數值型態的欄位作為特徵 (略過單雙字串)
feature_cols = ['獎號1', '獎號2', '獎號3', '獎號4', '獎號5', '和值', '均值', '首尾差']

X = []
Y = []

print("2. 正在構建時間序列資料 (用上一期推算下一期)...")
# 迴圈跑到倒數第二筆，因為最後一筆沒有「下一期」可以當作答案
for i in range(len(df) - 1): 
    # X 是第 i 期 (今天) 的特徵
    current_features = df.loc[i, feature_cols].values
    X.append(current_features)
    
    # Y 是第 i+1 期 (下一期) 的 5 個開獎號碼
    next_balls = df.loc[i+1, ['獎號1', '獎號2', '獎號3', '獎號4', '獎號5']].values
    Y.append(next_balls)

X = np.array(X)
Y = np.array(Y)

print(f"總共準備了 {len(X)} 筆訓練資料。")

print("3. 正在訓練隨機森林模型 (Random Forest)...")
# 建立隨機森林迴歸模型 (它可以同時預測 5 個數值)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)
print("模型訓練完成！")

# 4. 儲存模型
model_filename = 'lotto_model.pkl'
joblib.dump(model, model_filename)
print(f"✅ 機器學習模型已儲存為 {model_filename}")

# --- 測試一下模型的預測能力 ---
print("\n--- 預測測試 ---")
# 拿出資料庫中「最新的一期」來預測「未來的下一期」
latest_data = df.loc[len(df)-1, feature_cols].values.reshape(1, -1)
predicted_balls = model.predict(latest_data)[0]

# 將預測出來的浮點數四捨五入成整數
predicted_balls = np.round(predicted_balls).astype(int)
print(f"根據最新一期資料，模型預測下一期的號碼為: {predicted_balls}")