import pandas as pd
import numpy as np

# 1. 讀取 CSV 檔案
print("正在讀取資料...")
df = pd.read_csv('data.csv')

# 確認讀取到的前幾筆資料
print("\n--- 原始資料 ---")
print(df.head())

# 2. 擷取五個球號的欄位 (假設欄位名稱為 球號1 ~ 球號5)
ball_cols = ['獎號1', '獎號2', '獎號3', '獎號4', '獎號5']
balls_df = df[ball_cols]

# 3. 計算統計指標 (特徵工程 Feature Engineering)
print("\n正在計算統計指標 (和值、均值、單雙、首尾差)...")

# (A) 和值：5個號碼相加
df['和值'] = balls_df.sum(axis=1)

# (B) 均值：和值除以 5，並四捨五入到整數
df['均值'] = np.round(df['和值'] / 5).astype(int)

# (C) 單雙比：計算 5 個號碼中有幾個奇數(單)，幾個偶數(雙)
# 用餘數來判斷，除以 2 餘 1 為單數
odd_counts = (balls_df % 2 != 0).sum(axis=1)
even_counts = 5 - odd_counts
df['單雙'] = odd_counts.astype(str) + ':' + even_counts.astype(str)

# (D) 首尾差：最大的號碼(球號5) 減去 最小的號碼(球號1)
# 539開獎號碼通常已經由小排到大
df['首尾差'] = df['獎號5'] - df['獎號1']

# 4. 顯示計算結果
print("\n--- 加入統計指標後的資料 ---")
# 只顯示我們新算出來的欄位看看
print(df[['期別', '開獎日期', '和值', '均值', '單雙', '首尾差']].head())

# 5. 將處理好的資料另存為新的 CSV，供後續訓練模型使用
df.to_csv('processed_data.csv', index=False)
print("\n✅ 資料處理完成！已儲存為 processed_data.csv")