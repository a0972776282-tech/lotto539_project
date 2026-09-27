import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- 1. 頁面與快取設定 ---
st.set_page_config(page_title="今彩539 AI預測系統", layout="centered")
st.title("🔮 今彩539 機器學習預測系統")

# 使用快取加快網頁讀取速度
@st.cache_data 
def load_data():
    return pd.read_csv('processed_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('lotto_model.pkl')

df = load_data()
model = load_model()

# --- 2. 顯示歷史資料庫 ---
st.subheader("📊 目前資料庫 (最後 10 期)")
st.dataframe(df.tail(10))

# --- 3. 預測區塊 ---
st.subheader("🎯 下一期號碼預測")
if st.button("啟動 AI 預測"):
    # 擷取最新一期的特徵來預測下一期
    feature_cols = ['獎號1', '獎號2', '獎號3', '獎號4', '獎號5', '和值', '均值', '首尾差']
    latest_data = df.loc[len(df)-1, feature_cols].values.reshape(1, -1)
    
    # 進行預測並四捨五入
    predicted_balls = model.predict(latest_data)[0]
    predicted_balls = np.round(predicted_balls).astype(int)
    
    # 確保預測數字在合理範圍 (1~39)
    predicted_balls = np.clip(predicted_balls, 1, 39)
    
    st.success(f"**模型預測號碼為: {predicted_balls[0]}, {predicted_balls[1]}, {predicted_balls[2]}, {predicted_balls[3]}, {predicted_balls[4]}**")
    st.info("💡 提示：機器學習主要捕捉統計分佈，彩券本質為隨機事件，預測結果僅供技術實作參考。")

st.divider() # 分隔線

# --- 4. 校正回歸 (輸入新開獎號碼) ---
st.subheader("📝 校正回歸：輸入今日最新開獎號碼")
col1, col2 = st.columns(2)
with col1:
    new_issue = st.text_input("期別 (例如: 115235)")
    new_date = st.date_input("開獎日期", datetime.date.today())
with col2:
    new_balls_str = st.text_input("輸入5個號碼 (用逗號隔開，例: 5,12,17,28,35)")

if st.button("儲存並更新資料庫"):
    if new_issue and new_balls_str:
        try:
            # 將字串轉換為整數陣列並排序
            new_balls = [int(x.strip()) for x in new_balls_str.split(',')]
            new_balls.sort()
            
            if len(new_balls) != 5:
                st.error("請確認輸入了剛好 5 個號碼！")
            else:
                # 自動計算統計特徵
                sum_val = sum(new_balls)
                mean_val = int(round(sum_val / 5))
                odd_count = sum(1 for x in new_balls if x % 2 != 0)
                even_count = 5 - odd_count
                odd_even_str = f"{odd_count}:{even_count}"
                diff_val = new_balls[4] - new_balls[0]
                
                # 建立新一期的資料 DataFrame
                new_row = pd.DataFrame([{
                    '期別': int(new_issue),
                    '開獎日期': new_date.strftime("%Y/%m/%d"),
                    '獎號1': new_balls[0], '獎號2': new_balls[1], '獎號3': new_balls[2], 
                    '獎號4': new_balls[3], '獎號5': new_balls[4],
                    '和值': sum_val, '均值': mean_val, 
                    '單雙': odd_even_str, '首尾差': diff_val
                }])
                
                # 將新資料合併到原本的資料庫並存檔
                updated_df = pd.concat([df, new_row], ignore_index=True)
                updated_df.to_csv('processed_data.csv', index=False)
                
                # 清除網頁快取以顯示最新資料
                st.cache_data.clear()
                st.success("✅ 資料庫已成功更新！若要讓 AI 學習新資料，請關閉網頁並在終端機重新執行 `python train_model.py`。")
        except Exception as e:
            st.error(f"輸入格式有誤，請檢查。錯誤訊息: {e}")
    else:
        st.warning("請填寫完整的期別與號碼。")