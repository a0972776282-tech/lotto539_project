@echo off
chcp 65001 >nul
echo ==========================================
echo   🔮 今彩539 AI 預測系統 - 自動更新腳本
echo ==========================================
echo.

echo [1/3] 正在啟動虛擬環境並重新訓練 AI 模型...
call venv\Scripts\activate
python train_model.py
echo.

echo [2/3] 正在將新資料與新模型打包...
git add .
git commit -m "Auto-update: 系統自動更新與備份"
echo.

echo [3/3] 正在推送到 GitHub (將觸發雲端網頁自動更新)...
git push
echo.

echo ==========================================
echo  ✅ 全部完成！雲端網頁將在 1~2 分鐘內自動更新。
echo ==========================================
pause