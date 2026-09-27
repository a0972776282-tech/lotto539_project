@echo off
chcp 65001 >nul
echo ==========================================
echo   啟動今彩539本地端網頁...
echo   (如果要關閉網頁伺服器，請直接關閉此黑色視窗)
echo ==========================================
call venv\Scripts\activate
streamlit run app.py
pause