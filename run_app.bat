@echo off
cd /d "%~dp0"
echo ===================================================
echo  Starting EduPredict Streamlit App...
echo ===================================================

python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Python command me error aaya. 'py' command se try kar rahe hain...
    py -m streamlit run app.py
)

if errorlevel 1 (
    echo.
    echo ===================================================
    echo Error: Streamlit run nahi ho saka!
    echo Agar packages install nahi hain to ye command chalayein:
    echo   pip install -r requirement.txt
    echo ===================================================
    pause
)
