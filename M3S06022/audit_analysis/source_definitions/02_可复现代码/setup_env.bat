@echo off
chcp 65001 >nul 2>&1
echo ================================================
echo  SIC Trajectory Study — Environment Setup
echo ================================================
echo.

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.11+ and add to PATH.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo.

:: Create virtual environment
if not exist ".venv" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        Done: .venv created
) else (
    echo [1/4] Virtual environment already exists, skipping.
)

:: Activate
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install dependencies
echo [3/4] Installing Python packages...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo        Done: all packages installed

:: Verify
echo [4/4] Verifying installation...
python -c "import pandas, numpy, scipy, sklearn, matplotlib, xgboost, shap, lifelines; print('All packages OK')"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Package verification failed.
    pause
    exit /b 1
)

:: Check data directory
echo.
if exist "..\03_数据源\MIMIC-IV" (
    echo [OK] Data directory found: ..\03_数据源\
) else (
    echo [WARNING] Data directory not found at ..\03_数据源\
    echo           Make sure 03_数据源 is in the same parent folder as 02_可复现代码
)

echo.
echo ================================================
echo  Setup complete!
echo.
echo  To run scripts:
echo    1. Activate:   .venv\Scripts\activate.bat
echo    2. Run:        python 03_轨迹聚类与预测模型\run_sic_trajectory_analysis.py --full --force-k 4
echo ================================================
pause
