@echo off
:: ============================================================
::  EuroSAT High-Accuracy Pipeline
::  Hardware: Intel i7-9850H / 16 GB RAM / Quadro T1000
::  Run this from Command Prompt (not PowerShell)
:: ============================================================

echo.
echo ============================================================
echo   EuroSAT High-Accuracy Pipeline -- Local Run
echo   i7-9850H / 16 GB RAM / Quadro T1000 (CPU mode)
echo ============================================================
echo.

:: Activate the virtual environment
if not exist eurosat_env\Scripts\activate.bat (
    echo ERROR: Virtual environment not found.
    echo Please run this first:
    echo   cd /d c:\Users\pc\OneDrive\Desktop\DEPI\LTC
    echo   C:\Program Files\Python313\python.exe -m venv eurosat_env
    echo   eurosat_env\Scripts\pip install tensorflow numpy pandas matplotlib seaborn scikit-learn Pillow opencv-python rasterio
    pause
    exit /b 1
)

echo Activating virtual environment...
call eurosat_env\Scripts\activate.bat

echo.
echo Verifying packages...
python setup_verify.py
if errorlevel 1 (
    echo ERROR: Package check failed. Run pip install again.
    pause
    exit /b 1
)

echo.
echo Starting pipeline...
echo Results will be saved to: outputs\
echo Press Ctrl+C to stop at any time.
echo.
python eurosat_combined_pipeline.py

echo.
echo Pipeline finished. Check outputs\ for models and charts.
pause
