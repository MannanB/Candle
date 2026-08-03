@echo off
setlocal

REM Move from scripts\ to the project root.
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install cmake ninja pybind11 numpy
if errorlevel 1 exit /b 1

echo.
echo Environment setup complete.
endlocal
