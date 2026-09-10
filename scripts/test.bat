@echo off
setlocal

REM Move from scripts\ to the project root.
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo Missing virtual environment at:
    echo %CD%\.venv
    exit /b 1
)

call ".venv\Scripts\activate.bat"

REM Allow Python to import the wrapper and generated .pyd module.
set "PYTHONPATH=%CD%\python;%CD%\build;%CD%\build\Release;%PYTHONPATH%"

python python\test_candle.py
set "RESULT=%ERRORLEVEL%"

endlocal & exit /b %RESULT%
