@echo off
setlocal

REM Move from scripts\ to the project root.
cd /d "%~dp0.."

REM Initialize MSVC and Windows SDK.
set "VSROOT=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo Failed to initialize Visual Studio Build Tools.
    exit /b 1
)

REM Activate the project virtual environment.
if not exist ".venv\Scripts\activate.bat" (
    echo Missing virtual environment at:
    echo %CD%\.venv
    echo.
    echo Create it with:
    echo python -m venv .venv
    exit /b 1
)

call ".venv\Scripts\activate.bat"

REM Find pybind11's CMake package.
for /f "usebackq delims=" %%I in (`python -m pybind11 --cmakedir`) do (
    set "PYBIND11_DIR=%%I"
)

if not defined PYBIND11_DIR (
    echo Could not find pybind11.
    echo Install it with:
    echo python -m pip install pybind11
    exit /b 1
)

cmake -S . -B build -G Ninja -Dpybind11_DIR="%PYBIND11_DIR%"
if errorlevel 1 exit /b 1

echo.
echo Configuration complete.
endlocal
