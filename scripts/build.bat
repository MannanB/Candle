@echo off
setlocal

REM Move from scripts\ to the project root.
cd /d "%~dp0.."

set "VSROOT=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

if not exist "build\build.ninja" (
    echo Build directory is not configured. Running configure.bat...
    call "%~dp0configure.bat"
    if errorlevel 1 exit /b 1
)

cmake --build build
if errorlevel 1 exit /b 1

echo.
echo Build complete.
endlocal
