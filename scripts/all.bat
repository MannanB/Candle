@echo off
setlocal

call "%~dp0setup.bat"
if errorlevel 1 exit /b 1

call "%~dp0rebuild.bat"
if errorlevel 1 exit /b 1

call "%~dp0test.bat"
if errorlevel 1 exit /b 1

echo.
echo Setup, build, and test all succeeded.
endlocal
