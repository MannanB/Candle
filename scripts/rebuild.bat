@echo off
setlocal

call "%~dp0clean.bat"
if errorlevel 1 exit /b 1

call "%~dp0configure.bat"
if errorlevel 1 exit /b 1

call "%~dp0build.bat"
if errorlevel 1 exit /b 1

echo.
echo Full rebuild complete.
endlocal
