@echo off
setlocal

REM Move from scripts\ to the project root.
cd /d "%~dp0.."

if exist build (
    rmdir /s /q build
    echo Removed build directory.
) else (
    echo Nothing to clean.
)

endlocal
