@echo off
echo AutoDeployLR - Trying to run the application...
echo.

echo Attempting to run simple demo first...
echo.

REM Try different Python executables
if exist "C:\Python*\python.exe" (
    echo Found Python installation, running simple demo...
    C:\Python*\python.exe simple_demo.py
    goto :end
)

REM Try py launcher
py --version >nul 2>&1
if %errorlevel%==0 (
    echo Using py launcher...
    py simple_demo.py
    goto :end
)

REM Try python command
python --version >nul 2>&1
if %errorlevel%==0 (
    echo Using python command...
    python simple_demo.py
    goto :end
)

REM Try python3 command
python3 --version >nul 2>&1
if %errorlevel%==0 (
    echo Using python3 command...
    python3 simple_demo.py
    goto :end
)

echo.
echo ERROR: No Python installation found or Python is not in PATH.
echo.
echo Please:
echo 1. Install Python from https://www.python.org/
echo 2. Make sure Python is added to PATH during installation
echo 3. Or run the script manually if you have Python installed
echo.

:end
pause