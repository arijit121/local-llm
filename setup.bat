@echo off
echo Setting up Local AI Studio Environment...

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete! To run the server, use: venv\Scripts\uvicorn main:app --reload
pause
