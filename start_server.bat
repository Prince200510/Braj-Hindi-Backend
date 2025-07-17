@echo off
echo Starting Braj-Hindi Translation API Server...
echo.
echo Make sure you have installed the requirements:
echo pip install -r requirements.txt
echo.
echo Starting server on http://localhost:8000
echo.
cd /d "p:\Ai Agent\Vraj Project\backend"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
