SleepyDetect Portable Package
=============================

Target PC requirements
- Windows 10/11 (or industrial Windows), 64-bit
- Install VC++ 2015-2022 x64 Redistributable if missing
- Camera drivers working (for realtime detection)
- Path without Chinese characters (example: D:\SleepyDetect_Portable)

Deploy steps
1. Copy the whole SleepyDetect_Portable folder to the industrial PC
2. Double-click start.bat
3. Open browser: http://127.0.0.1:8000/
4. Login with existing account, or register at /accounts/register/

Custom port
  start.bat 8001

Notes
- First launch may take tens of seconds while loading YOLO/dlib models.
- This is a full portable bundle (Python + venv + app + weights), not a single exe.
- PyTorch / OpenCV / dlib make a single-file exe unreliable on industrial PCs.
