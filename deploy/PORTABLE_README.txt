SleepyDetect Portable Package
=============================

Target PC requirements
- Windows 10/11 (or industrial Windows), 64-bit
- Install VC++ 2015-2022 x64 Redistributable if missing
- For industrial GigE cameras: install Hikrobot MVS 4.4.0 (64-bit) with Runtime
- Camera drivers working (for system webcam mode)
- Path without Chinese characters (example: D:\SleepyDetect_Portable)

Deploy steps
1. Copy the whole SleepyDetect_Portable folder to the industrial PC
2. Double-click start.bat
3. Open browser: http://127.0.0.1:8000/
4. On realtime page choose GigE device or system webcam, then Start

Custom port
  start.bat 8001

Notes
- Close MVS client preview before starting GigE detection (device exclusive).
- First launch may take tens of seconds while loading YOLO/dlib models.
- This bundle does NOT include MVS; install MVS separately on the target PC.
