@echo off
set "PYTHON=C:\Users\venka\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
set "PYTHONPATH=%~dp0;%~dp0backend;C:\Users\venka\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\site-packages;%~dp0.venv\Lib\site-packages"
set "SUMO_HOME=%~dp0.venv\Lib\site-packages\sumo"
set "PATH=%SUMO_HOME%\bin;%PATH%"
"%PYTHON%" %*
