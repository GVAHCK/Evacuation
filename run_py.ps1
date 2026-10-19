$PYTHON = "C:\Users\venka\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
$appDataSite = "C:\Users\venka\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\site-packages"
$sitePackages = Join-Path $PSScriptRoot ".venv\Lib\site-packages"
$backendDir = Join-Path $PSScriptRoot "backend"
$env:PYTHONPATH = "$PSScriptRoot;$backendDir;$appDataSite;$sitePackages"
$env:SUMO_HOME = Join-Path $sitePackages "sumo"
$env:PATH = (Join-Path $env:SUMO_HOME "bin") + ";" + $env:PATH
& $PYTHON $args
