@echo off
rem Double-click: build and flash a Mirra voice satellite (asks for the room name)
set /p NAME="Room name of this speaker (e.g. Kueche): "
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0flash.ps1" -Name "%NAME%" %*
pause
