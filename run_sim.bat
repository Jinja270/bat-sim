@echo off
title Bat Swarm Neuroevolution - Dual Window Studio (60 FPS)
cd /d "%~dp0"
echo =====================================================================
echo  Launching Dual Studio: Swarm Maze (Left) + Neural Network (Right)
echo =====================================================================
echo.
python maze.py --neural-window
if errorlevel 1 (
    echo.
    echo Simulation exited with an error.
    pause
)
