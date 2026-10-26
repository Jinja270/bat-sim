@echo off
title Bat Swarm Maze Navigation Simulation
cd /d "%~dp0"
echo =====================================================================
echo  Launching Autonomous Swarm Maze Simulation (60 FPS)
echo =====================================================================
echo.
python maze.py
if errorlevel 1 (
    echo.
    echo Simulation exited with an error.
    pause
)
