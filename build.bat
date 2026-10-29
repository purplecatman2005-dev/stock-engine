@echo off
rem build.bat - build the core engine with g++ (MinGW-w64)
rem NOTE: keep this file ASCII-only. Japanese comments get mangled by the
rem       cp932 console and show up as bogus command errors.
setlocal
cd /d "%~dp0"
set FLAGS=-O2 -std=c++23 -march=native -Wall -Wextra -static
echo [1/4] bench.exe
g++ %FLAGS% bench.cpp -o bench.exe || exit /b 1
echo [2/4] backtest.exe
g++ %FLAGS% backtest.cpp -o backtest.exe || exit /b 1
echo [3/4] live.exe
g++ %FLAGS% live.cpp -o live.exe -lws2_32 || exit /b 1
echo [4/4] replay.exe
g++ %FLAGS% replay.cpp -o replay.exe || exit /b 1
echo.
echo done. next:
echo   bench.exe                    : verify tick table + book vs std::map, measure speed
echo   backtest.exe                 : OFI signal backtest with null baseline
echo   live.exe                     : receive from adapter on TCP 9501, record to captures\
echo   replay.exe captures\xxx.bin  : verify and score a recording
