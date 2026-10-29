@echo off
rem build_ml.bat - full GBDT pipeline: dump features, train + codegen, build, verify
rem NOTE: keep this file ASCII-only (cp932 console mangles Japanese comments).
rem
rem Training seeds 42-45, threshold picked on the held-out tail of the same set.
rem seed 777 is reserved for the out-of-sample backtest (backtest_ml.exe).
rem A "null" dataset (no information in the order flow) is produced alongside so
rem train_export.py can report how much of the score is real information.
setlocal
cd /d "%~dp0"
set FLAGS=-O2 -std=c++23 -march=native -Wall -Wextra -static
set PY=..\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [1/5] dump_features.exe
g++ %FLAGS% dump_features.cpp -o dump_features.exe || exit /b 1

echo [2/5] dump features (200k events/day x 6 days, seeds 42-45 + null)
.\dump_features.exe 200000 6 42 features.csv || exit /b 1
for %%s in (43 44 45) do (
    .\dump_features.exe 200000 6 %%s features_s%%s.csv || exit /b 1
)
.\dump_features.exe 200000 6 42 features_null.csv --null || exit /b 1

echo [3/5] train LightGBM + generate model.hpp
"%PY%" train_export.py --features "features.csv,features_s43.csv,features_s44.csv,features_s45.csv" --null-features features_null.csv --trees 300 --leaves 15 --lr 0.03 --fire-pct 95 || exit /b 1

echo [4/5] backtest_ml.exe
g++ %FLAGS% backtest_ml.cpp -o backtest_ml.exe || exit /b 1

echo [5/5] verify python vs c++ predictions
.\backtest_ml.exe --verify verify.csv || exit /b 1

echo.
echo done. next:
echo   backtest_ml.exe             : out-of-sample GBDT backtest (seed 777) with null baseline
