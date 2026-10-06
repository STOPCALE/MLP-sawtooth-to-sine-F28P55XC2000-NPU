@echo off
rem ==============================================================
rem  一键运行 saw2sin_mlp.py 的辅助脚本 (双击即可)
rem
rem  作用: 先激活 conda 环境 torch, 再运行例程.
rem  为什么要激活: conda 环境里的 numpy 依赖 MKL 数学库,
rem  这些 DLL 位于 <环境>\Library\bin, 只有激活环境时才会加入
rem  DLL 搜索路径; 直接裸调 python.exe 会因找不到 DLL 而崩溃.
rem ==============================================================
call E:\anaconda\Scripts\activate.bat E:\anaconda\envs\torch
python "%~dp0saw2sin_mlp.py"
pause
