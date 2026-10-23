@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYEXE=python"
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
:menu
cls
echo ======================================================
echo      SRT-M15 + Qwen3 4B / RTX 3080 LOCAL SUITE
echo ======================================================
echo.
echo [1] Check Python / CUDA / Ollama
echo [2] Chat now with pretrained local Qwen
echo [3] Generate first 40 bilingual Qwen teaching drafts
echo [4] Scan first batch for human review
echo [5] Export HUMAN-approved first batch
echo [6] Train original SRT-M15 student on reviewed data
echo [7] Chat with trained ORIGINAL SRT-M15 student
echo [8] Evaluate student against heldout dialogue cases
echo [9] Show exact review-approval instructions
echo [0] Exit
echo.
set /p PICK=Choose number: 
if "%PICK%"=="1" goto doctor
if "%PICK%"=="2" goto chat
if "%PICK%"=="3" goto generate
if "%PICK%"=="4" goto scan
if "%PICK%"=="5" goto export
if "%PICK%"=="6" goto train
if "%PICK%"=="7" goto student
if "%PICK%"=="8" goto evaluate
if "%PICK%"=="9" goto reviewhelp
if "%PICK%"=="0" goto finish
echo Invalid selection.
goto back
:doctor
"%PYEXE%" srt_full_pipeline.py doctor
goto back
:chat
"%PYEXE%" srt_full_pipeline.py chat
goto back
:generate
"%PYEXE%" srt_full_pipeline.py generate --pairs 40 --offset 0
goto back
:scan
"%PYEXE%" srt_full_pipeline.py review --drafts local_data/qwen_teacher/batch_000000_000040.jsonl --reviews local_data/qwen_teacher/reviews.jsonl
goto back
:export
"%PYEXE%" srt_full_pipeline.py export --drafts local_data/qwen_teacher/batch_000000_000040.jsonl --reviews local_data/qwen_teacher/reviews.jsonl --output local_data/qwen_teacher/reviewed_sft.jsonl
goto back
:train
"%PYEXE%" srt_full_pipeline.py train --data local_data/qwen_teacher/reviewed_sft.jsonl --device auto --output checkpoints/qwen_student
goto back
:student
"%PYEXE%" srt_full_pipeline.py student
goto back
:evaluate
"%PYEXE%" srt_full_pipeline.py evaluate
goto back
:reviewhelp
echo.
echo IMPORTANT: Before approval, read the Qwen response and verify its
echo correctness, privacy, suitability and provenance/usage rights.
echo.
echo Example for manually approving a verified example:
echo python review_deepseek_learning.py --drafts local_data/qwen_teacher/batch_000000_000040.jsonl --reviews local_data/qwen_teacher/reviews.jsonl approve "qwen-pair-000000-zh" --rights-confirmed --evidence "Manually checked evidence and source rights."
echo.
echo The ID is ONLY an example. Use the real ID from menu option [4].
echo See README_SRT_FULL_SUITE.md for details.
goto back
:back
echo.
pause
goto menu
:finish
endlocal
