@echo off
setlocal

:: ---------------------------------------------------------------------------
:: chat.bat — lancia l'orchestratore gg_agentic nel conda env auto_sglang
:: Usare dalla root del progetto oppure col doppio click.
:: ---------------------------------------------------------------------------

:: Radice del progetto = cartella dove si trova questo .bat
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: Nome dell'ambiente conda da attivare
set "CONDA_ENV=auto_sglang"

:: --- Se siamo gia' nell'env giusto, salta l'attivazione ---
if "%CONDA_DEFAULT_ENV%"=="%CONDA_ENV%" (
    echo [chat.bat] Ambiente conda "%CONDA_ENV%" gia' attivo.
    goto :run
)

:: --- Cerca conda in miniforge3 (priorita') poi anaconda3 ---
set "CONDA_BASE="
if exist "%USERPROFILE%\AppData\Local\miniforge3\Scripts\activate.bat" (
    set "CONDA_BASE=%USERPROFILE%\AppData\Local\miniforge3"
) else if exist "%USERPROFILE%\AppData\Local\anaconda3\Scripts\activate.bat" (
    set "CONDA_BASE=%USERPROFILE%\AppData\Local\anaconda3"
) else if exist "%ProgramData%\miniforge3\Scripts\activate.bat" (
    set "CONDA_BASE=%ProgramData%\miniforge3"
) else if exist "%ProgramData%\Anaconda3\Scripts\activate.bat" (
    set "CONDA_BASE=%ProgramData%\Anaconda3"
)

if "%CONDA_BASE%"=="" (
    echo [ERRORE] conda non trovato. Installa miniforge3 o anaconda3.
    exit /b 1
)

echo [chat.bat] Attivazione conda da: %CONDA_BASE%
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_ENV%"
if errorlevel 1 (
    echo [ERRORE] Impossibile attivare l'ambiente "%CONDA_ENV%".
    exit /b 1
)

:run
echo [chat.bat] Avvio gg_agentic\chat.py ...
echo.
python gg_agentic\chat.py
