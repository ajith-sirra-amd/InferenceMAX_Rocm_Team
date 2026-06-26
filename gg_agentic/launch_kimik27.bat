@echo off
setlocal EnableDelayedExpansion

:: ---------------------------------------------------------------------------
:: launch_kimik27.bat
::
:: Verifica che nessun benchmark stia girando sui runner mi355x, poi
:: dispatcha e2e-tests.yml con kimik2.7-fp4-mi355x-vllm-agentic-lmcache
::
:: Uso:
::   launch_kimik27.bat [--ref BRANCH] [--force] [--dry-run]
:: ---------------------------------------------------------------------------

set "WORKFLOW=e2e-tests.yml"
set "CONFIG_KEY=kimik2.7-fp4-mi355x-vllm-agentic-lmcache"
set "CONFIG_FILES=.github/configs/amd-master.yaml"
set "RUNNER_LABEL=mi355x"
set "REPO=ROCm/InferenceMAX_rocm"
set "REF="
set "FORCE=0"
set "DRY_RUN=0"

:: --- Parsing argomenti ---
:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--ref"     ( set "REF=%~2" & shift & shift & goto :parse_args )
if "%~1"=="--force"   ( set "FORCE=1" & shift & goto :parse_args )
if "%~1"=="--dry-run" ( set "DRY_RUN=1" & shift & goto :parse_args )
echo Argomento sconosciuto: %~1
exit /b 1
:done_args

:: --- Branch di default = branch corrente ---
if "%REF%"=="" (
    for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "REF=%%b"
    if "!REF!"=="" set "REF=chore/agentx-v0.4"
)

:: --- Rimappa GITHUB_TOKEN su GH_TOKEN (il classic PAT e' bloccato da ROCm) ---
if defined GITHUB_TOKEN (
    if not defined GH_TOKEN (
        set "GH_TOKEN=%GITHUB_TOKEN%"
    )
    set "GITHUB_TOKEN="
)

:: --- Verifica gh autenticato ---
gh auth status >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] gh CLI non autenticato. Esegui: gh auth login
    exit /b 1
)

echo ========================================
echo  launch_kimik27.bat
echo  Repo   : %REPO%
echo  Ref    : %REF%
echo  Config : %CONFIG_KEY%
echo ========================================
echo.

:: ---------------------------------------------------------------------------
:: Controlla run attive su mi355x
:: ---------------------------------------------------------------------------
echo [*] Controllo run attive su runner '%RUNNER_LABEL%' ...

:: Scarica le run in_progress in un file temporaneo
set "TMPFILE=%TEMP%\gh_runs_%RANDOM%.json"

gh run list --repo "%REPO%" --workflow "%WORKFLOW%" --status in_progress ^
    --json databaseId,displayTitle --limit 20 > "%TMPFILE%" 2>nul
if errorlevel 1 ( echo [] > "%TMPFILE%" )

:: Controlla anche le run in coda
set "TMPFILE2=%TEMP%\gh_runs2_%RANDOM%.json"
gh run list --repo "%REPO%" --workflow "%WORKFLOW%" --status queued ^
    --json databaseId,displayTitle --limit 20 > "%TMPFILE2%" 2>nul
if errorlevel 1 ( echo [] > "%TMPFILE2%" )

:: Usa python per estrarre gli ID e cercare job su mi355x
set "CHECKSCRIPT=%TEMP%\gh_check_%RANDOM%.py"
(
echo import json, subprocess, sys
echo.
echo runner = "%RUNNER_LABEL%"
echo repo   = "%REPO%"
echo.
echo def load^(path^):
echo     try:
echo         with open^(path^) as f: return json.load^(f^)
echo     except: return []
echo.
echo runs = load^(r"%TMPFILE%"^) + load^(r"%TMPFILE2%"^)
echo if not runs:
echo     print^("FREE"^)
echo     sys.exit^(0^)
echo.
echo busy = []
echo for r in runs:
echo     rid = r["databaseId"]
echo     try:
echo         out = subprocess.check_output^(
echo             ["gh", "run", "view", str^(rid^), "--repo", repo, "--json", "jobs"],
echo             stderr=subprocess.DEVNULL
echo         ^)
echo         jobs = json.loads^(out^).get^("jobs", []^)
echo     except: continue
echo     for j in jobs:
echo         labels = j.get^("labels", []^)
echo         name   = j.get^("name", ""^).lower^(^)
echo         status = j.get^("status", ""^)
echo         if runner in labels or runner in name:
echo             if status in ^("in_progress", "queued", "waiting"^):
echo                 busy.append^(f"  Run #{rid}: {j['name']} [{status}]"^)
echo.
echo if busy:
echo     print^("BUSY"^)
echo     for b in busy: print^(b^)
echo else:
echo     print^("FREE"^)
) > "%CHECKSCRIPT%"

:: Esegui il check
set "RUNNER_STATUS=FREE"
for /f "usebackq delims=" %%L in (`python "%CHECKSCRIPT%" 2^>nul`) do (
    if "%%L"=="BUSY" ( set "RUNNER_STATUS=BUSY" ) else (
    if "%%L"=="FREE" ( set "RUNNER_STATUS=FREE" ) else (
        echo %%L
    ))
)

:: Pulizia file temporanei
del /q "%TMPFILE%" "%TMPFILE2%" "%CHECKSCRIPT%" 2>nul

:: ---------------------------------------------------------------------------
:: Decisione
:: ---------------------------------------------------------------------------
if "%RUNNER_STATUS%"=="BUSY" (
    if "%FORCE%"=="1" (
        echo [ATTENZIONE] Runner occupato ma --force specificato. Procedo comunque.
    ) else (
        echo.
        echo [ERRORE] Runner '%RUNNER_LABEL%' e' occupato - porta 8888 probabilmente in uso.
        echo          Aspetta che le run attive finiscano, poi riprova.
        echo          Usa --force per ignorare questo controllo.
        exit /b 1
    )
) else (
    echo [OK] Runner '%RUNNER_LABEL%' libero.
)

:: ---------------------------------------------------------------------------
:: Dispatch
:: ---------------------------------------------------------------------------
echo.
echo [*] Dispatch:
echo      workflow : %WORKFLOW%
echo      ref      : %REF%
echo      config   : %CONFIG_KEY%
echo.

if "%DRY_RUN%"=="1" (
    echo [dry-run] Comando che verrebbe eseguito:
    echo   gh workflow run %WORKFLOW% --repo %REPO% --ref %REF% -f "generate-cli-command=test-config --config-files %CONFIG_FILES% --config-keys %CONFIG_KEY%"
    exit /b 0
)

gh workflow run "%WORKFLOW%" ^
    --repo "%REPO%" ^
    --ref "%REF%" ^
    -f "generate-cli-command=test-config --config-files %CONFIG_FILES% --config-keys %CONFIG_KEY%"

if errorlevel 1 (
    echo [ERRORE] Dispatch fallito.
    exit /b 1
)

echo.
echo [OK] Workflow dispatchato. Stato aggiornato tra pochi secondi...
echo.
timeout /t 5 /nobreak >nul

gh run list --repo "%REPO%" --workflow "%WORKFLOW%" --limit 3
