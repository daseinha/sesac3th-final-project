@echo off
REM NoteMate launcher -- a Desktop shortcut points to this file
REM (2026-10-08, see deferred-launch-exit-ux memory note).
REM Lets the demo start the service with one icon click, no terminal needed.
REM
REM Flow: (1) check if the server is already up -> (2) if not, start it
REM (minimized window, keeps running in background) -> (3) health check
REM (retry http://localhost:8000/ until it returns 200) -> (4) open the
REM browser in app mode (no address bar/tabs, looks like a real app).
REM On failure, show a readable message instead of a raw traceback.
REM
REM NOTE: keep this file ASCII-only. Korean text in .bat files has caused
REM cmd.exe parsing errors before (codepage mismatch) -- same reason .env
REM stays English-only (see env-file-english-only memory note).

setlocal
cd /d "%~dp0\.."

set URL=http://localhost:8000/
set APP_URL=http://localhost:8000
set HEALTH_FILE=%TEMP%\notemate_health_%RANDOM%.txt

echo Preparing NoteMate...

call :check_health
if "%HEALTH%"=="200" (
    echo Already running. Opening the screen.
    goto :open_browser
)

echo Starting the server (first start may take a few seconds)...
start "NoteMate server (do not close -- minimize instead)" /min cmd /c "uv run uvicorn src.server:app --port 8000"

set RETRY=0
:wait_loop
REM "timeout" needs a real console on STDIN and fails ("Input redirection is
REM not supported") when launched in some non-interactive contexts -- using
REM ping as a ~1s delay instead avoids that (classic batch-file trick, no
REM console/stdin dependency).
ping -n 2 127.0.0.1 >nul
set /a RETRY+=1
call :check_health
if "%HEALTH%"=="200" goto :open_browser
if %RETRY% GEQ 30 goto :fail
goto :wait_loop

:open_browser
REM Try Chrome first, then Edge, then whatever the default browser is.
REM "start" cannot reliably report "program not found" via errorlevel, so
REM check well-known install paths directly instead.
REM --start-maximized: open full-size right away, so a small/odd leftover
REM window size never causes the bottom bar to look squeezed/overlapping
REM during the demo.
set CHROME_PATH=
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set CHROME_PATH=%ProgramFiles%\Google\Chrome\Application\chrome.exe
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set CHROME_PATH=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set CHROME_PATH=%LocalAppData%\Google\Chrome\Application\chrome.exe
if defined CHROME_PATH (
    start "" "%CHROME_PATH%" --app=%APP_URL% --start-maximized
    exit /b 0
)

set EDGE_PATH=
if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set EDGE_PATH=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe
if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set EDGE_PATH=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe
if defined EDGE_PATH (
    start "" "%EDGE_PATH%" --app=%APP_URL% --start-maximized
    exit /b 0
)

REM Neither found by path -- fall back to the default browser, normal window.
start %URL%
exit /b 0

:check_health
REM "< nul" gives curl an explicit (empty) stdin -- without it curl.exe can
REM sometimes fail immediately with "Input redirection is not supported"
REM depending on how this script itself was launched (known Windows quirk).
curl -s -o nul -w "%%{http_code}" %URL% < nul > "%HEALTH_FILE%" 2>nul
set /p HEALTH=<"%HEALTH_FILE%"
del "%HEALTH_FILE%" 2>nul
exit /b 0

:fail
echo.
echo ============================================
echo  Could not start the NoteMate server.
echo  Please check:
echo   - API keys / DB connection info in .env
echo   - whether port 8000 is already used by something else
echo   - your internet connection
echo ============================================
echo.
pause
exit /b 1
