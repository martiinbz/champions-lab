param([int]$Simulaciones = 20000, [int]$Semilla = 42)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $projectPython = if (Test-Path '.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }
    & $projectPython -m champions.cli update
    if ($LASTEXITCODE -ne 0) { throw 'La descarga ha fallado; no se crea una previsión con datos incompletos.' }
    & $projectPython -m champions.cli simulate --simulations $Simulaciones --seed $Semilla
    if ($LASTEXITCODE -ne 0) { throw 'La simulación ha fallado. Revisar el mensaje anterior.' }
} finally {
    Pop-Location
}
