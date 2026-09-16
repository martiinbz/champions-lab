param([int]$Simulaciones = 20000, [int]$Semilla = 42)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -m champions.cli update
    if ($LASTEXITCODE -ne 0) { throw 'La descarga ha fallado; no se crea una previsión con datos incompletos.' }
    python -m champions.cli simulate --simulations $Simulaciones --seed $Semilla
    if ($LASTEXITCODE -ne 0) { throw 'La simulación ha fallado. Revisar el mensaje anterior.' }
} finally {
    Pop-Location
}
