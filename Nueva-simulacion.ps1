param(
    [Parameter(Mandatory = $false)]
    [ValidateScript({ $_ -ge 100 })]
    [long]$Simulaciones = 200000,
    [int]$Semilla = 42
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $projectPython = if (Test-Path '.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }
    Write-Host "Iniciando $Simulaciones simulaciones con semilla $Semilla..."
    & $projectPython -m champions.cli simulate --simulations $Simulaciones --seed $Semilla
    if ($LASTEXITCODE -ne 0) {
        throw 'La simulación ha fallado. Revisa el mensaje anterior.'
    }
} finally {
    Pop-Location
}
