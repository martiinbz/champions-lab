$ErrorActionPreference = 'Stop'
$headers = @{ 'User-Agent' = 'ChampionsLab/0.1 (local educational dashboard)' }
$destination = Join-Path $PSScriptRoot '..\app\assets\crests'
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$pages = [ordered]@{
    'AEK Athens' = 'AEK Athens F.C.'
    'Arsenal' = 'Arsenal F.C.'
    'Aston Villa' = 'Aston Villa F.C.'
    'Atlético de Madrid' = 'Atlético Madrid'
    'Barcelona' = 'FC Barcelona'
    'Bayern München' = 'FC Bayern Munich'
    'Bodø/Glimt' = 'FK Bodø/Glimt'
    'Borussia Dortmund' = 'Borussia Dortmund'
    'Club Brugge' = 'Club Brugge KV'
    'Como' = 'Como 1907'
    'Fenerbahçe' = 'Fenerbahçe S.K. (football)'
    'Feyenoord' = 'Feyenoord'
    'Galatasaray' = 'Galatasaray S.K. (football)'
    'Inter' = 'Inter Milan'
    'LASK' = 'LASK'
    'Leipzig' = 'RB Leipzig'
    'Lens' = 'RC Lens'
    'Lille' = 'Lille OSC'
    'Liverpool' = 'Liverpool F.C.'
    'Manchester City' = 'Manchester City F.C.'
    'Manchester United' = 'Manchester United F.C.'
    'Napoli' = 'SSC Napoli'
    'PSV Eindhoven' = 'PSV Eindhoven'
    'Paris Saint-Germain' = 'Paris Saint-Germain F.C.'
    'Porto' = 'FC Porto'
    'Real Betis' = 'Real Betis'
    'Real Madrid' = 'Real Madrid CF'
    'Roma' = 'AS Roma'
    'Sabah' = 'Sabah FC (Azerbaijan)'
    'Shakhtar Donetsk' = 'FC Shakhtar Donetsk'
    'Slavia Praha' = 'SK Slavia Prague'
    'Slovan Bratislava' = 'ŠK Slovan Bratislava'
    'Sporting CP' = 'Sporting CP'
    'Stuttgart' = 'VfB Stuttgart'
    'Viking' = 'Viking FK'
    'Villarreal' = 'Villarreal CF'
}

$files = [ordered]@{}
$sources = [ordered]@{}
$counter = 0
foreach ($team in $pages.Keys) {
    $counter += 1
    $title = $pages[$team]
    $filename = ('{0:d2}.png' -f $counter)
    $target = Join-Path $destination $filename
    $encoded = [System.Uri]::EscapeDataString($title)
    $summaryUrl = "https://en.wikipedia.org/api/rest_v1/page/summary/$encoded"
    Start-Sleep -Milliseconds 1100
    $summary = Invoke-RestMethod -Uri $summaryUrl -Headers $headers -TimeoutSec 30
    $imageUrl = $summary.thumbnail.source
    if (-not $imageUrl) { throw "No crest image found for $team ($title)" }
    if (-not (Test-Path $target)) {
        Invoke-WebRequest -Uri $imageUrl -Headers $headers -OutFile $target -TimeoutSec 60
    }
    $files[$team] = $filename
    $sources[$team] = [ordered]@{ page = $summary.content_urls.desktop.page; image = $imageUrl }
}

$files | ConvertTo-Json | Set-Content -Path (Join-Path $destination 'team-crests.json') -Encoding utf8
$sources | ConvertTo-Json -Depth 3 | Set-Content -Path (Join-Path $destination 'sources.json') -Encoding utf8
Write-Host "Downloaded $($files.Count) crest images."
