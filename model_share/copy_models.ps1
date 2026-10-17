$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$store = Join-Path $env:USERPROFILE '.ollama\models\blobs'

$pairs = @(
    @{ src = 'sha256-b9a2f9f354d0d12c282353129961a4327d4774493bde2ec59c9236cb62bb5eaf'; dst = 'aria-goal.gguf' },
    @{ src = 'sha256-74799d4b582bdd89493ae9b53b25cc853e314fb527d8f6451b5ea5f02534b14a'; dst = 'base-4b.gguf' },
    @{ src = 'sha256-cb5bddd781b9e39053af5ef36ae9c8edbb1f196a5ab8412ea19e84bb0a70fb6a'; dst = 'sql-adapter.gguf' },
    @{ src = 'sha256-c4e4885997a080518e9066c8c02a11be532f201a4f3ed69ea1c5ba7bc652e0c5'; dst = 'story-adapter.gguf' }
)

foreach ($p in $pairs) {
    $src = Join-Path $store $p.src
    $dst = Join-Path $root $p.dst
    if (-not (Test-Path -LiteralPath $src)) { Write-Host "MISSING blob: $src"; continue }
    Copy-Item -LiteralPath $src -Destination $dst
    Write-Host "copied -> $($p.dst)"
}
Write-Host "`nDone. Zip the whole model_share folder and send it to your mates."