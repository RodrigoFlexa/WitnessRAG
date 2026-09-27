# Estudo de ablação do WitnessRAG no LoCoMo com gpt-4o-mini (Windows), docs/ablacao.md.
# Roda o método completo PRIMEIRO e depois cada ablação; no fim, a tabela pareada.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run-ablation.ps1
#   ... -ChunkTokens 512 -TopK 20                  # outro orçamento
#   ... -Variants full                             # só o completo
#   ... -Conversation 1 -Questions 5               # teste rápido
param(
    [int]$ChunkTokens = 2048,
    [int]$TopK = 5,
    [int]$Pool = 0,
    [string]$Variants = "full no-plan no-proof no-verify no-temporal-score",
    [string]$Conversation = "all",
    [int]$Questions = 0,
    [string]$EmbedDevice = "cpu",
    [string]$Root = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if ($Pool -le 0) { $Pool = [Math]::Max(20, 2 * $TopK) }
if (-not $Root) { $Root = "runs\ablation-openai-c$ChunkTokens-k$TopK" }
New-Item -ItemType Directory -Force -Path $Root | Out-Null

foreach ($variant in ($Variants -split "\s+" | Where-Object { $_ })) {
    $profile = if ($variant -eq "full") { "witnessrag" } else { "abl-$variant" }
    $out = Join-Path $Root $variant
    Write-Host "=== $(Get-Date -Format s) $variant (perfil $profile) -> $out"
    & powershell -ExecutionPolicy Bypass -File scripts\run-witness-openai-locomo.ps1 `
        -Method proof -Profile $profile -TopK $TopK -ChunkTokens $ChunkTokens -Pool $Pool `
        -Conversation $Conversation -Questions $Questions -EmbedDevice $EmbedDevice -Output $out
    Write-Host "=== $(Get-Date -Format s) $variant terminou (código $LASTEXITCODE)"
}

$py = "python"
foreach ($c in @(".venv-bench\Scripts\python.exe", ".venv\Scripts\python.exe", "venv\Scripts\python.exe")) {
    if (Test-Path $c) { $py = (Resolve-Path $c).Path; break }
}
& $py scripts\ablation-report.py $Root
