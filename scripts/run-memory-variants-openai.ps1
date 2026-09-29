# Compara 10/20/40 fatos na primeira conversa, reutilizando a execucao v2 salva.
param(
    [string]$Python = "",
    [string]$SourceRun = "runs\local-plans-v2-gpt4omini-all",
    [string]$Output = "runs\memory-reflector-conv00",
    [string]$Model = "gpt-4o-mini",
    [int]$Conversation = 0,
    [int]$Questions = 0,
    [string]$BaseUrl = "",
    [ValidateSet("statement", "triple")][string]$Body = "statement",
    [int]$TargetMaxTokens = 384,
    [int]$ReflectorMaxTokens = 1024,
    [int]$ReaderMaxTokens = 128,
    [switch]$MemoryOnly,
    [switch]$DryRun,
    [switch]$AllowCodeUpdate
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = "python"
    foreach ($candidate in @(".venv-bench\Scripts\python.exe", ".venv\Scripts\python.exe", "venv\Scripts\python.exe")) {
        $candidatePath = Join-Path $Root $candidate
        if (Test-Path -LiteralPath $candidatePath) { $Python = $candidatePath; break }
    }
}
Push-Location $Root
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONUNBUFFERED = "1"
    $flags = @("--source-run", $SourceRun, "--output", $Output,
        "--conversation", "$Conversation", "--questions", "$Questions",
        "--facts", "10", "20", "40", "--model", $Model, "--body", $Body,
        "--target-max-tokens", "$TargetMaxTokens", "--reflector-max-tokens", "$ReflectorMaxTokens",
        "--reader-max-tokens", "$ReaderMaxTokens", "--resume")
    if ($BaseUrl) { $flags += @("--base-url", $BaseUrl) }
    if ($MemoryOnly) { $flags += "--memory-only" }
    if ($DryRun) { $flags += "--dry-run" }
    if ($AllowCodeUpdate) { $flags += "--allow-code-update" }
    & $Python (Join-Path $PSScriptRoot "run-memory-variants.py") @flags
    if ($LASTEXITCODE -ne 0) { throw "Comparacao interrompida. Confira o erro acima; a retomada usa a mesma pasta." }
}
finally { Pop-Location }
