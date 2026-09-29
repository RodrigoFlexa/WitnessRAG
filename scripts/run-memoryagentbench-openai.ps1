# MemoryAgentBench: standard WitnessRAG local-v2, isolated from LoCoMo runs.
param(
    [string]$Python = "python",
    [string]$Model = "gpt-4o-mini",
    [ValidateSet("paper", "official")][string]$Protocol = "paper",
    [ValidateSet("all", "paper")][string]$Suite = "all",
    [ValidateSet("auto", "cuda", "cpu")][string]$EmbedDevice = "auto",
    [string]$EmbedModel = "BAAI/bge-m3",
    [string[]]$Sources = @(),
    [string[]]$Splits = @(),
    [int]$MaxQuestions = 0,
    [int]$MaxContexts = 0,
    [int]$FactBudget = 40,
    [string]$FactRerank = "cross-encoder/ms-marco-MiniLM-L6-v2",
    [ValidateRange(0, 1)][double]$ConflictRecencyWeight = 0.65,
    [string]$Output = "runs\memoryagentbench-standard-paper",
    [string]$Cache = "runs\.cache\memoryagentbench",
    [switch]$PrepareOnly,
    [switch]$InspectOnly,
    [switch]$Judge,
    [switch]$NoReflection
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$oldHashSeed = $env:PYTHONHASHSEED
$oldEncoding = $env:PYTHONIOENCODING
try {
    Push-Location $Root
    $env:PYTHONHASHSEED = "0"
    $env:PYTHONIOENCODING = "utf-8"
    & $Python -X utf8 -u -m benchmarks.memoryagentbench prepare --cache $Cache
    if ($LASTEXITCODE -ne 0) { throw "Preparacao falhou; confira o erro acima." }
    if ($PrepareOnly) { return }
    $selectionArgs = @("--suite", $Suite, "--protocol", $Protocol)
    if ($Sources.Count) { $selectionArgs += "--sources"; $selectionArgs += $Sources }
    if ($Splits.Count) { $selectionArgs += "--splits"; $selectionArgs += $Splits }
    if ($InspectOnly) {
        & $Python -X utf8 -u -m benchmarks.memoryagentbench inspect --cache $Cache --chunks @selectionArgs
        if ($LASTEXITCODE -ne 0) { throw "Validacao do protocolo falhou." }
        return
    }
    $runArgs = @("run", "--cache", $Cache, "--output", $Output, "--backend", "openai",
        "--model", $Model, "--embed-device", $EmbedDevice, "--embed-model", $EmbedModel,
        "--fact-budget", "$FactBudget", "--fact-rerank=$FactRerank",
        "--conflict-recency-weight", $ConflictRecencyWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--max-questions", "$MaxQuestions", "--max-contexts", "$MaxContexts", "--resume")
    $runArgs += $selectionArgs
    if ($NoReflection) { $runArgs += "--no-reflection" }
    & $Python -X utf8 -u -m benchmarks.memoryagentbench @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Execucao interrompida. Rode o mesmo comando para retomar." }
    if ($Judge) {
        & $Python -X utf8 -u -m benchmarks.memoryagentbench judge --cache $Cache --output $Output
        if ($LASTEXITCODE -ne 0) { throw "Avaliador interrompido. Predicoes salvas; retome o comando judge." }
    }
} finally {
    Pop-Location
    $env:PYTHONHASHSEED = $oldHashSeed
    $env:PYTHONIOENCODING = $oldEncoding
}
