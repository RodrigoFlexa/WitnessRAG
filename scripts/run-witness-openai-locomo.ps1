# LoCoMo (10 conversas) com WitnessRAG usando a API pública da OpenAI (Windows).
# Mesmo perfil registrado de scripts/run-witness-azure-locomo.sh; só o LLM muda.
#
#   1. OPENAI_API_KEY=... no arquivo .env na raiz do repositório
#   2. powershell -ExecutionPolicy Bypass -File scripts\run-witness-openai-locomo.ps1
#
# Smoke test:  ... -Conversation 0 -Questions 5 -Output runs\smoke-openai
# Controlador de prova: -Method proof -Profile proof-v4 (perfis em
# scripts/proof-profiles.sh). Varredura: -TopK 20 -ChunkTokens 512 -Pool 40
# -ExtraFlags "--proof-edit-fraction 0.4".
param(
    [string]$Output = "",
    [string]$Model = "gpt-4o-mini",
    [string]$Method = "witnessrag",
    [string]$Profile = "proof",
    [int]$TopK = 5,
    [int]$ChunkTokens = 2048,
    [int]$Pool = 20,
    [string]$ExtraFlags = "",
    [string]$Conversation = "all",
    [int]$Questions = 0,
    [int]$Concurrency = 8,
    [string]$EmbedModel = "BAAI/bge-m3",
    [string]$EmbedDevice = "cpu",
    [string]$Python = "",
    [double]$Hours = 72
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Python) {
    $Python = "python"
    foreach ($c in @(".venv-bench\Scripts\python.exe", ".venv\Scripts\python.exe", "venv\Scripts\python.exe")) {
        if (Test-Path $c) { $Python = (Resolve-Path $c).Path; break }
    }
}
if (-not $Output) {
    $Suffix = if ($Method -eq "proof") { $Profile } else { $Method }
    $Output = "runs\witness-suite-locomo-openai-$Model-$Suffix"
}
$CacheDir = Join-Path $Root "runs\.cache\witness-openai"
New-Item -ItemType Directory -Force -Path $Output, $CacheDir | Out-Null

# A chave vem do .env (lido pelo próprio wrag.config) ou do ambiente.
if (-not $env:OPENAI_API_KEY -and -not (Select-String -Path ".env" -Pattern "^\s*OPENAI_API_KEY\s*=\s*\S" -Quiet -ErrorAction SilentlyContinue)) {
    throw "Defina OPENAI_API_KEY no .env ou no ambiente."
}

$env:WRAG_LLM_BACKEND = "openai"
$env:OPENAI_MODEL = $Model
$env:WRAG_CONTINUE_ON_CONTENT_FILTER = "1"
$env:WRAG_EMBED_BACKEND = "st"
$env:WRAG_EMBED_MODEL = $EmbedModel
$env:WRAG_EMBED_DEVICE = $EmbedDevice
$env:WRAG_CACHE_DIR = $CacheDir
$env:WRAG_LLM_CACHE = "1"; $env:WRAG_EMBED_CACHE = "1"; $env:PYTHONHASHSEED = "42"
$env:WRAG_AZURE_CONCURRENCY = "$Concurrency"
$env:PYTHONUTF8 = "1"

& $Python -m wrag.cli diag-openai --model $Model
if ($LASTEXITCODE -ne 0) { throw "diag-openai falhou; confira a chave e o modelo." }

$flags = @("--backend", "openai", "--model", $Model, "--concurrency", "$Concurrency", "--gpu", "0",
    "--tokenizer-model", "Qwen/Qwen2.5-14B-Instruct",
    "--dataset", "locomo", "--locomo-conversation", $Conversation,
    "--methods", $(if ($Method -eq "proof") { "witnessrag" } else { $Method }),
    "--embed-model", $EmbedModel, "--embed-device", $EmbedDevice,
    "--locomo-chunk-tokens", "$ChunkTokens", "--locomo-ie-window-tokens", "512", "--top-k", "$TopK", "--qa-max-tokens", "128",
    "--witness-candidate-pool", "$Pool", "--answer-set", "--temporal-annotations", "--evidence-reader",
    "--cache-dir", $CacheDir, "--hours", "$Hours", "--output", $Output)
if ($Questions -gt 0) { $flags += @("--questions", "$Questions") }
switch ($Method) {
    "witnessrag" { $flags += @("--binding-aware-grounding", "--vocab-compile", "--hybrid-fallback",
                               "--dialogue-ie", "--selective-witness", "--gap-context-rescue") }
    "hybrid" { }
    "proof" {
        # Mesmos perfis de scripts/proof-profiles.sh.
        $flags += @("--binding-aware-grounding", "--vocab-compile", "--hybrid-fallback",
                    "--dialogue-ie", "--gap-context-rescue", "--proof-controller")
        switch ($Profile) {
            "proof" { }
            "proof-no-verify" { $flags += "--no-proof-verify" }
            "proof-partial" { $flags += "--partial-evidence" }
            "proof-1cycle" { $flags += @("--proof-cycles", "1") }
            "proof-v4" { $flags += @("--typed-variables", "--item-set-proofs", "--witness-delivery",
                                     "mixed", "--abductive-premises") }
            "v4-typed" { $flags += @("--typed-variables", "--item-set-proofs") }
            "v4-excerpts" { $flags += @("--witness-delivery", "excerpts") }
            "v4-abductive" { $flags += "--abductive-premises" }
            "v4-no-types" { $flags += @("--item-set-proofs", "--witness-delivery", "mixed",
                                        "--abductive-premises") }
            "v4-mixed" { $flags += @("--witness-delivery", "mixed") }
            default { throw "Profile desconhecido: $Profile" }
        }
    }
    default { throw "Method deve ser witnessrag, hybrid ou proof." }
}
if ($ExtraFlags) { $flags += ($ExtraFlags -split "\s+" | Where-Object { $_ }) }
if (Test-Path (Join-Path $Output "pilot.json")) { $flags += "--resume" }

& $Python -m wrag.pilot @flags
exit $LASTEXITCODE
