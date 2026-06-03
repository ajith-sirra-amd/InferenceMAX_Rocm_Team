# download_artifacts.ps1 — Download GitHub Actions artifacts for a given run.
#
# Usage:
#   .\scripts\download_artifacts.ps1 -RunId <run_id> [-OutDir <dir>] [-Filter <pattern>]
#
# Examples:
#   # Download all artifacts from a run
#   .\scripts\download_artifacts.ps1 -RunId 26763854552
#
#   # Download to a custom directory
#   .\scripts\download_artifacts.ps1 -RunId 26763854552 -OutDir C:\Downloads\run_xyz
#
#   # Download only aggregated results and per-run bmk/agentic data (skip heavy server_logs)
#   .\scripts\download_artifacts.ps1 -RunId 26763854552 `
#       -Filter "agentic_aggregated|results_bmk|^bmk_agentic|^agentic_"
#
# Requirements:
#   - gh CLI (https://cli.github.com/) authenticated with an OAuth token.
#     The ROCm org blocks classic PATs (ghp_*). If $env:GITHUB_TOKEN is set to
#     a classic PAT, clear it before running:
#       $env:GITHUB_TOKEN = $null

param(
    [Parameter(Mandatory)][string]$RunId,
    [string]$OutDir = ".\artifacts\$RunId",
    [string]$Filter = "",
    [string]$Repo   = "ROCm/InferenceMAX_rocm"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "gh CLI not found. Install from https://cli.github.com/"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "==> Fetching artifact list for run $RunId ..." -ForegroundColor Cyan

# Collect all artifacts (gh API paginates at 100)
$allArtifacts = @()
$page = 1
do {
    $response = gh api "repos/$Repo/actions/runs/$RunId/artifacts?per_page=100&page=$page" | ConvertFrom-Json
    $allArtifacts += $response.artifacts
    $page++
} while ($response.artifacts.Count -eq 100)

Write-Host "==> Found $($allArtifacts.Count) artifacts." -ForegroundColor Cyan

# Apply name filter
if ($Filter) {
    $allArtifacts = $allArtifacts | Where-Object { $_.name -match $Filter }
    Write-Host "==> Filter matched $($allArtifacts.Count) artifacts." -ForegroundColor Yellow
}

# Print list
Write-Host ""
foreach ($a in $allArtifacts) {
    $sizeMb  = [math]::Floor($a.size_in_bytes / 1MB)
    $expired = if ($a.expired) { " [EXPIRED]" } else { "" }
    Write-Host ("  {0,-80} {1,4} MB{2}" -f $a.name, $sizeMb, $expired)
}
Write-Host ""

$downloaded = 0; $skipped = 0; $failed = 0

foreach ($artifact in $allArtifacts) {
    $dest = Join-Path (Resolve-Path $OutDir) $artifact.name

    if ($artifact.expired) {
        Write-Host "  SKIP (expired):  $($artifact.name)" -ForegroundColor DarkGray
        $skipped++; continue
    }
    if (Test-Path $dest) {
        Write-Host "  SKIP (exists):   $($artifact.name)" -ForegroundColor DarkGray
        $skipped++; continue
    }

    Write-Host -NoNewline "  Downloading: $($artifact.name) ... "
    try {
        # gh run download handles binary zip extraction natively
        gh run download $RunId `
            --repo $Repo `
            --name $artifact.name `
            --dir $dest 2>&1 | Out-Null

        if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }

        Write-Host "OK" -ForegroundColor Green
        $downloaded++
    }
    catch {
        Write-Host "FAILED ($_)" -ForegroundColor Red
        if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
        $failed++
    }
}

Write-Host ""
Write-Host "==> Done.  Downloaded: $downloaded   Skipped: $skipped   Failed: $failed" -ForegroundColor Cyan
Write-Host "==> Output: $(Resolve-Path $OutDir)"
