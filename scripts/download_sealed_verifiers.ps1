param(
    [string]$SshHost = "dd-a6000",
    [int]$Port = 10048,
    [switch]$IncludeGeometry,
    [switch]$IncludeComparators
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$structured = "/home/user/GridPathPlanning-structured-pairs-v1/checkpoints"
$localGeometry = "/home/user/GridPathPlanning-local-occupancy-v1/checkpoints"
$models = @(
    @{
        Name = "set_seed42"
        Remote = "$structured/relational_set_v1/t5_base/verify"
        Local = "relational_set_v1/t5_base/verify"
        CheckpointHash = "dc3ee8ac605759d7a93e758f8432773c068a33236d84b5af0d7c5f5f417ed8a6"
        CompletedHash = "bbc2fc1f9c703ecfcb4cd6bf3add092a6324c88f974d2241924f7bffdf96f104"
        WeightHash = "3eeefd73ebf8e40f831dc32e646b1769a37d5801b4003e69567d995fef203be8"
    },
    @{
        Name = "set_seed43"
        Remote = "$structured/relational_set_multiseed_v1/seed_43/t5_base/verify"
        Local = "relational_set_multiseed_v1/seed_43/t5_base/verify"
        CheckpointHash = "eef8066a90bd8e5c48ec0ec95ef013d851a8804f17ba3e56ed8d5de159f25f94"
        CompletedHash = "40a0eff75b76b5788bd309d42b4f3ae91f2c73abcddac0f3237c6bb8b401ffc6"
        WeightHash = "acb11d93f3e7a5651f04e121f91bd434a0521cd5c1e9e323129addfdd6e615b6"
    },
    @{
        Name = "set_seed44"
        Remote = "$structured/relational_set_multiseed_v1/seed_44/t5_base/verify"
        Local = "relational_set_multiseed_v1/seed_44/t5_base/verify"
        CheckpointHash = "1d601d07dbe6d52a4d7529f3e3b02c158356e3a113feb2d9c06b79955570a493"
        CompletedHash = "17a7e1f3b84d26d53a1e3ba4a0987b1a59669b3036cd787da05eaa1464049620"
        WeightHash = "fe8e720d3eb42252062710c015a7d92f823d748b5b60cdb1051b1773a5a17375"
    }
)
if ($IncludeGeometry) {
    $models += @{
        Name = "geometry_gate_0.7"
        Remote = "$localGeometry/local_occupancy_v1/t5_base/verify"
        Local = "local_occupancy_v1/t5_base/verify"
        CheckpointHash = "44217193efc2124113b9ae9a68e73a1541f7281120c7fcf1d16bff14946364ab"
        CompletedHash = "1854196b6cb8005ae2cbca0fd669cfe731215517fa65364db9e365d88d26d0bd"
        WeightHash = "23d96f5d072e76cab17c147539d2f6586ad6b65a56529f5f2b56023a47067b09"
    }
}
if ($IncludeComparators) {
    $models += @(
        @{
            Name = "original_structured_V"
            Remote = "$structured/improvement_verify_v1/t5_base/verify"
            Local = "improvement_verify_v1/t5_base/verify"
            CheckpointHash = "0328836697200aa04a979b919eaf597149704cdc789566100c07d5dd0d50720d"
            CompletedHash = "c26eed554829e52a1671f8ff8e2ccf632a309adeeddf1fd1a1687370c52728d2"
            WeightHash = "41aae273afb09ca96ca074a87bcfec43eac571a316103bea260b6fd62e990b3a"
        },
        @{
            Name = "v2_generator"
            Remote = "$structured/v2/t5_base/generate"
            Local = "v2/t5_base/generate"
            CheckpointHash = "0759f2c1f1b9bf3a6c2d04e3f9349cd5bef1a083fcbcbdb0ad52ccd6902a219d"
            CompletedHash = "75e2d383cdafa2b4ec70359dcf9c0dcd68ace952b36ff1f9239e16beb8d26981"
            WeightHash = "2e3ffab1370bb0f912944e0a9038b5e852be560be570cfa79711ba90d902ed9e"
            ExpectedStep = 2250
        }
    )
}

function Assert-Hash([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "SHA256 mismatch: $Path (expected $Expected, got $actual)"
    }
}

function Get-VerifiedFile([string]$Remote, [string]$Destination, [string]$Expected) {
    if (Test-Path -LiteralPath $Destination) {
        Assert-Hash $Destination $Expected
        Write-Host "Verified existing: $Destination"
        return
    }
    $part = "$Destination.part"
    & scp -P $Port "${SshHost}:$Remote" $part
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Remote" }
    Assert-Hash $part $Expected
    Move-Item -LiteralPath $part -Destination $Destination
    Write-Host "Downloaded and verified: $Destination"
}

foreach ($model in $models) {
    $target = Join-Path $root "checkpoints/$($model.Local)"
    $latest = Join-Path $target "latest"
    New-Item -ItemType Directory -Path $latest -Force | Out-Null
    Get-VerifiedFile "$($model.Remote)/completed.json" `
        (Join-Path $target "completed.json") $model.CompletedHash
    Get-VerifiedFile "$($model.Remote)/latest/checkpoint.json" `
        (Join-Path $latest "checkpoint.json") $model.CheckpointHash
    $done = Get-Content -LiteralPath (Join-Path $target "completed.json") -Raw |
        ConvertFrom-Json
    $checkpoint = Get-Content -LiteralPath (Join-Path $latest "checkpoint.json") -Raw |
        ConvertFrom-Json
    $step = if ($model.ContainsKey("ExpectedStep")) { $model.ExpectedStep } else { 750 }
    if ($done.status -ne "complete" -or
        $done.fingerprint -ne $checkpoint.binding.fingerprint -or
        $done.state.global_step -ne $step -or
        $checkpoint.state.global_step -ne $step -or
        $checkpoint.files.'model.safetensors' -ne $model.WeightHash) {
        throw "Checkpoint identity mismatch: $($model.Name)"
    }
    foreach ($file in $checkpoint.files.PSObject.Properties) {
        Get-VerifiedFile "$($model.Remote)/latest/$($file.Name)" `
            (Join-Path $latest $file.Name) $file.Value
    }
    Write-Host "Complete: $($model.Name)"
}
