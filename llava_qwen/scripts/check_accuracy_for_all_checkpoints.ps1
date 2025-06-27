[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [string]$checkpointsDir = ".\checkpoints\llava-qwen-finetune",

    [Parameter(Mandatory=$false)]
    [string]$scriptPath = "llava_qwen\check_accuracy.py",

    [Parameter(Mandatory=$false)]
    [string]$testData = ".\data\we_math\test.json",

    [Parameter(Mandatory=$false)]
    [string]$startingLine = "The correct answer is"
)

$testDataDirectory = Split-Path -Parent $testData
$testDataDirName = Split-Path $testDataDirectory -Leaf

Get-ChildItem $checkpointsDir -Directory | ForEach-Object {
    $checkpointFolder = $_.Name
    $checkpointNumber = $checkpointFolder -replace "^checkpoint-", ""

    $modelPath = Join-Path $checkpointsDir $checkpointFolder
    $outputFile = ".\$testDataDirName" + "_steps_$checkpointNumber.json"

    Write-Host "Processing checkpoint: $checkpointFolder"
    Write-Host "Using model path: $modelPath"
    Write-Host "Output will be saved to: $outputFile"

    $command = "python $scriptPath --model_path $modelPath --test_data $testData --output_file $outputFile --starting_line_to_ignore `"$startingLine`""
    Write-Host "Running command: $command"

    python $scriptPath --model_path $modelPath --test_data $testData --output_file $outputFile --starting_line_to_ignore "$startingLine"

    Write-Host "Finished processing checkpoint: $checkpointFolder`n"
}
