#!/bin/bash
# Usage example:
# ./run_checkpoints.sh --checkpoints_dir "./checkpoints/llava-qwen-finetune" --script_path "llava_qwen/check_accuracy.py" --test_data "./data/we_math/test.json" --starting_line "The correct answer is"

checkpointsDir="./checkpoints/llava-qwen-finetune"
scriptPath="llava_qwen/check_accuracy.py"
testData="./data/we_math/test.json"
startingLine="The correct answer is"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoints_dir) checkpointsDir="$2"; shift ;;
        --script_path) scriptPath="$2"; shift ;;
        --test_data) testData="$2"; shift ;;
        --starting_line) startingLine="$2"; shift ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

testDataDirectory=$(dirname "$testData")
testDataDirName=$(basename "$testDataDirectory")

for checkpointDir in "$checkpointsDir"/*; do
    if [ -d "$checkpointDir" ]; then
        checkpointFolder=$(basename "$checkpointDir")
        checkpointNumber=${checkpointFolder#"checkpoint-"}

        modelPath="$checkpointDir"
        outputFile="./${testDataDirName}_steps_${checkpointNumber}.json"

        echo "Processing checkpoint: $checkpointFolder"
        echo "Using model path: $modelPath"
        echo "Output will be saved to: $outputFile"

        command="python $scriptPath --model_path \"$modelPath\" --test_data \"$testData\" --output_file \"$outputFile\" --starting_line_to_ignore \"$startingLine\""
        echo "Running command: $command"

        python "$scriptPath" --model_path "$modelPath" --test_data "$testData" --output_file "$outputFile" --starting_line_to_ignore "$startingLine"

        echo "Finished processing checkpoint: $checkpointFolder"
        echo ""
    fi
done
