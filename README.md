## Application of Zeroth-Order Optimization Methods for Fine-Tuning of Multimodal Large Language Models

This repository contains code for the term work "Application of Zeroth-Order Optimization Methods for Fine-Tuning of Multimodal Large Language Models" by Tatarinov Maksim.

The final report is available in the `latex/main.pdf`: [link](latex/main.pdf).

### Steps to reproduce

1. Open the Terminal or Command Prompt in the root directory of this repository.

    ```bash
    cd path/to/project
    ```

2. Activate the virtual environment.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    For Windows:

    ```bash
    python3 -m venv venv
    .\venv\Scripts\activate
    ```

3. Make sure you've installed the requirements.

    You can do it by running

    ```bash
    python -m pip install -r requirements.txt
    ```

    You might have to check the [official PyTorch website](https://pytorch.org/get-started/locally/)
    for the exact `torch` version for your machine.

4. Make sure you've downloaded the datasets.

    You can do it by running

    ```bash
    bash llava_qwen/scripts/utils/load_datasets.sh
    ```

   or its Windows equivalent ending in
   `.ps1` file.

5. Make sure you've downloaded the model.

    You can do it by running

    ```bash
    python llava_qwen/download_and_patch_llava_qwen.py --version <version>
    ```

    where `<version>` is the size of the model you want to download (`0.5b` or `7b`).

6. For the specific configuration
   you can run the following command:

    ```bash
    bash llava_qwen/scripts/run_mezo.sh --model_name_or_path <model_name_or_path> --data_path <data_path> --mm_tunable_parts <mm_tunable_parts> [--learning_rate <learning_rate>] [--per_device_train_batch_size <batch_size>]
    ```

    For `AdamW` fine-tuning, use `llava_qwen/scripts/run_adam.sh` instead.

    ...or their Windows equivalents ending in
   `.ps1` file.

    For example, to run the training with the `llava-onevision-qwen2-0.5b-si` model on `we_math` dataset with
    all multimodal parts tunable and using `MeZO` optimization method, you can run:

    ```bash
    bash llava_qwen/scripts/run_mezo.sh --model_name_or_path ".\models\llava-onevision-qwen2-0.5b-si" --data_path "data/we_math/train.json" --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model"
    ```

7. To evaluate the checkpoints,
   you can run the following command:

    ```bash
    bash llava_qwen/scripts/check_accuracy_for_all_checkpoints.sh --test_data <test.json of dataset>
    ```

   or its Windows equivalent ending in
   `.ps1` file.
