import argparse
import json
import shutil
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets


def prepare_dataset_folder(dataset_path: Path):
    """Create the output folder if it doesn't already exist."""
    dataset_path.mkdir(parents=True, exist_ok=True)


def create_balanced_splits(dataset, split_ratios=(0.4, 0.2, 0.4), seed=42):
    """
    Split the dataset (a list of examples) into train, validation, and test splits,
    ensuring that the splits are balanced by the boolean answer field.

    Args:
        dataset (list): List of examples from the combined dataset.
        split_ratios (tuple): Ratios for train, validation, and test splits.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary with keys 'train', 'val', and 'test'.
    """
    yes_examples = [example for example in dataset if example.get("answer") is True]
    no_examples = [example for example in dataset if example.get("answer") is False]

    if len(yes_examples) != len(no_examples):
        print(f"Warning: The dataset is not perfectly balanced. Yes: {len(yes_examples)}, No: {len(no_examples)}")
        min_len = min(len(yes_examples), len(no_examples))
        yes_examples = yes_examples[:min_len]
        no_examples = no_examples[:min_len]

    random.seed(seed)
    random.shuffle(yes_examples)
    random.shuffle(no_examples)

    def split_examples(examples, ratios):
        total = len(examples)
        train_end = int(ratios[0] * total)
        val_end = train_end + int(ratios[1] * total)
        return examples[:train_end], examples[train_end:val_end], examples[val_end:]

    yes_train, yes_val, yes_test = split_examples(yes_examples, split_ratios)
    no_train, no_val, no_test = split_examples(no_examples, split_ratios)

    train_split = yes_train + no_train
    val_split = yes_val + no_val
    test_split = yes_test + no_test

    random.shuffle(train_split)
    random.shuffle(val_split)
    random.shuffle(test_split)

    return {"train": train_split, "val": val_split, "test": test_split}


def process_split_balanced(split_data, split_name: str, limit: int) -> list:
    """
    Process a given split from the BoolQ dataset into the desired JSON format.

    Args:
        split_data (list): List of examples for this split.
        split_name (str): A name for the split (e.g., "train", "val", or "test").
        limit (int): Total maximum number of processed records for the split.

    Returns:
        list: Processed and balanced dataset records.
    """
    processed_data = []

    label_map = {True: "Yes", False: "No"}

    records_by_label = {True: [], False: []}
    for record in split_data:
        ans = record.get("answer")
        if ans in label_map:
            records_by_label[ans].append(record)

    for label in [True, False]:
        current_limit = limit // 2 + (limit % 2 if label is True else 0)
        count = 0
        for idx, record in enumerate(records_by_label[label]):
            if count >= current_limit:
                break

            sample = {"id": f"{split_name}_{label}_{idx}"}

            passage = record.get("passage", "").strip()
            question_text = record.get("question", "").strip()
            prompt = f"{passage.strip()}\nQuestion: {question_text.capitalize().strip()}? Yes or No?\nAnswer:"

            sample["conversations"] = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": label_map[label]}
            ]
            processed_data.append(sample)
            count += 1

    random.shuffle(processed_data)
    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the google/boolq dataset into balanced train, validation, and test splits."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/google_boolq",
        help="Output directory for the prepared dataset."
    )
    parser.add_argument("--train_limit", type=int, default=1000, help="Max number of records for the train split")
    parser.add_argument("--val_limit", type=int, default=500, help="Max number of records for the validation split")
    parser.add_argument("--test_limit", type=int, default=1000, help="Max number of records for the test split")
    args = parser.parse_args()

    dataset_path = Path(args.output_dir)
    if dataset_path.exists():
        user_input = input(f"Output directory '{dataset_path}' already exists. Remove it and continue? [y/N]: ")
        if user_input.lower() == "y":
            shutil.rmtree(dataset_path)
        else:
            print("Operation cancelled. Exiting.")
            exit(0)

    prepare_dataset_folder(dataset_path)

    tmp_path = dataset_path / "tmp"

    print("Loading the google/boolq train split...")
    ds_train = load_dataset("google/boolq", split="train", cache_dir=tmp_path)
    print("Loading the google/boolq validation split...")
    ds_val = load_dataset("google/boolq", split="validation", cache_dir=tmp_path)

    print("Combining train and validation splits...")
    combined_dataset = concatenate_datasets([ds_train, ds_val])
    print(f"Total combined examples: {len(combined_dataset)}")

    combined_examples = list(combined_dataset)

    print("Creating balanced splits from combined dataset...")
    balanced_splits = create_balanced_splits(combined_examples, split_ratios=(0.4, 0.2, 0.4), seed=42)

    dataset_splits = {
        "train": (balanced_splits["train"], args.train_limit),
        "val": (balanced_splits["val"], args.val_limit),
        "test": (balanced_splits["test"], args.test_limit)
    }

    for split_name, (split_data, limit) in dataset_splits.items():
        print(f"Processing split '{split_name}' aiming for {limit} records (balanced)...")
        processed_examples = process_split_balanced(split_data, split_name, limit)

        output_file = dataset_path / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_examples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(processed_examples)} examples to {output_file}")

    print("All splits have been processed and saved successfully.")
    print("Cleaning up temporary files...")
    shutil.rmtree(tmp_path)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
