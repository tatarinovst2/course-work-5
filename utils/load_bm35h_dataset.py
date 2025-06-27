import argparse
import json
import shutil
from pathlib import Path
import random

from datasets import load_dataset
from tqdm import tqdm

from constants import ROOT_DIR
from common import prepare_dataset_folder, resize_image


def create_balanced_splits(dataset, split_ratios=(0.4, 0.2, 0.4), seed=42):
    """
    Split the dataset into train, validation, and test sets with balanced labels.

    Args:
        dataset (Dataset): The original dataset.
        split_ratios (tuple): Ratios for train, validation, and test splits.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with 'train', 'val', and 'test' splits as lists of examples.
    """
    yes_examples = [example for example in dataset if example['label'] == 1]
    no_examples = [example for example in dataset if example['label'] == 0]

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

    return {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }


def process_split_balanced(split_data, split_name: str, images_path: Path, dataset_root: Path,
                           limit: int, include_entries_without_images: bool, question: str) -> list:
    """
    Process a given dataset split into the desired format while ensuring balanced labels.

    The limit is applied per label so that the final processed list contains an equal
    number of examples for label 1 ("Yes") and label 0 ("No"). If the overall limit is odd,
    one extra record will be added for label 1.

    Args:
        split_data (list): The dataset split to process as a list of examples.
        split_name (str): The name of the split (e.g., 'train', 'val', 'test').
        images_path (Path): Path to save images.
        dataset_root (Path): Root path of the dataset.
        limit (int): Total maximum number of processed records to output.
        include_entries_without_images (bool): Whether to include records with missing images.
        question (str): The fixed question prompt to include in each record.

    Returns:
        list: Processed dataset split in the desired format.
    """
    processed_data = []
    label_map = {1: 'Yes', 0: 'No'}

    # Separate the records by label.
    records_by_label = {1: [], 0: []}
    for record in split_data:
        lbl = record.get("label", None)
        if lbl in label_map:
            records_by_label[lbl].append(record)

    # Shuffle the records for each label.
    for lbl in records_by_label:
        random.shuffle(records_by_label[lbl])

    per_label_limit = limit // 2
    remainder = limit % 2

    for label in [1, 0]:
        current_limit = per_label_limit + (remainder if label == 1 else 0)
        count = 0
        for idx, record in enumerate(records_by_label[label]):
            if count >= current_limit:
                break

            sample = {"id": f"{split_name}_{label}_{idx}"}

            if "image" in record and record["image"]:
                image_obj = record["image"]
                image_obj = resize_image(image_obj, max_size=384)
                image_file_path = images_path / f"{split_name}_{label}_{idx}.png"
                try:
                    image_obj.save(image_file_path)
                    sample["image"] = str(image_file_path.relative_to(dataset_root))
                except Exception as e:
                    print(f"Error saving image for record {sample['id']}: {e}")
                    if not include_entries_without_images:
                        continue
                    else:
                        sample["image"] = ""
            else:
                if not include_entries_without_images:
                    continue
                else:
                    sample["image"] = ""

            answer_text = label_map[label]
            prompt = question

            sample["conversations"] = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer_text}
            ]

            processed_data.append(sample)
            count += 1

    # Shuffle the final combined list to mix Yes and No examples.
    random.shuffle(processed_data)
    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the Yes-No-Brain-Tumor dataset by creating balanced train, validation, and test splits."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(ROOT_DIR / "data" / "yes_no_brain_tumor"),
        help="Output directory for the prepared dataset."
    )
    parser.add_argument("--train_limit", type=int,
                        default=1000, help="Max number of records for train split")
    parser.add_argument("--val_limit", type=int,
                        default=500, help="Max number of records for validation split")
    parser.add_argument("--test_limit", type=int,
                        default=1000, help="Max number of records for test split")
    parser.add_argument("--include_entries_without_images",
                        action="store_true", help="Include records even if they are missing images.")
    parser.add_argument("--prompt_variant",
                        type=int, default=4,
                        help="Variant of the question prompt to use (4 for default).")
    args = parser.parse_args()

    dataset_path = Path(args.output_dir)

    dataset_path = dataset_path.with_name(f"{dataset_path.name}_{args.prompt_variant}")

    if dataset_path.exists():
        user_input = input(f"Output directory '{dataset_path}' already exists. Remove it and continue? [y/N]: ")
        if user_input.lower() == "y":
            shutil.rmtree(dataset_path)
        else:
            print("Operation cancelled. Exiting.")
            exit(0)

    prepare_dataset_folder(dataset_path)

    images_path = dataset_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    tmp_path = dataset_path / "tmp"

    print("Loading the Yes-No-Brain-Tumor dataset...")
    ds = load_dataset("youngp5/Yes-No-Brain-Tumor", split='train', cache_dir=tmp_path)
    print("Dataset loaded successfully.")

    print("Creating balanced splits (train, val, test)...")
    balanced_splits = create_balanced_splits(ds)

    dataset_splits = {
        "train": (balanced_splits["train"], args.train_limit),
        "val": (balanced_splits["val"], args.val_limit),
        "test": (balanced_splits["test"], args.test_limit)
    }

    if args.prompt_variant == 0:
        question = "<image>\nQuestion: Are there signs of a tumor in the image? Yes or No?\nAnswer: " # With space at the end
    elif args.prompt_variant == 1:
        question = "<image>\nQuestion: Are there signs of a tumor in the image? Yes or No?\nAnswer:"
    elif args.prompt_variant == 2:
        question = "<image>\nQuestion: Are there signs of a tumor? Yes or No?\nAnswer:"
    elif args.prompt_variant == 3:
        question = "<image>\nQuestion: Are there signs of a tumor in the image? Yes or No?"
    elif args.prompt_variant == 4:
        question = "<image>\nAre there signs of a tumor in the image? Yes or No?\nAnswer:"
    elif args.prompt_variant == 5: # NO NEW LINES
        question = "<image> Are there signs of a tumor in the image? Yes or No? Answer:"
    elif args.prompt_variant == 6:
        question = "<image>\nAre there signs of a tumor in the image? Respond with Yes or No.\nAnswer:"
    else:
        raise ValueError(f"Invalid prompt variant: {args.prompt_variant}. Choose from 0 to 7.")

    for split_name, (split_data, limit) in dataset_splits.items():
        print(f"Processing split '{split_name}' aiming for {limit} records (balanced)...")
        processed_examples = process_split_balanced(
            split_data,
            split_name,
            images_path,
            dataset_path,
            limit,
            args.include_entries_without_images,
            question
        )
        output_file = dataset_path / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_examples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(processed_examples)} examples to {output_file}")

    print("All splits have been processed and saved successfully.")
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()
