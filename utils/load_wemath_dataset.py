import argparse
import json
from pathlib import Path
import shutil
import random
import re

from datasets import load_dataset

from constants import ROOT_DIR
from common import prepare_dataset_folder, resize_image


def process_split(
        split_data,
        split_name: str,
        images_path: Path,
        dataset_root: Path,
        start_index=0,
        rng: random.Random = None,
):
    """
    Process a given dataset split into 'llava' format.

    Args:
        split_data: The dataset split to process.
        split_name (str): Name of the split (e.g., 'train', 'test', 'val').
        images_path (Path): Path to save images.
        dataset_root (Path): Root path of the dataset.
        start_index (int): Starting index for sample IDs.
        reformat_options (bool): If True, remove letter labels from choices and shuffle them.
        rng (random.Random): Random generator for reproducible shuffling.

    Returns:
        list: Processed data examples in 'llava' format.
    """
    processed_data = []
    for idx, record in enumerate(split_data):
        sample_id = start_index + idx
        sample = {"id": str(sample_id)}

        if "image_path" not in record or not record["image_path"]:
            continue

        image_obj = record["image_path"]
        if image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")

        image_obj = resize_image(image_obj)
        image_file_path = images_path / f"{split_name}_{sample_id}.jpg"
        image_obj.save(image_file_path)
        sample["image"] = str(image_file_path.relative_to(dataset_root))

        # Split the option string on ';' as options are separated by semicolons.
        option_strs = record["option"].split(';')
        options = []
        for opt in option_strs:
            opt_clean = opt.strip()
            # Remove the letter prefix if present (e.g., "A. " or "B) ")
            opt_clean = re.sub(r'^[A-Z][.)]\s*', '', opt_clean)
            options.append(opt_clean)

        # Identify the correct option using the answer letter.
        answer_letter = record["answer"].strip().upper()
        index = ord(answer_letter) - ord('A')
        if 0 <= index < len(options):
            correct_option_text = options[index]
        else:
            correct_option_text = record["answer"]

        if rng:
            rng.shuffle(options)
        else:
            random.shuffle(options)

        options_str = "; ".join(options[:-1]) + " or " + options[-1]
        prompt = (
            f"<image>\n{record['question']}\nOptions: {options_str}\nAnswer:"
        )
        conversation_human = {"from": "human", "value": prompt}
        conversation_gpt = {"from": "gpt", "value": correct_option_text}

        sample["conversations"] = [conversation_human, conversation_gpt]
        processed_data.append(sample)
    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and split the We-Math dataset into fixed sample limits for train, test, and optionally validation splits."
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1000,
        help="Number of training samples (default 1000)"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=200,
        help="Number of validation samples (default 200)"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=500,
        help="Number of testing samples (default 500)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "data" / "we_math"),
        help="Output directory for the prepared dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling and option shuffling."
    )
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

    images_path = dataset_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    tmp_path = dataset_path / "tmp"

    ds = load_dataset("We-Math/We-Math", cache_dir=tmp_path)
    full_dataset = ds["testmini"]

    total_requested = args.train_samples + args.test_samples + args.val_samples
    if len(full_dataset) < total_requested:
        raise ValueError(
            f"Requested total samples ({total_requested}) exceed available samples in testmini ({len(full_dataset)})."
        )

    shuffled_dataset = full_dataset.shuffle(seed=args.seed)

    current_index = 0
    train_set = shuffled_dataset.select(range(current_index, current_index + args.train_samples))
    current_index += args.train_samples

    if args.val_samples > 0:
        val_set = shuffled_dataset.select(range(current_index, current_index + args.val_samples))
        current_index += args.val_samples
    else:
        val_set = None

    test_set = shuffled_dataset.select(range(current_index, current_index + args.test_samples))

    dataset_splits = {"train": train_set, "test": test_set}
    if val_set is not None:
        dataset_splits["val"] = val_set

    rng = random.Random(args.seed)
    for split_name, split_data in dataset_splits.items():
        print(f"Processing {len(split_data)} records for the '{split_name}' split...")
        processed_examples = process_split(
            split_data,
            split_name,
            images_path,
            dataset_path,
            rng=rng,
        )
        output_file = dataset_path / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_examples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(processed_examples)} examples to {output_file}")

    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()
