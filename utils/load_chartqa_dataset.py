import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from constants import ROOT_DIR
from common import prepare_dataset_folder, resize_image


def process_split(split_data, split_name: str, images_path: Path | str,
                  dataset_root: Path | str, prompt_variant: int = 0, big_images: bool = False) -> list:
    """
    Process a given dataset split into 'llava' format.
    """
    processed_data = []
    for idx, record in tqdm(enumerate(split_data), total=len(split_data),
                            desc=f"Processing {split_name}"):
        sample_id = idx
        sample = {"id": str(sample_id)}

        if "image" not in record or not record["image"]:
            continue

        image_obj = record["image"]

        if not big_images:
            image_obj = resize_image(image_obj, max_size=384)
        else:
            image_obj = resize_image(image_obj, max_size=768)

        image_file_path = Path(images_path) / f"{split_name}_{sample_id}.png"
        image_obj.save(image_file_path)

        sample["image"] = str(Path(image_file_path).relative_to(dataset_root))

        if prompt_variant == 0:
            conversation_human = {
                "from": "human",
                "value": f"<image>\n{record['query']}\nAnswer:"
            }
        elif prompt_variant == 1:
            conversation_human = {
                "from": "human",
                "value": f"<image>\n{record['query']} Find the answer in the image.\nAnswer:"
            }
        elif prompt_variant == 2:
            conversation_human = {
                "from": "human",
                "value": f"<image>\nFind the answer in the image for: {record['query']}\nAnswer:"
            }
        elif prompt_variant == 3:
            conversation_human = {
                "from": "human",
                "value": f"<image>\nGiven the chart, answer: {record['query']}\nAnswer:"
            }
        elif prompt_variant == 4:
            conversation_human = {
                "from": "human",
                "value": f"<image>\nReview the chart. What is the answer to: \"{record['query']}\"?\nAnswer:"
            }
        else:
            raise Exception("Unsupported prompt variant!")
        conversation_gpt = {
            "from": "gpt",
            "value": record["label"][0]
        }

        sample["conversations"] = [conversation_human, conversation_gpt]
        processed_data.append(sample)

    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the ChartQA dataset by processing its train, val, and test splits."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(ROOT_DIR / "data" / "chartqa"),
        help="Output directory for the prepared dataset."
    )
    parser.add_argument("--train_limit", type=int,
                        default=1000,
                        help="Max number of records for train split")
    parser.add_argument("--val_limit", type=int,
                        default=500,
                        help="Max number of records for val split")
    parser.add_argument("--test_limit", type=int,
                        default=1000,
                        help="Max number of records for test split")
    parser.add_argument("--prompt_variant", type=int, default=2,
                        help="Variant of the user prompt to use.")
    parser.add_argument("--big_images", action="store_true",
                        help="If set, images will be resized to 768x768 pixels instead of 384x384.")
    args = parser.parse_args()

    dataset_path = Path(args.output_dir)
    dataset_path = dataset_path.with_name(f"{dataset_path.name}_{args.prompt_variant}") \
        if not args.big_images else dataset_path.with_name(f"{dataset_path.name}_{args.prompt_variant}_big")

    if dataset_path.exists():
        user_input = input(f"Output directory '{dataset_path}' already exists. "
                           f"Remove it and continue? [y/N]: ")
        if user_input.lower() == "y":
            shutil.rmtree(dataset_path)
        else:
            print("Operation cancelled. Exiting.")
            exit(0)

    prepare_dataset_folder(dataset_path)

    images_path = dataset_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    tmp_path = dataset_path / "tmp"

    print("Loading the ChartQA dataset...")
    ds = load_dataset("HuggingFaceM4/ChartQA", cache_dir=tmp_path)
    print("Dataset loaded successfully.")

    train_data = ds["train"]
    val_data = ds["val"]
    test_data = ds["test"]

    if len(train_data) > args.train_limit:
        train_data = train_data.shuffle(seed=42).select(range(args.train_limit))
    if len(val_data) > args.val_limit:
        val_data = val_data.shuffle(seed=42).select(range(args.val_limit))
    if len(test_data) > args.test_limit:
        test_data = test_data.shuffle(seed=42).select(range(args.test_limit))

    dataset_splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for split_name, split_data in dataset_splits.items():
        print(f"Processing {len(split_data)} records for the '{split_name}' split...")
        processed_examples = process_split(split_data, split_name, images_path, dataset_path,
                                           args.prompt_variant, args.big_images)
        output_file = dataset_path / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_examples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(processed_examples)} examples to {output_file}")

    print("All splits have been processed and saved successfully.")

    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()
