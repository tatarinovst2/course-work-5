import json
from datasets import load_dataset
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def create_sample_dataset(num_samples=100):
    """
    Create a sample dataset in the format LLaVA expects from the Flickr30k dataset as an example.

    Args:
        num_samples (int): Number of samples to load and process.

    Returns:
        int: Number of samples successfully processed.
    """
    images_path = ROOT_DIR / "data" / "images"
    train_json_path = ROOT_DIR / "data" / "train.json"
    images_path.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset("nlphuji/flickr30k", split="test", streaming=True, cache_dir="data/cache")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0

    formatted_data = []

    for idx, item in enumerate(dataset):
        if idx >= num_samples:
            break

        try:
            image = item['image']
            image_path = images_path /f"image_{idx}.jpg"
            image.save(image_path)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

        formatted_item = {
            "id": str(idx),
            "image": f"image_{idx}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe this image in detail."
                },
                {
                    "from": "gpt",
                    "value": item['caption'][0]
                }
            ]
        }
        formatted_data.append(formatted_item)

    try:
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2)
    except Exception as e:
        print(f"Error saving formatted data: {e}")
        return len(formatted_data)

    print(f"Successfully created dataset with {len(formatted_data)} samples.")
    return len(formatted_data)


if __name__ == "__main__":
    create_sample_dataset()
