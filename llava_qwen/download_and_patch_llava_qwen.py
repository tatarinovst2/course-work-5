import argparse
import json
import os
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download model repository and update config.json"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["0.5b", "7b"],
        default="0.5b",
        help="Specify which version to download: '0.5b' (default) or '7b'."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="models",
        help="Local directory to download the model repositories."
    )
    args = parser.parse_args()

    repo_id = f"lmms-lab/llava-onevision-qwen2-{args.version}-si"

    print("Downloading repository...")
    target_dir = os.path.join(args.local_dir, f"llava-onevision-qwen2-{args.version}-si")
    repo_dir = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=target_dir)
    print(f"Repository downloaded to: {repo_dir}")

    config_path = os.path.join(repo_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"config.json not found in {repo_dir}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    current_model_type = config.get("model_type")
    print(f"Current model_type: {current_model_type}")

    if current_model_type == "llava":
        config["model_type"] = "qwen2"
        print("Updated model_type to 'qwen2'")
    else:
        print("No update necessary; model_type is already updated or set differently.")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("The config.json file has been updated.")


if __name__ == "__main__":
    main()
