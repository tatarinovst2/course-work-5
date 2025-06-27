import argparse
import json
from pathlib import Path
import copy

from tqdm import tqdm
from PIL import Image
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from load_mezo_checkpoint import apply_mezo_state_from_path


def get_current_torch_device() -> str:
    """
    Get the current torch device.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_inference(model_path: str, test_data_path: str, output_file: str,
                  mezo_checkpoint_path: str = None, mezo_checkpoint_steps: int = None,
                  debug: bool = False, batch_size: int = 1):
    """
    Run inference on the test dataset and output results into a JSONL file.

    Args:
        model_path (str): Path to the fine-tuned LLaVA model directory.
        test_data_path (str): Path to the test JSON file.
        output_file (str): Output JSONL file where each result is saved in a separate line.
        batch_size (int): Batch size for processing (default: 1).
    """
    device = get_current_torch_device()
    print(f"Using device: {device}")

    print("Loading the pretrained model...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava_qwen",
        attn_implementation="sdpa",
        torch_dtype="bfloat16"
    )
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    if mezo_checkpoint_path:
        print(f"Applying MeZO updates from checkpoint: {mezo_checkpoint_path}")
        model = apply_mezo_state_from_path(
            model,
            mezo_checkpoint_path,
            steps=mezo_checkpoint_steps
        )

    print("Model loaded successfully.\n")

    test_data_path = Path(test_data_path)
    base_dir = test_data_path.parent

    print(f"Loading the test dataset from {test_data_path}...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples.\n")

    results = []

    for record in tqdm(test_data, desc="Inferencing", unit="sample"):
        if "image" in record:
            image_path = Path(record["image"])
            if not image_path.is_absolute():
                image_path = base_dir / image_path

            if not image_path.is_file():
                print(f"Image file not found: {image_path}")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_img.to(dtype=torch.bfloat16, device=device) for _img in image_tensor]
        else:
            image_tensor = None

        conv_template = "qwen_1_5"
        question = f"{record['conversations'][0]['value']}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if debug:
            print(f"=========\nProcessing sample ID: {record['id']}, prompt: {prompt}")

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(device)

        image_sizes = [image.size] if "image" in record else None

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=100
            )

        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if debug:
            print(f"Generated text: {generated_text}")

        ground_truth = record["conversations"][1]["value"].strip()

        result_record = {
            "id": record["id"],
            "predicted": generated_text,
            "ground_truth": ground_truth
        }
        results.append(result_record)

        torch.cuda.empty_cache()

    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_file_path}")
    print("Max GPU memory usage during inference:")
    print(f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference on test dataset and save results in JSONL format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned LLaVA model directory."
    )
    parser.add_argument(
        "--mezo_checkpoint_path",
        type=str,
        default=None,
        help="Path to the MeZO checkpoint file. If provided, will apply MeZO updates to the model."
    )
    parser.add_argument(
        "--mezo_checkpoint_steps",
        type=int,
        default=None,
        help="Number of MeZO update steps to apply. If None, all updates will be applied."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to the test JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the inference results (JSONL file)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)."
    )

    args = parser.parse_args()
    run_inference(args.model_path, args.test_data, args.output_file,
                  batch_size=args.batch_size, mezo_checkpoint_path=args.mezo_checkpoint_path,
                  mezo_checkpoint_steps=args.mezo_checkpoint_steps)


if __name__ == "__main__":
    main()
