import argparse

from dataset_inference import run_inference
from get_accuracy_results import evaluate_results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on the test dataset and then evaluate predictions."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned LLaVA model directory."
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
    parser.add_argument(
        "--starting_line_to_ignore",
        type=str,
        default="",
        help="Will not take this starting line into account when direct matching."
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
    args = parser.parse_args()

    print("=== Running Inference Stage ===")
    run_inference(args.model_path, args.test_data, args.output_file,
                  batch_size=args.batch_size, mezo_checkpoint_path=args.mezo_checkpoint_path,
                  mezo_checkpoint_steps=args.mezo_checkpoint_steps)

    print("\n=== Evaluating Inference Results ===")
    evaluate_results(args.output_file, args.starting_line_to_ignore)


if __name__ == "__main__":
    main()
