import argparse
import re
import json
from pathlib import Path

from utils import str2bool


def normalize_answer(answer: str, ignore_starting_line: str = "") -> str:
    """
    Normalize the answer string for exact matching.
    """
    answer = answer.lower().strip()
    if answer.endswith(".") or answer.endswith(";"):
        answer = answer[:-1]
    if answer.startswith(ignore_starting_line.lower()):
        answer = answer[len(ignore_starting_line):]
    return answer.strip()


def evaluate_results(results_file: str, starting_line_to_ignore: str,
                     create_results_file: bool = True, check_for_inclusion: bool = False):
    """
    Evaluate the results saved in a JSONL file by checking the 'correct' flags.

    Args:
        results_file (str): Path to the JSONL inference results file.

    Returns:
        tuple: (number of correct predictions, total number of samples, list of detailed results)
    """
    results_path = Path(results_file)
    if not results_path.is_file():
        print(f"Results file not found: {results_path}")
        return 0, 0, []

    total = 0
    correct = 0
    detailed_results = []

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total += 1
            generated_text = record.get("predicted")
            ground_truth = record.get("ground_truth")

            if check_for_inclusion:
                if re.search(r'\b' + re.escape(normalize_answer(ground_truth, starting_line_to_ignore)) + r'\b', normalize_answer(generated_text, starting_line_to_ignore)):
                    is_correct = True
                else:
                    is_correct = False
            else:
                is_correct = normalize_answer(generated_text, starting_line_to_ignore) == normalize_answer(ground_truth, starting_line_to_ignore)
            if is_correct:
                correct += 1
            detailed_results.append(record)

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    print("Evaluation Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    if create_results_file:
        results_output_path = results_path.with_name(results_path.stem + "_accuracy.json")
        with open(results_output_path, "w", encoding="utf-8") as out_f:
            json.dump({
                "total": total,
                "correct": correct,
                "accuracy": accuracy
            }, out_f, ensure_ascii=False, indent=4)
        print(f"Detailed results saved to {results_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inference results from a JSONL file."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the inference results JSONL file."
    )
    parser.add_argument(
        "--starting_line_to_ignore",
        type=str,
        default="",
        help="Will not take this starting line into account when direct matching."
    )
    parser.add_argument(
        "--check_for_inclusion",
        type=str2bool,
        default=False,
        help="If True, will check if the ground truth is included in the generated text."
    )
    parser.add_argument(
        "--create_results_file",
        type=str2bool,
        default=True,
        help="If True, will create a new JSON file with the accuracy results."
    )
    args = parser.parse_args()
    evaluate_results(args.results_file, args.starting_line_to_ignore,
                     create_results_file=args.create_results_file,
                     check_for_inclusion=args.check_for_inclusion)


if __name__ == "__main__":
    main()
