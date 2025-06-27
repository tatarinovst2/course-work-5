import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_trainer_state(file_path: str, max_steps: int = None):
    """
    Reads the trainer_state.json file from file_path and extracts the step and loss values.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], []

    if "log_history" not in data:
        print(f"Warning: 'log_history' not found in {file_path}.")
        return [], []

    steps = []
    losses = []
    for entry in data["log_history"]:
        if "step" in entry and "loss" in entry:
            step = entry["step"]
            if max_steps is not None and step > max_steps:
                continue
            steps.append(step)
            losses.append(entry["loss"])

    return steps, losses


def smooth_data(y: list[float], window_size_ratio: float = 0.02) -> list[float]:
    """
    Smooth the data using a window of size relative to the data length.

    :param y: The data to smooth.
    :param window_size_ratio: The ratio of the window size to the data length (e.g., 0.02 for 2%).
    :return: Smoothed data with the same length as the input.
    """
    if not y:
        return y

    N = len(y)
    window_size = max(3, int(N * window_size_ratio))

    if window_size >= N:
        return y

    if window_size % 2 == 0:
        pad_left = window_size // 2
        pad_right = (window_size // 2) - 1
    else:
        pad_left = pad_right = window_size // 2

    padded = np.pad(y, (pad_left, pad_right), mode='edge')

    smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    return smoothed.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss from one or more trainer_state.json files."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Paths to trainer_state.json files (space separated list)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for each file (should match the number of files)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps to plot (e.g., 1000)"
    )
    parser.add_argument(
        "--smooth_ratio",
        type=float,
        default=0.02,
        help="Smoothing window size ratio relative to the data length (default: 0.02)"
    )

    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        parser.error("The number of files must match the number of labels.")

    plt.figure(figsize=(10, 6))
    for file_path, label in zip(args.files, args.labels):
        steps, losses = plot_trainer_state(file_path, max_steps=args.max_steps)
        if steps:
            losses_smoothed = smooth_data(losses, window_size_ratio=args.smooth_ratio)
            plt.plot(steps, losses_smoothed, label=label)
        else:
            print(f"No valid data found in {file_path}.")

    plt.xlabel("Шаги")
    plt.ylabel("Потери")
    plt.title("График потерь обучения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# Example usage:
# python create_complex_loss_plot.py --files run1/trainer_state.json run2/trainer_state.json --labels Adam ZO --max_steps 1000
