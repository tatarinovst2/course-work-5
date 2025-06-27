import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).parent.parent


def parse_path(path: str | Path) -> Path:
    """
    Ensure that the path is absolute and is in a pathlib.Path format.

    :param path: The path to parse.
    :return: The parsed path.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def smooth_data(y: list[float], window_size_ratio: float = 0.02) -> list[float]:
    """
    Smooth the data using a window of size relative to the data length.

    Instead of zero-padding the edges, this function pads the data by duplicating
    the border values. This ensures that the smoothed output has the same length as
    the input.

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

    # Calculate padding sizes
    if window_size % 2 == 0:
        pad_left = window_size // 2
        pad_right = (window_size // 2) - 1
    else:
        pad_left = pad_right = window_size // 2

    # Pad the original data by duplicating the edge values
    padded = np.pad(y, (pad_left, pad_right), mode='edge')
    # Use 'valid' mode so that the output length is len(padded) - window_size + 1, which equals N
    smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    return smoothed.tolist()


def plot_graphs_based_on_log_history(log_history: list[dict], output_dir: str | Path,
                                     metrics: list[str]) -> None:
    """
    Plot the graphs based on the log_history.

    :param log_history: The list of all logs from the Trainer.
    :param output_dir: The directory in which the plots will be created.
    :param metrics: The metrics which to create apart from training and test loss.
    """
    parsed_output_directory = parse_path(output_dir)

    plot_training_and_test_loss(log_history, parsed_output_directory / "loss-plot-epoch.png",
                                plot_epochs=True)
    plot_training_and_test_loss(log_history, parsed_output_directory / "loss-plot-step.png",
                                plot_epochs=False)

    for metric in metrics:
        plot_metric(metric, log_history,
                    parsed_output_directory / f"{metric}-plot-epoch.png", plot_epochs=True)
        plot_metric(metric, log_history,
                    parsed_output_directory / f"{metric}-plot-step.png", plot_epochs=False)


def plot_metric(metric: str, log_history: list[dict], output_path: str | Path,
                plot_epochs: bool = True, window_size_ratio: float = 0.01,
                applying_smoothing_threshold: int = 100) -> None:
    """
    Plot the metric using information from the log history.

    :param metric: The metric to plot (e.g. "rouge-1").
    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
    :param window_size_ratio: The ratio of window size relative to data length for smoothing.
    """
    metric_values = []
    steps = []
    epochs = []

    for entry in log_history:
        if metric.strip() in entry:
            metric_values.append(entry[metric])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])

    if len(metric_values) > applying_smoothing_threshold:
        metric_values = smooth_data(metric_values, window_size_ratio=window_size_ratio)
        if plot_epochs:
            epochs = smooth_data(epochs, window_size_ratio=window_size_ratio)
        else:
            steps = smooth_data(steps, window_size_ratio=window_size_ratio)

    plt.figure(figsize=(8, 4))

    if plot_epochs:
        plt.plot(epochs, metric_values, label=metric, marker='o', linestyle='-', color='0.5')
        plt.title(f"{metric} на протяжении эпох")
        plt.xlabel('Эпохи')
    else:
        plt.plot(steps, metric_values, label=metric, marker='o', linestyle='-', color='0.5')
        plt.title(f"{metric} на протяжении шагов")
        plt.xlabel('Шаги')

    plt.ylabel(metric)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_training_and_test_loss(log_history: list[dict], output_path: str | Path,
                                plot_epochs: bool = True, window_size_ratio: float = 0.01,
                                applying_smoothing_threshold: int = 100) -> None:
    """
    Plot the training and test loss using information from the log history.

    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
    :param window_size_ratio: The ratio of window size relative to data length for smoothing.
    :raises ValueError: If the train losses and test losses have different lengths.
    """
    train_losses = []
    test_losses = []
    steps = []
    epochs = []

    for entry in log_history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])
        if 'eval_loss' in entry:
            test_losses.append(entry['eval_loss'])

    if len(train_losses) != len(test_losses) and test_losses:
        print(f"Train losses: {train_losses}, test losses: {test_losses}, "
              f"steps: {steps}, epochs: {epochs}")
        raise ValueError("Train losses and test losses have different lengths")

    if len(train_losses) > applying_smoothing_threshold:
        train_losses = smooth_data(train_losses, window_size_ratio=window_size_ratio)
        if plot_epochs:
            epochs = smooth_data(epochs, window_size_ratio=window_size_ratio)
        else:
            steps = smooth_data(steps, window_size_ratio=window_size_ratio)

        if test_losses:
            test_losses = smooth_data(test_losses, window_size_ratio=window_size_ratio)

    plt.figure(figsize=(8, 4))

    if plot_epochs:
        plt.plot(epochs, train_losses, label='Потери на обучающем наборе данных',
                 linestyle='-', color='0.4')
        if test_losses:
            plt.plot(epochs, test_losses, label='Потери на валидационном наборе данных',
                     linestyle='-', color='0.8')
        plt.title('Потери на протяжении эпох обучения')
        plt.xlabel('Эпоха')
    else:
        plt.plot(steps, train_losses, label='Потери на обучающем наборе данных',
                 linestyle='-', color='0.4')
        if test_losses:
            plt.plot(steps, test_losses, label='Потери на валидационном наборе данных',
                     linestyle='-', color='0.8')
        plt.title('Потери на протяжении шагов обучения')
        plt.xlabel('Шаги')

    plt.ylabel('Потери')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(parse_path(output_path))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
