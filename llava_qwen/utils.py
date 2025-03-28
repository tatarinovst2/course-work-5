from pathlib import Path

import matplotlib.pyplot as plt
import torch

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


def report_memory(stage: str = ""):
    """
    Reports the current memory usage for CUDA and MPS devices, along with process-level memory.

    Args:
        stage (str): A descriptive label for the current stage of execution (e.g., "Before Training Step").
                     This helps in contextualizing the memory report.
    """
    print(f"\n=== Memory Report: {stage} ===")

    # Checking for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        print("CUDA is available. Reporting CUDA device memory usage:")
        device_count = torch.cuda.device_count()
        for device in range(device_count):
            device_name = torch.cuda.get_device_name(device)
            allocated = torch.cuda.memory_allocated(device) / 1e6  # Convert bytes to MB
            reserved = torch.cuda.memory_reserved(device) / 1e6  # Convert bytes to MB
            max_allocated = torch.cuda.max_memory_allocated(device) / 1e6
            max_reserved = torch.cuda.max_memory_reserved(device) / 1e6

            print(f"\nDevice {device}: {device_name}")
            print(f"  Allocated Memory: {allocated:.2f} MB")
            print(f"  Reserved Memory: {reserved:.2f} MB")
            print(f"  Max Allocated Memory: {max_allocated:.2f} MB")
            print(f"  Max Reserved Memory: {max_reserved:.2f} MB")

    # Checking for MPS (Apple Silicon)
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        print("MPS is available. Reporting MPS device memory usage:")
        try:
            allocated = torch.mps.current_allocated_memory() / 1e6  # Convert bytes to MB
            # torch.mps.max_memory_allocated() requires tracking enabled
            if hasattr(torch.mps, 'max_memory_allocated'):
                max_allocated = torch.mps.max_memory_allocated() / 1e6
                print(f"  Current Allocated Memory: {allocated:.2f} MB")
                print(f"  Max Allocated Memory: {max_allocated:.2f} MB")
            else:
                # If max_memory_allocated is not available
                print(f"  Current Allocated Memory: {allocated:.2f} MB")
                print("  Max Allocated Memory: Not Available")
        except AttributeError:
            print("  Detailed MPS memory metrics are not available in this PyTorch version.")

    else:
        print("No CUDA or MPS device is available.")
        # Optionally, still report process memory usage
        try:
            import os
            import psutil

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss = mem_info.rss / 1e6  # Resident Set Size in MB
            vms = mem_info.vms / 1e6  # Virtual Memory Size in MB

            print(f"\nProcess Memory Usage:")
            print(f"  RSS (Resident Set Size): {rss:.2f} MB")
            print(f"  VMS (Virtual Memory Size): {vms:.2f} MB")
        except ImportError:
            print(
                "  `psutil` is not installed. To install it, run `pip install psutil` for detailed process memory info.")

    print("=== End of Memory Report ===\n")


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
                plot_epochs: bool = True) -> None:
    """
    Plot the metric using information from the log history.

    :param metric: The metric to plot (e.g. "rouge-1").
    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
    """
    metric_values = []
    steps = []
    epochs = []

    for entry in log_history:
        if metric.strip() in entry:
            metric_values.append(entry[metric])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])

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
                                plot_epochs: bool = True) -> None:
    """
    Plot the training and test loss using information from the log history.

    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
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
