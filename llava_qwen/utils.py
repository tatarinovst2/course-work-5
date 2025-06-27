import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import psutil
try:
    import pynvml
except ImportError:
    print("pynvml is not installed. GPU memory monitoring will not be available.")
    pynvml = None
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


def full_memory_report():
    print("=== FULL MEMORY REPORT ===")

    # 1. System-level memory information using psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("\n--- Process Memory Info (psutil) ---")
    print(f"RSS:   {mem_info.rss / 1024 ** 2:.2f} MB")
    print(f"VMS:   {mem_info.vms / 1024 ** 2:.2f} MB")
    # Additional fields: shared, text, lib, data, if available
    if hasattr(mem_info, "shared"):
        print(f"Shared: {mem_info.shared / 1024 ** 2:.2f} MB")
    if hasattr(mem_info, "text"):
        print(f"Text:   {mem_info.text / 1024 ** 2:.2f} MB")
    if hasattr(mem_info, "data"):
        print(f"Data:   {mem_info.data / 1024 ** 2:.2f} MB")

    # 2. Torch memory stats based on your available device
    print("\n--- Torch Memory Stats ---")
    if torch.backends.mps.is_available():
        print("Device: MPS")
        try:
            # torch.mps.memory_stats() returns a dict with lots of info
            mps_stats = torch.mps.current_allocated_memory()
            print(f"Current Allocated Memory: {mps_stats / 1024 ** 2:.2f} MB")
            driver_stats = torch.mps.driver_allocated_memory()
            print(f"Driver Allocated Memory: {driver_stats / 1024 ** 2:.2f} MB")
        except Exception as e:
            print("Error retrieving MPS memory stats:", e)
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
        print("Device: CUDA")
        print(torch.cuda.memory_summary(device, abbreviated=False))
    else:
        print("No GPU device available - running on CPU only.")
        # For CPU, you might use tracemalloc (or leave it to psutil)



def get_optimizer_state_memory(optimizer):
    total_bytes = 0
    for state in optimizer.state_dict()["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                mem = value.numel() * value.element_size()
                total_bytes += mem
                print(f"State key [{key}]: size={value.shape}, bytes={mem}")
    print(f"Total Optimizer State Memory: {total_bytes/1024**2:.2f} MB")
    return total_bytes


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

            driver_stats = torch.mps.driver_allocated_memory()
            print(f"Driver Allocated Memory: {driver_stats / 1024 ** 2:.2f} MB")
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


import numpy as np

import numpy as np


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


def monitor(training_pid, interval=5):
    """
    Periodically checks overall GPU memory usage and prints the report.
    Exits if:
      - The training process (specified by training_pid) ends.
      - NVML (or the CUDA driver) raises an error.

    Args:
        training_pid (int): Process ID of the training job.
        interval (int or float): Time in seconds between memory checks.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as err:
        print("Failed to initialize NVML:", err)
        sys.exit(1)

    print("Started GPU memory monitoring...")

    while True:
        try:
            try:
                os.kill(training_pid, 0)
            except OSError:
                print("Training process has ended. Exiting monitor.")
                break

            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used = mem_info.used
                total = mem_info.total
                ratio = used / total
                print(
                    f"GPU {i}: {used / (1024 ** 2):.2f} MB used / {total / (1024 ** 2):.2f} MB total "
                    f"(Usage Ratio: {ratio:.2f})"
                )
            print("-----")

        except pynvml.NVMLError as nv_err:
            print("NVML error encountered:", nv_err)
            break
        except Exception as e:
            print("Unexpected error:", e)
            break

        time.sleep(interval)

    pynvml.nvmlShutdown()
    print("GPU monitor exiting.")
    sys.exit(0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
