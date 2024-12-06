import pathlib
import regex as re
import matplotlib.pyplot as plt
import numpy as np

# Path to the log file
def log_extractor(path: str) -> tuple[list[np.int16], list[np.float16], list[np.float16], list[np.float16]]:
    """
    Reads txt log file and extracts the step, loss, time, and step_time information.
    """
    log_path = pathlib.Path(path)
    # Read the entire file
    with log_path.open("r", encoding="utf-8") as f:
        log = f.read()

    # Define the regular expression
    info_pattern = r"(?<=step:)(?P<step>\d+).+(?<=train_loss:)(?P<loss>\d+.\d+).+(?<=train_time:)(?P<time>\d+).+(?<=step_avg:)(?P<step_time>\d+.\d+)"

    # Find all matches
    matches = re.finditer(info_pattern, log)

    steps, losses, times, step_times = [], [], [], []

    for info in matches:
        step = np.int16(info.group("step"))
        loss = np.float16(info.group("loss"))
        time = np.float16(info.group("time"))
        step_time = np.float16(info.group("step_time"))

        steps.append(step)
        losses.append(loss)
        times.append(time)
        step_times.append(step_time)
    
    return (steps, losses, times, step_times)

def plot_loss_vs_steps(steps: list[np.int16], losses: list[np.float16]) -> None:
    """
    Plots the loss vs. steps.
    """
    # Calculate average loss for each method
    avg_loss = np.mean(np.array(losses))

    # Plot the data
    plt.plot(steps, losses, '-o', color='orange', label='Modified NanoGTP', linewidth=2, markersize=6)

    # Add horizontal lines for average loss
    plt.axhline(y=avg_loss, color='orange', linestyle='dotted', linewidth=1, label='Avg Loss (Modified NanoGPT)')

    # Customize the plot
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs. Steps', fontsize=14)
    plt.legend(fontsize=10)

    # Customize x and y ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Tighten layout and show the plot
    plt.tight_layout()
    plt.show()

step_list, loss_list, times, step_times = log_extractor(path="logs.txt")
plot_loss_vs_steps(step_list, loss_list)
