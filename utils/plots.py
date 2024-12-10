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
    info_pattern = r"(?<=Step\s)(?P<step>\d+):\strain_loss=(?P<train_loss>\d+.\d+),\stime=(?P<step_time>\d+.\d+).+lr=(?P<learning_rate>\d+.\d+[e\-+]+\d+)"

    # Find all matches
    matches = re.finditer(info_pattern, log)

    steps, losses, times, lr_array = [], [], [], []

    for info in matches:
        step = np.int16(info.group("step"))
        loss = np.float16(info.group("train_loss"))
        step_time = np.float16(info.group("step_time"))
        learning_rate = np.float16(info.group("learning_rate"))

        steps.append(step)
        losses.append(loss)
        times.append(step_time)
        lr_array.append(learning_rate)
    
    return (steps, losses, times, lr_array)

def plot_loss_vs_steps(steps: list[np.int16], steps_a, losses: list[np.float16], losses_a) -> None:
    """
    Plots the loss vs. steps.
    """
    # Calculate average loss for each method
    avg_loss = np.mean(np.array(losses))
    avg_loss_a = np.mean(np.array(losses_a))

    # Plot the data
    plt.plot(steps[::128], losses[::128], '-o', color='orange', label='Modified NanoGTP', linewidth=2, markersize=6)
    plt.plot(steps_a[::128], losses_a[::128], '-o', color='black', label='Base NanoGTP', linewidth=2, markersize=6)

    # Add horizontal lines for average loss
    plt.axhline(y=avg_loss, color='orange', linestyle='dotted', linewidth=1, label='Avg Loss (Modified NanoGPT)')
    plt.axhline(y=avg_loss_a, color='black', linestyle='dotted', linewidth=1, label='Avg Loss (Base NanoGPT)')

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

step_list, loss_list, times, step_times = log_extractor(path="diff_losses.log")
step_list_a, loss_list_a, times, step_times = log_extractor(path="base_losses.log")
plot_loss_vs_steps(step_list, step_list_a, loss_list, loss_list_a)