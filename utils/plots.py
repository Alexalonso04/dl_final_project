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
        step_time = np.float32(info.group("step_time"))
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
    # Plot the data
    plt.plot(steps[::128], losses[::128], '-o', color='orange', label='Differential Transformer', linewidth=2, markersize=6)
    plt.plot(steps_a[::128], losses_a[::128], '-o', color='black', label='Base Model', linewidth=2, markersize=6)

    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs. Steps', fontsize=14)
    plt.legend(fontsize=10)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig('report/images/loss_vs_steps.png', dpi=300)
    plt.show()

def plot_time_vs_steps(steps: list[np.int16], steps_a, times: list[np.float16], times_a) -> None:
    """
    Plots the time vs. steps.
    """

    # Plot the data
    plt.plot(steps[::128], times[::128], '-o', color='orange', label='Differential Transformer', linewidth=2, markersize=6)
    plt.plot(steps_a[::128], times_a[::128], '-o', color='black', label='Base Model', linewidth=2, markersize=6)

    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Time vs. Steps', fontsize=14)
    plt.legend(fontsize=10)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Tighten layout and show the plot
    plt.tight_layout()
    plt.savefig('report/images/time_vs_steps.png', dpi=300)
    plt.show()

def time_bar_plot(total_time_a, total_time_b):

    fig, ax = plt.subplots()

    models = ['Base', 'Differential']
    training_time = [total_time_b/60000/60, total_time_a/60000/60]
    bar_colors = ['tab:orange', 'tab:blue']

    ax.bar(models, training_time, color=bar_colors, width=0.3)

    # # Add horizontal lines
    # ax.axhline(y=training_time[0], color='tab:orange', linestyle='--')
    ax.axhline(y=training_time[1], xmin=0.04, xmax=0.95, color='tab:blue', linestyle='--')

    # Add arrow annotation
    ax.annotate('', xy=(0.03, training_time[0]), xytext=(0.03, training_time[1]),
                arrowprops=dict(arrowstyle='<->', color='black'))

    # Add text annotation for the difference
    diff = abs(training_time[0] - training_time[1])
    ax.text(0.05, (training_time[0] + training_time[1]) / 2, f'{diff:.2f} hrs', 
            va='center', ha='left', color='black', fontsize=10)

    ax.set_xlabel('Models')
    ax.set_ylabel('Time (hrs)')
    ax.set_title('Total Training Time')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('report/images/training_times.png', dpi=300)
    plt.show()
    
step_list, loss_list, times_a, lr = log_extractor(path="diff_losses.log")
step_list_a, loss_list_a, times_b, lr_a = log_extractor(path="base_losses.log")

# plot_loss_vs_steps(step_list, step_list_a, loss_list, loss_list_a)
# plot_time_vs_steps(step_list, step_list_a, times, times_a)

time_bar_plot(np.sum(times_a), np.sum(times_b))
