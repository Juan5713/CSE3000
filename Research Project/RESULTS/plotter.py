import pandas as pd
import matplotlib.pyplot as plt


def plot_data(file_path, fig_path, steps_count, reachable):
    # Load the CSV file
    df = pd.read_csv(file_path)
    plt.rcParams.update({'font.size': 14})

    # Set up the figure and axis
    plt.figure(figsize=(12, 8))

    # Create the bar plot
    bar_width = 0.4
    positions = range(len(df['Behavioral Policy']))
    positions_shifted = [p + bar_width for p in positions]
    error_config = {'capsize': 5, 'capthick': 2, 'elinewidth': 1.5}

    bars1 = plt.bar(positions, df['IQL Score'], width=bar_width, alpha=.6, label='IQL Score', color='b',
                    yerr=df['IQL Score Std Dev'], error_kw=error_config)
    bars2 = plt.bar(positions_shifted, df['BC Score'], width=bar_width, alpha=.6, label='BC Score', color='r',
                    yerr=df['BC Score Std Dev'], error_kw=error_config)

    # Add an extra plot for the legend entry for the error bars
    plt.errorbar([], [], yerr=1, fmt=' ', color='k', elinewidth=1.5, capsize=5, label='Standard Deviation')

    # Add values on top of each bar
    # for bar in bars1:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 1), ha='center', va='bottom')
    # for bar in bars2:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 1), ha='center', va='bottom')

    # Set the x-ticks and labels
    plt.xlabel('Behavioral Policy', labelpad=20)
    plt.ylabel('Reward over 40 topologies')
    plt.xticks([p + bar_width / 2 for p in positions], df['Behavioral Policy'], rotation=45, ha='right')

    # Set y-axis limits
    plt.ylim(0, 40)

    # Set the title and legend
    reachability = 'reachable' if reachable else 'unreachable'
    plt.title('Reward obtained by IQL and BC on {} test set, {} training steps'.format(reachability, steps_count))
    plt.legend(title='Score Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the rotated x-axis labels and legend
    plt.tight_layout()

    # Save the figure with higher resolution
    plt.savefig(fig_path, dpi=300)


def plot_together():
    fig, axes = plt.subplots(2, 1, figsize=(14, 18), sharey=True)
    plt.rcParams.update({'font.size': 14})
    error_config = {'capsize': 5, 'capthick': 2, 'elinewidth': 1.5}
    bar_width = 0.4
    file_paths = [
        "tuned_reachable/reachable_full_10000_summary.csv",
        "tuned_unreachable/unreachable_full_10000_summary.csv"
    ]
    fig_path = "all_sets_full_10000.png"
    for i in range(2):
        df = pd.read_csv(file_paths[i])
        positions = range(len(df['Behavioral Policy']))
        positions_shifted = [p + bar_width for p in positions]
        bars1 = axes[i].bar(positions, df['IQL Score'], width=bar_width, alpha=.6, label='IQL Score', color='b',
                            yerr=df['IQL Score Std Dev'], error_kw=error_config)
        bars2 = axes[i].bar(positions_shifted, df['BC Score'], width=bar_width, alpha=.6, label='BC Score', color='r',
                            yerr=df['BC Score Std Dev'], error_kw=error_config)
        axes[i].set_xticks([p + bar_width / 2 for p in positions])
        axes[i].set_xticklabels(df['Behavioral Policy'], rotation=45, ha='right')
        axes[i].set_xlabel('Behavioral Policy')
        axes[i].set_ylabel('Reward over 40 topologies')
        axes[i].set_ylim(0, 40)

    plt.title("Reward obtained by IQL and BC on reachable (top) and unreachable (bottom) test sets, 10000 training "
              "steps", x=0.5, y=1.5)
    # Create a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    plt.legend(handles, labels, title='Score Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the rotated x-axis labels and legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure with higher resolution
    plt.savefig(fig_path, dpi=300)


steps = [100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000]

for i in range(len(steps)):
    # plot_data("tuned_reachable_v2/reachable_full_{}_summary.csv".format(steps[i]),
    #           "tuned_reachable_v2/reachable_full_{}.png".format(steps[i]),
    #           steps[i], True)
    # plot_data("tuned_unreachable_v2/unreachable_full_{}_summary.csv".format(steps[i]),
    #           "tuned_unreachable_v2/unreachable_full_{}.png".format(steps[i]),
    #           steps[i], False)
    plot_data("tuned_train_v2/train_full_{}_summary.csv".format(steps[i]),
              "tuned_train_v2/train_full_{}.png".format(steps[i]),
              steps[i], False)

# plot_together()
