import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the data
csv_file = 'tuned_unreachable_v2/unreachable_full_allsteps_summary.csv'  # replace with your actual file path
df = pd.read_csv(csv_file)
plt.rcParams.update({'font.size': 14})

# Filter the DataFrame to include only the specified policies
policies_to_include = ['Expert', 'Epsilon-greedy', 'Random']
df = df[df['Behavioral Policy'].isin(policies_to_include)]

# Define the behavioral policies
policies = df['Behavioral Policy'].unique()

# Plot for IQL with shaded area for standard deviation
plt.figure(figsize=(12, 6))
for policy in policies:
    subset = df[df['Behavioral Policy'] == policy]
    training_steps = subset['Training Steps'].to_numpy()
    iql_scores = subset['IQL Score'].to_numpy()
    iql_std_devs = subset['IQL Score Std Dev'].to_numpy()  # Assuming you have a column for standard deviation

    # Plotting the line
    plt.plot(training_steps, iql_scores, label=policy)
    # Plotting the shaded area
    plt.fill_between(training_steps, iql_scores - iql_std_devs, iql_scores + iql_std_devs, alpha=0.2)

# Add custom legend entry for the shaded area
shaded_patch = mpatches.Patch(color='gray', alpha=0.2, label='Standard Deviation')
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(shaded_patch)
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

plt.xlabel('Training Steps')
plt.ylabel('Average Reward over 40 topologies')
plt.title('IQL average rewards on unreachable test set over different numbers of training steps')
plt.grid(True)
plt.xlim(0, 60000)
plt.ylim(0, 40)
plt.savefig('tuned_unreachable_v2/iql_allsteps_unreachable.png', bbox_inches='tight')  # Save the figure

# Plot for BC with shaded area for standard deviation
plt.figure(figsize=(12, 6))
for policy in policies:
    subset = df[df['Behavioral Policy'] == policy]
    training_steps = subset['Training Steps'].to_numpy()
    bc_scores = subset['BC Score'].to_numpy()
    bc_std_devs = subset['BC Score Std Dev'].to_numpy()  # Assuming you have a column for standard deviation

    # Plotting the line
    plt.plot(training_steps, bc_scores, label=policy)
    # Plotting the shaded area
    plt.fill_between(training_steps, bc_scores - bc_std_devs, bc_scores + bc_std_devs, alpha=0.2)

# Add custom legend entry for the shaded area
shaded_patch = mpatches.Patch(color='gray', alpha=0.2, label='Standard Deviation')
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(shaded_patch)
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

plt.xlabel('Training Steps')
plt.ylabel('Average Reward over 40 topologies')
plt.title('BC average rewards on unreachable test set over different numbers of training steps')
plt.grid(True)
plt.xlim(0, 60000)
plt.ylim(0, 40)
plt.savefig('tuned_unreachable_v2/bc_allsteps_unreachable.png', bbox_inches='tight')  # Save the figure
