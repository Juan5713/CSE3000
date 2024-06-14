import pandas as pd
import matplotlib.pyplot as plt

# Load the data
csv_file = 'tuned_unreachable/unreachable_full_allsteps_summary.csv'  # replace with your actual file path
df = pd.read_csv(csv_file)
plt.rcParams.update({'font.size': 14})

# Define the behavioral policies
policies = df['Behavioral Policy'].unique()

# Plot for IQL
plt.figure(figsize=(12, 6))
for policy in policies:
    subset = df[df['Behavioral Policy'] == policy]
    training_steps = subset['Training Steps'].to_numpy()
    iql_scores = subset['IQL Score'].to_numpy()
    plt.plot(training_steps, iql_scores, label=policy)
plt.xlabel('Training Steps')
plt.ylabel('Average Reward over 40 topologies')
plt.title('IQL average rewards on unreachable test set over different numbers of training steps')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
plt.grid(True)
plt.xlim(0, 60000)
plt.ylim(0, 40)
plt.savefig('tuned_unreachable/iql_allsteps_unreachable.png', bbox_inches='tight')  # Save the figure

# Plot for BC
plt.figure(figsize=(12, 6))
for policy in policies:
    subset = df[df['Behavioral Policy'] == policy]
    training_steps = subset['Training Steps'].to_numpy()
    bc_scores = subset['BC Score'].to_numpy()
    plt.plot(training_steps, bc_scores, label=policy)
plt.xlabel('Training Steps')
plt.ylabel('Average Reward over 40 topologies')
plt.title('BC average rewards on unreachable test set over different numbers of training steps')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
plt.grid(True)
plt.xlim(0, 60000)
plt.ylim(0, 40)
plt.savefig('tuned_unreachable/bc_allsteps_unreachable.png', bbox_inches='tight')  # Save the figure
