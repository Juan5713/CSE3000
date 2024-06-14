import pandas as pd


def sort_by_policy(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Define the custom order for the 'Behavioral Policy' column
    custom_order = [
        "Expert",
        "Mixed Suboptimal",
        "Mixed Random",
        "Epsilon-greedy",
        "Boltzmann Softmax: Tau 0.5",
        "Boltzmann Softmax: Tau 1.5",
        "Random"
    ]

    # Convert the 'Behavioral Policy' column to a categorical type with the custom order
    data['Behavioral Policy'] = pd.Categorical(data['Behavioral Policy'], categories=custom_order, ordered=True)

    # Sort the DataFrame by the 'Behavioral Policy' column
    sorted_data = data.sort_values(by='Behavioral Policy')

    # Save the sorted data to a new CSV file
    output_file_path = file_path
    sorted_data.to_csv(output_file_path, index=False)


steps = [100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000]

for i in range(len(steps)):
    sort_by_policy("tuned_reachable/reachable_full_{}_summary.csv".format(steps[i]))
    sort_by_policy("tuned_unreachable/unreachable_full_{}_summary.csv".format(steps[i]))
