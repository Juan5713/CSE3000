import pandas as pd
import numpy as np


def transform_data(original_file, summary_file):
    # Load the data
    data = pd.read_csv(original_file)

    # Group by Behavioral Policy and Training Steps, then compute mean and std dev
    grouped = data.groupby(['Behavioral Policy', 'Training Steps']).agg(
        IQL_Mean=('IQL Score', 'mean'),
        BC_Mean=('BC Score', 'mean'),
        IQL_Std=('IQL Score', 'std'),
        BC_Std=('BC Score', 'std')
    ).reset_index()

    # Rename columns
    grouped = grouped.rename(columns={
        'Behavioral Policy': 'Behavioral Policy',
        'IQL_Mean': 'IQL Score',
        'BC_Mean': 'BC Score',
        'Training Steps': 'Training Steps',
        'IQL_Std': 'IQL Score Std Dev',
        'BC_Std': 'BC Score Std Dev'
    })

    # Save to new CSV
    grouped.to_csv(summary_file, index=False)


steps = [100, 200, 500]

for i in range(len(steps)):
    # transform_data("tuned_reachable_v2/reachable_full_{}.csv".format(steps[i]),
    #                "tuned_reachable_v2/reachable_full_{}_summary.csv".format(steps[i]))
    # transform_data("tuned_unreachable_v2/unreachable_full_{}.csv".format(steps[i]),
    #                "tuned_unreachable_v2/unreachable_full_{}_summary.csv".format(steps[i]))
    transform_data("tuned_train_v2/train_full_{}.csv".format(steps[i]),
                   "tuned_train_v2/train_full_{}_summary.csv".format(steps[i]))
