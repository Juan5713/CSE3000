# CSE3000
### An evaluation of the generalization capabilities of Implicit Q-Learning
This is repository containing the code, datasets and results in full for the CSE3000 Research Project, in light of enabling reproducible research. The file structure is as follows:
- `CORL`: is a direct clone from the Clean Offline RL library available at https://github.com/tinkoff-ai/CORL; except for the IQL implementation that has been adapted for discrete control, as well as the logging for IQL, which has been changed to ouput to the result csv files. Additionally, a BC model is trained and evaluated alongside the IQL model. The BC model is from the _d3rlpy_ library available at https://github.com/takuseno/d3rlpy.
- `four_room`: is a direct clone from the repository containing the environment used by the research group available at https://github.com/MWeltevrede/four_room
- `DATASETS`: contains all the datasets used in the evaluation, generated with `store_dataset_gen.py`
- `RESULTS`: contains all the csv files of the results for reachable and unreachable generalization
- `models`: contains a DQN model trained to around 50% rewards, used to build the datasets involving suboptimal actions
