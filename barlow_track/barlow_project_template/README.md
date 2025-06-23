# Template for barlow project training

## Folder Structure

The `barlow_project_template` directory is organized as follows:

```
barlow_project_template/
├── log/                                  # Output logs
├── checkpoints/                          # Output checkpoints
├── train_config.yaml                     # Configuration file for experiments
├── README.md                             # Project documentation
└── hyperparameter_search_template.txt    # Template for searching through hyperparameters, if used
```

## Training a network: basic

To start, you need:
1. A working conda environment
2. To clone this code (or to be able to copy this project folder from another location)
3. A dataset in the wbfm structure for training

Copy the 'barlow_project_template' folder to the desired location, modify the config file as desired, and use the script on the cluster:

```
conda activate wbfm
python barlow_track/scripts/train_barlow_clusterer.py -p /path/to/train_config.yaml
```

This will produce output in the log/ and checkpoints/ folders, as well as the following in the main folder:
1. resnet50.pth - the final trained weights
2. args.pickle  - the hyperparameters for training; also needed for reading the network


## Training multiple networks: hyperparameter search

From the barlow_project_template folder, make a parent folder with the hyperparameter_search_template.yaml file, and modify it to set which parameters to vary.
Optionally, set alternative defaults by also including a train_config.yaml file (this is required if you want to use wandb).
Then run:

```
conda activate wbfm
python barlow_track/scripts/optimize_hyperparameters.py -p /path/to/hyperparameter_search_template.yaml
```

By default this runs on the cluster, thus you may want to run it in a tmux session.
The python script will submit batch jobs (up to 100 total), and 

This will create a series of nested folders in the parent folder, which are all copies of this single-project folder.
They are named trial_N, with the actually used hyperparameters saved in their individual train_config.yaml files.
Finally, the best parameters and trial are saved to "best_parameters.yaml" in the parent folder.
