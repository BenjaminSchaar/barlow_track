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

Copy this folder to the desired location, modify the config file as desired, and use the script:

```
python barlow_track/scripts/train_barlow_clusterer.py -p path/to/train_config.yaml
```

This will produce output in the log/ and checkpoints/ folders, as well as the following in the main folder:
1. resnet50.pth - the final trained weights
2. args.pickle  - the hyperparameters for training; also needed for reading the network

