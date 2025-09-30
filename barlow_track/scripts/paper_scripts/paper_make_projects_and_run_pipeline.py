

import os

import tqdm
from barlow_track.scripts.multiproject_scripts import create_projects_and_traces_from_barlow_folder
from barlow_track.scripts.optimize_hyperparameters import optimize_hyperparameters
from multiprocessing import Process


if __name__ == "__main__":
    model_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/"
    projects_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/"

    trained_model_dirs = ['inverse_augmentation_sweep_zimmer', 'inverse_augmentation_sweep_flavell', 'inverse_augmentation_sweep_samuel']
    
    run_locally = False
    num_parallel_jobs = 10
    DEBUG = False
    all_p = []

    for models_dir in trained_model_dirs:
        print(f"Submitting job for {models_dir}")
        parts = models_dir.split('_')
        lab_name = parts[-1]
        new_location = os.path.join(projects_parent_dir, lab_name, '_'.join(parts[:-1]))
        
        create_projects_and_traces_from_barlow_folder(new_location, models_dir)

    print("Finished; please check the SLURM queue for running jobs.")
