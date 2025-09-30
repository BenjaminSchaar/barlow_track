

import os

from barlow_track.utils.utils_multiproject_analysis import create_projects_and_traces_from_barlow_folder


if __name__ == "__main__":
    model_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/"
    projects_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/"

    trained_model_dirs = ['inverse_augmentation_sweep_zimmer', 'inverse_augmentation_sweep_flavell', 'inverse_augmentation_sweep_samuel']
    
    run_locally = False
    num_parallel_jobs = 10
    DEBUG = False
    all_p = []

    for model_name in trained_model_dirs:
        print(f"Submitting job for {model_name}")
        parts = model_name.split('_')
        lab_name = parts[-1]
        new_location = os.path.join(projects_parent_dir, lab_name, '_'.join(parts[:-1]))
        models_dir = os.path.join(model_parent_dir, model_name)
        
        create_projects_and_traces_from_barlow_folder(new_location, models_dir)

    print("Finished; please check the SLURM queue for running jobs.")
