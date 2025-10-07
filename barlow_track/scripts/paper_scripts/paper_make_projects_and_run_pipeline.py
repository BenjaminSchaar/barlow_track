

import os

from barlow_track.utils.utils_multiproject_analysis import create_projects_and_traces_from_barlow_folder


if __name__ == "__main__":
    model_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/"
    projects_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/"

    trained_model_dirs = [#'untrained_zimmer', 'untrained_flavell', 
                          'untrained_samuel', 'untrained_leifer',]
    base_lab_names = ['zimmer', 'flavell', 'samuel', 'leifer']
    
    DEBUG = False
    
    for model_name in trained_model_dirs:
        print(f"Submitting job for {model_name}")
        parts = model_name.split('_')
        if parts[-1] in base_lab_names:
            lab_name = parts[-1]
            new_location = os.path.join(projects_parent_dir, lab_name, '_'.join(parts[:-1]))
        else:
            print(f"Could not identify lab name; assuming this folder should be applied to all labs")
            new_location = [os.path.join(projects_parent_dir, lab_name, '_'.join(parts[:-1])) for lab_name in base_lab_names]
        models_dir = os.path.join(model_parent_dir, model_name)
        
        create_projects_and_traces_from_barlow_folder(new_location, models_dir, use_label_propagation=True)

    print("Finished; please check the SLURM queue for running jobs.")
