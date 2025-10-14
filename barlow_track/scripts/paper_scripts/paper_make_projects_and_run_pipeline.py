

import os

from barlow_track.utils.utils_multiproject_analysis import create_projects_and_traces_from_barlow_folder


if __name__ == "__main__":
    model_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/"
    projects_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/"

    trained_model_dirs = [#'untrained_zimmer', 'untrained_flavell', 
                          #'untrained_samuel', 
                          #'untrained_leifer',
                          #'baseline_leifer', 'only_new_loss_leifer', 'only_original_loss_leifer', 'training_data_sweep_leifer', 
                          #'inverse_augmentation_sweep_leifer', 
                          #  'training_data_sweep_leifer', #'only_original_loss_leifer',
                          #  'training_data_sweep_flavell',
                          #  'training_data_sweep_zimmer',
                           'hyperparameter_sweep_leifer'
                          ]
    
    rule_opts = {'zimmer': {'target_rule': "traces"}, 'flavell': {'target_rule': "traces"}, 
                 'leifer': {'target_rule': "barlow_tracking", 'restart_rule': 'alt_barlow_embedding'}}
    use_label_propagation = True
    only_create_projects = False
    DEBUG = False
    
    for model_name in trained_model_dirs:
        print(f"Submitting job for {model_name}")
        parts = model_name.split('_')
        lab_name = parts[-1]
        opts = rule_opts[lab_name]
        new_location = os.path.join(projects_parent_dir, lab_name, '_'.join(parts[:-1]))
        models_dir = os.path.join(model_parent_dir, model_name)
        
        create_projects_and_traces_from_barlow_folder(new_location, models_dir, 
                                                      only_create_projects=only_create_projects, use_label_propagation=use_label_propagation, **opts)

    print("Finished; please check the SLURM queue for running jobs.")
