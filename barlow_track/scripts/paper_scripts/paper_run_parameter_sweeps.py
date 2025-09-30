import os
import tqdm


if __name__ == "__main__":
    parent_dir = "/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/"

    sweep_dirs = []#'training_data_sweep_zimmer', 'training_data_sweep_flavell', 'training_data_sweep_samuel']
    augmentation_dirs = ['augmentation_sweep_zimmer', 'augmentation_sweep_flavell', 'augmentation_sweep_samuel']
    
    DEBUG = False
    all_p = []

    for dir in sweep_dirs:
        print(f"Submitting job for {dir}")
        hyperparameter_template_path = os.path.join(parent_dir, dir, 'hyperparameter_search_template.yaml')
        direct_parameter_sweep = True
        one_at_a_time_sweep = False

        p = Process(target=optimize_hyperparameters, 
                    args=(hyperparameter_template_path, run_locally, num_parallel_jobs, direct_parameter_sweep, one_at_a_time_sweep, dir, DEBUG))
        p.start()
        all_p.append(p)
        
    for dir in augmentation_dirs:
        print(f"Submitting job for {dir}")
        hyperparameter_template_path = os.path.join(parent_dir, dir, 'hyperparameter_search_template.yaml')
        direct_parameter_sweep = False
        one_at_a_time_sweep = True
        
        p = Process(target=optimize_hyperparameters, 
                    args=(hyperparameter_template_path, run_locally, num_parallel_jobs, direct_parameter_sweep, one_at_a_time_sweep, dir, DEBUG))
        p.start()
        all_p.append(p)
    
    for p in tqdm(all_p, desc="Joining processes (these may hang)"):
        p.join()
