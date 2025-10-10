

import os
from barlow_track.utils.utils_multiproject_analysis import create_projects_and_traces_from_barlow_folder


def main():

    model_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/"
    projects_parent_dir = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/"

    trained_model_dir = 'baseline'  # Will have lab name appended
    lab_all_gt = {
        'zimmer': [
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format',
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/2022-11-23_worm11_updated_format',
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022-12-10_updated_format'
        ],
        'flavell': [
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/2025_07_01',
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/for_Charlie2-2022-03-16-03',
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/forCharlie_2-2022-03-24-02',
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/forCharlie_2-2022-03-30-01',
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/forCharlie_2-2022-03-30-02'
        ],
        'leifer': [
            '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/projects/leifer_project_gt'
        ]
        }
    
    target_rule_dict = {'zimmer': "traces", 'flavell': 'traces', 'leifer': "alt_barlow_embedding"}
    use_label_propagation = True
    DEBUG = True
    
    # Loop over different labs
    for gt_lab_name, gt_paths in lab_all_gt.items():
        # Loop over the ground truth projects
        for i, gt_path in enumerate(gt_paths):
            # Loop over other labs; run the baseline network for datasets from the same lab but not the first
            for network_lab_name in lab_all_gt.keys():
                if i == 0 and network_lab_name == gt_lab_name:
                    print(f"Skipping already analyzed gt at {gt_path} for the {network_lab_name} lab")
                    continue
                # print(f"Submitting jobs for ground truth dataset from {gt_lab_name} for network trained on {network_lab_name} (gt_path: {gt_path})")

                # Projects should be added to the network being used, not the gt path
                gt_name = os.path.basename(gt_path)
                new_project_dirname = f"{trained_model_dir}_generalization_{gt_lab_name}_{gt_name}"
                new_location = os.path.join(projects_parent_dir, network_lab_name, new_project_dirname)

                # Models come from the network_lab_name, not gt
                models_dir = os.path.join(model_parent_dir, f"{trained_model_dir}_{network_lab_name}")

                # Add yaml to gt_path
                gt_path = os.path.join(gt_path, 'project_config.yaml')

                # Target for leifer gt is different (no traces possible)
                target_rule = target_rule_dict[gt_lab_name]
                
                create_projects_and_traces_from_barlow_folder(new_location, models_dir, gt_path, use_label_propagation=use_label_propagation, target_rule=target_rule,
                                                              DEBUG=DEBUG)
                print(f"Submitting jobs for new projects in {new_location} from models in {models_dir} from gt at {gt_path}")

                if DEBUG:
                    return
                

if __name__ == "__main__":
    main()
    print("Finished; please check the SLURM queue for running jobs.")
