import os
import subprocess
from pathlib import Path
import re
from ruamel.yaml import YAML

from wbfm.utils.external.utils_yaml import recursive_dict_update
from wbfm.utils.general.utils_filenames import get_location_of_installed_project
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import make_project_like


def create_projects_and_traces_from_barlow_folder(new_location, models_dir, finished_path=None, model_fname='resnet50.pth', use_projection_space=False,
         single_trial=False, use_tracklets=False, use_label_propagation=False, target_rule="traces", restart_rule=None, DEBUG=False):

    if isinstance(new_location, list):
        for loc in new_location:
            create_projects_and_traces_from_barlow_folder(loc, models_dir, finished_path, model_fname, use_projection_space,
                                                         single_trial, use_tracklets, use_label_propagation, DEBUG)
        return

    wbfm_home = get_location_of_installed_project()
    models_dir = Path(models_dir)
    use_projection_space = use_projection_space

    if finished_path is None:
        # Then load it from the train_config.yaml file in the main model_dir
        fname = os.path.join(models_dir, 'train_config.yaml')
        with open(fname, 'r') as f:
            training_config = YAML().load(f)
        project_path = training_config['project_path']
    else:
        project_path = finished_path

    if single_trial:
        # Just one trial, directly from models_dir
        trial_dirs = [models_dir]
    else:
        # Collect all trial_* subdirectories
        trial_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and re.match(r"trial_\d+", d.name)])

    if not trial_dirs:
        print(f"No trial folders found in {models_dir} matching pattern 'trial_#'")
        return

    if DEBUG and not single_trial:
        print(f"[DEBUG] Found {len(trial_dirs)} trial folders")
        print(f"[DEBUG] Will only process the first trial: {trial_dirs[0].name}")
        trial_dirs = trial_dirs[:1]

    #########################################################################################
    # Create projects and update the config file to target the proper barlow model
    #########################################################################################
    for trial_dir in trial_dirs:
        trial_name = trial_dir.name
        # try:
        barlow_model_path = Path(trial_dir) / Path(model_fname)
        print(f"Starting pipeline for {trial_name}")
        if DEBUG:
            print(f"[DEBUG] Model path: {barlow_model_path}")

        try:
            new_project_name = make_project_like(
                project_path=project_path, 
                target_directory=new_location, 
                target_suffix=trial_name,
                steps_to_keep=['preprocessing', 'segmentation', 'nwb'],
                verbose=3 if DEBUG else 0
            )
        except FileExistsError:
            print(f"Project already exists in folder {new_location} with for {trial_name}; skipping")
            continue
        if not barlow_model_path.is_file():
            print(f"Warning: Model file not found: {barlow_model_path} - skipping {trial_name}")
            continue

        # Two options: use tracklets or direct segmentation
        project_data = ProjectData.load_final_project_data(new_project_name, verbose=0)
        project_config = project_data.project_config
        if use_tracklets:
            tracklet_config = project_config.get_training_config()
            config_updates = dict(
                tracker_params=dict(
                    use_barlow_network=True,
                    encoder_opt=dict(
                        network_path=str(barlow_model_path),
                        use_projection_space=use_projection_space
                    )
                ),
                pairwise_matching_params=dict(add_affine_to_candidates=False)
            )
            recursive_dict_update(tracklet_config.config, config_updates)
            tracklet_config.update_self_on_disk()
            print("New config settings: ", tracklet_config)
        else:
            snakemake_config = project_config.get_snakemake_config()
            config_updates = dict(use_barlow_tracker=True, barlow_model_path=str(barlow_model_path))
            snakemake_config.config.update(config_updates)
            snakemake_config.update_self_on_disk()
            if use_label_propagation:
                # Also update the tracking config file
                tracking_config = project_config.get_tracking_config()
                config_updates = dict(barlow_tracker=dict(tracking_mode='label_propagation'))
                recursive_dict_update(tracking_config.config, config_updates)
                tracking_config.update_self_on_disk()

    #########################################################################################
    # Actually submit jobs for full pipeline
    #########################################################################################
    # Note that the script is already recursive

    CMD = ["bash", os.path.join(wbfm_home, 'wbfm', 'scripts', 'cluster', 'run_all_projects_in_parent_folder.sh')]
    CMD.extend(["-t", new_location,  "-s" , target_rule])
    if restart_rule is not None:
        CMD.extend(["-R", restart_rule])
    if DEBUG:
        # Dryrun
        CMD.append("-n")
    subprocess.call(CMD)

    print(f"All jobs for {len(trial_dirs)} trials in folder {new_location} submitted successfully.")
