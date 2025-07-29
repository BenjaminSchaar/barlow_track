from wbfm.pipeline.traces import full_step_4_make_traces_from_config
from wbfm.utils.projects.project_config_classes import make_project_like
from wbfm.utils.projects.finished_project_data import ProjectData
from barlow_track.utils.track_using_barlow import track_using_barlow_from_config
from pathlib import Path


def copy_and_retrack(original_project, target_directory, model_fname):
    """
    Copy a project and retrack it
    """
    steps_to_keep = ['preprocessing', 'segmentation', 'training_data']

    # Copy
    try:
        target_project_name = make_project_like(original_project, target_directory=target_directory,
                                                steps_to_keep=steps_to_keep)
    except FileExistsError:
        print(f"Project already exists in {target_directory}. Skipping copy step and continuing.")
        # Generate name in the same way as make_project_like
        project_dir = Path(original_project).parent
        target_project_name = Path(target_directory).joinpath(project_dir.name)

    # Load and retrack
    project_data = ProjectData.load_final_project_data_from_config(target_project_name)
    track_using_barlow_from_config(project_data.project_config, model_fname=model_fname)

    # Make and save full traces
    full_step_4_make_traces_from_config(project_data.project_config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Copy and retrack a project')
    parser.add_argument('--original_project', type=str, help='The original project config file')
    parser.add_argument('--target_directory', type=str, help='The target directory for the new project')
    parser.add_argument('--model_fname', type=str, help='The model file to use for tracking')
    args = parser.parse_args()

    copy_and_retrack(args.original_project, args.target_directory, args.model_fname)
