from wbfm.pipeline.traces import full_step_4_make_traces_from_config
from wbfm.utils.projects.project_config_classes import make_project_like
from wbfm.utils.projects.finished_project_data import ProjectData
from barlow_track.utils.track_using_barlow import track_using_barlow_from_config


def copy_and_retrack(original_project, target_directory, model_fname):
    """
    Copy a project and retrack it
    """
    # fname = "/lisc/scratch/neurobiology/zimmer/zihaozhai/WBFM/project/2024-10-02_13-46_SWF1088_2per_worm4-2024-10-02/project_config.yaml"
    # fname = "/lisc/scratch/neurobiology/zimmer/zihaozhai/WBFM/project/2024-10-02_14-56_SWF1088_2per_worm8-2024-10-02/project_config.yaml"
    steps_to_keep = ['preprocessing', 'segmentation', 'training_data']
    # target_directory = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects_future/barlow_challenging_datasets/neuropal'

    # Copy
    target_project_name = make_project_like(original_project, target_directory=target_directory,
                                            steps_to_keep=steps_to_keep)

    # Load and retrack
    project_data = ProjectData.load_final_project_data_from_config(target_project_name)
    # model_fname = '/lisc/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search_neuropal/resnet50.pth'
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
