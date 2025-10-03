import argparse

from barlow_track.utils.utils_multiproject_analysis import create_projects_and_traces_from_barlow_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline: copy → track → extract (all via SBATCH)")
    parser.add_argument("--new-location", required=True, help="Base path for new projects")
    parser.add_argument("--models-dir", required=True, help="Folder containing trial subfolders with models OR a single trial directory when --single-trial is used")
    parser.add_argument("--finished-path", default=None, help="Path to finished project, usually an analyzed ground truth project")
    parser.add_argument("--model-fname", default="resnet50.pth", help="Model filename inside each trial folder")
    parser.add_argument("--use_projection_space", action="store_true", help="Using projection space or final embedding space")
    parser.add_argument("--single-trial", action="store_true", help="Treat --models-dir as a single trial directory instead of a folder of trials")
    parser.add_argument("--use_tracklets", action="store_true", help="Use tracklets instead of clustering to build final tracks")
    parser.add_argument("--use_label_propagation", action="store_true", help="Use alternate clustering method")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (runs only one trial with verbose output)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    create_projects_and_traces_from_barlow_folder(args.new_location, args.models_dir, args.finished_path, args.model_fname,
         args.use_projection_space, args.single_trial, args.use_tracklets, args.use_label_propagation, 
         args.debug)
