# Use the Ax library to optimize hyperparameters
# See: https://ax.dev/tutorials/submitit.html
import argparse
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from IPython.core.display_functions import display
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import exp_to_df
import yaml  # We are only using this for reading
from ruamel.yaml import YAML
from submitit import AutoExecutor, LocalJob, DebugJob

from barlow_track.scripts.train_barlow_clusterer import train_barlow_network


def main(hyperparameter_path, run_locally=False, DEBUG=False):
    if DEBUG:
        run_locally = True
    if hyperparameter_path is None:
        raise ValueError("Please provide a hyperparameter template path")
    with open(hyperparameter_path, 'r') as f:
        hyperparameter_args = yaml.safe_load(f)

    # Full training script

    # Set up baseline parameters; load from template yaml file
    fname = hyperparameter_args['baseline_params_path']
    if fname is None:
        # Then assume there is a template in the same folder as the hyperparameter_path
        fname = os.path.join(os.path.dirname(hyperparameter_path), 'train_config.yaml')
    logging.info(f"Loading baseline parameters from {fname}")
    with open(fname, 'r') as f:
        baseline_params = yaml.safe_load(f)
    if DEBUG:
        experiment_parent_folder = '/lisc/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search_debug'
        baseline_params['wandb_name'] = 'barlow-hyperparameter-search-debug'
        baseline_params['num_frames'] = 20
        baseline_params['epochs'] = 2
        baseline_params['print_freq'] = 10
    else:
        experiment_parent_folder = Path(hyperparameter_path).parent
        if run_locally:
            baseline_params['wandb_name'] = 'barlow-hyperparameter-search-local'

    def evaluate(parameters):
        # Add the baseline parameters
        args = SimpleNamespace(**parameters)
        test_losses = train_barlow_network(args)
        return {"result": test_losses['test_loss']}

    # Set up the Ax client
    ax_client = AxClient(enforce_sequential_optimization=DEBUG)
    # Read parameters from yaml file
    parameters = list(hyperparameter_args['hyperparameters'])
    ax_client.create_experiment(
        name="my_experiment",
        parameters=parameters,
        objectives={"result": ObjectiveProperties(minimize=True)},
    )

    # Set up SubmitIt
    # Log folder and cluster. Specify cluster='local' or cluster='debug' to run the jobs locally during development.
    # When we're are ready for deployment, switch to cluster='slurm'
    if run_locally:
        executor = AutoExecutor(folder="/tmp/submitit_runs", cluster='debug')
    else:
        # Can't use /tmp/submitit_runs because the cluster can't access it
        # https://github.com/facebookincubator/submitit/blob/main/docs/tips.md
        executor = AutoExecutor(folder=experiment_parent_folder, cluster='slurm')
        logging.info(f"Running experiments in folder: {experiment_parent_folder}")

    # About 50 epochs per day
    num_days = int(baseline_params['epochs'] / 50) + 1
    executor.update_parameters(timeout_min=65 * 12 * num_days)
    if not run_locally:
        executor.update_parameters(slurm_time=f"{num_days}-00:00:00")
        executor.update_parameters(cpus_per_task=16)
        executor.update_parameters(slurm_mem="128G")
        executor.update_parameters(slurm_partition="basic,gpu")
        executor.update_parameters(slurm_job_name="barlow_hyperparameter_search")
        executor.update_parameters(slurm_gres="gpu:1")
    total_budget = 5 if DEBUG else 30
    num_parallel_jobs = 1 if (DEBUG or run_locally) else 10

    jobs = []
    submitted_jobs = 0
    # Run until all the jobs have finished and our budget is used up.
    while submitted_jobs < total_budget or jobs:
        for job, trial_index in jobs[:]:
            # Poll if any jobs completed
            # Local and debug jobs don't run until .result() is called.
            if job.done() or type(job) in [LocalJob, DebugJob]:
                # The log file isn't being produced, so print the stdout instead
                result = job.result()
                ax_client.complete_trial(trial_index=trial_index, raw_data=result)
                jobs.remove((job, trial_index))

                # Display the current trials.
                display(exp_to_df(ax_client.experiment))

        # Schedule new jobs if there is availablity
        trial_index_to_param, _ = ax_client.get_next_trials(
            max_trials=min(num_parallel_jobs - len(jobs), total_budget - submitted_jobs))
        for trial_index, parameters in trial_index_to_param.items():
            # Make a new folder in the parent folder
            this_folder = os.path.join(experiment_parent_folder, f"trial_{trial_index}")
            logging.info(f"Making parameter files for trial {trial_index} in folder {this_folder}")
            os.makedirs(this_folder, exist_ok=False)
            os.makedirs(os.path.join(this_folder, 'log'), exist_ok=False)
            os.makedirs(os.path.join(this_folder, 'checkpoints'), exist_ok=False)
            parameters['project_dir'] = this_folder
            # Add the baseline parameters, and save in this folder
            parameters = {**baseline_params, **parameters}
            YAML().dump(parameters, open(os.path.join(this_folder, 'train_config.yaml'), 'w'))
            # Actually submit
            job = executor.submit(evaluate, parameters)
            submitted_jobs += 1
            jobs.append((job, trial_index))
            time.sleep(1)

        # Sleep for a bit before checking the jobs again to avoid overloading the cluster.
        # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
        # Update every couple of minutes, because these jobs are very slow
        time.sleep(5*60)

    best_parameters, (means, covariances) = ax_client.get_best_parameters()
    print(f'Best set of parameters: {best_parameters}')
    print(f'Mean objective value: {means}')
    # The covariance is only meaningful when multiple objectives are present.

    render(ax_client.get_contour_plot())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameter_template_path', '-p', default=None)
    parser.add_argument('--run_locally', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')

    args = parser.parse_args()
    hyperparameter_template_path = args.hyperparameter_template_path
    run_locally = args.run_locally
    DEBUG = args.DEBUG

    main(hyperparameter_template_path, run_locally, DEBUG)
