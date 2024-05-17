# Use the Ax library to optimize hyperparameters
# See: https://ax.dev/tutorials/submitit.html
import os
import time
from types import SimpleNamespace

from IPython.core.display_functions import display
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import exp_to_df
from ruamel.yaml import YAML
from submitit import AutoExecutor, LocalJob, DebugJob

from barlow_track.scripts.train_barlow_clusterer import train_barlow_network


DEBUG = False  # Set to True to run a small number of epochs for debugging
LOCAL = False  # Set to True to run locally

if DEBUG:
    LOCAL = True

# Full training script

# Set up baseline parameters; load from template yaml file
fname = '/lisc/scratch/neurobiology/zimmer/wbfm/code/barlow_track/barlow_track/barlow_project_template/train_config.yaml'
baseline_params = YAML().load(open(fname))
if DEBUG:
    experiment_parent_folder = '/lisc/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search_debug'
    baseline_params['wandb_name'] = 'barlow-hyperparameter-search-debug'
    baseline_params['num_frames'] = 20
    baseline_params['epochs'] = 2
    baseline_params['print_freq'] = 10
else:
    experiment_parent_folder = '/lisc/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search'
    baseline_params['epochs'] = 30
    if LOCAL:
        baseline_params['wandb_name'] = 'barlow-hyperparameter-search-local'
    else:
        baseline_params['wandb_name'] = 'barlow-hyperparameter-search'


def evaluate(parameters):
    # Add the baseline parameters
    args = SimpleNamespace(**parameters)
    test_losses = train_barlow_network(args)
    return {"result": test_losses['test_loss']}


# Set up the Ax client
ax_client = AxClient(enforce_sequential_optimization=DEBUG)
ax_client.create_experiment(
    name="my_experiment",
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 1e-4], "log_scale": True},
        {"name": "learning_rate_weights", "type": "range", "bounds": [0.1, 10.0], "log_scale": True},
        {"name": "embedding_dim", "type": "range", "bounds": [256, 2048], "log_scale": True, "value_type": "int"},
        # {"name": "lambd_obj", "type": "range", "bounds": [0.0, 5.0]},  # Optimizing this affects the loss function
        # {"name": "train_both_correlations", "type": "choice", "values": [True, False], "value_type": "bool"},
    ],
    objectives={"result": ObjectiveProperties(minimize=True)},
)

# Set up SubmitIt
# Log folder and cluster. Specify cluster='local' or cluster='debug' to run the jobs locally during development.
# When we're are ready for deployment, switch to cluster='slurm'
if LOCAL:
    executor = AutoExecutor(folder="/tmp/submitit_runs", cluster='debug')
else:
    # Can't use /tmp/submitit_runs because the cluster can't access it
    # https://github.com/facebookincubator/submitit/blob/main/docs/tips.md
    executor = AutoExecutor(folder=experiment_parent_folder, cluster='slurm')
executor.update_parameters(timeout_min=60 * 12)
if not LOCAL:
    executor.update_parameters(slurm_time="0-06:00:00")
    executor.update_parameters(cpus_per_task=4)
    executor.update_parameters(slurm_partition="basic,gpu")
    executor.update_parameters(slurm_job_name="barlow_hyperparameter_search")
    executor.update_parameters(slurm_gres="gpu:1")


total_budget = 10 if DEBUG else 100
num_parallel_jobs = 3 if DEBUG else 1

jobs = []
submitted_jobs = 0
# Run until all the jobs have finished and our budget is used up.
while submitted_jobs < total_budget or jobs:
    for job, trial_index in jobs[:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if job.done() or type(job) in [LocalJob, DebugJob]:
            # The log file isn't being produced, so print the stdout instead
            # print(job.stdout())
            # print(job.stderr())
            result = job.result()
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            jobs.remove((job, trial_index))

    # Schedule new jobs if there is availablity
    trial_index_to_param, _ = ax_client.get_next_trials(
        max_trials=min(num_parallel_jobs - len(jobs), total_budget - submitted_jobs))
    for trial_index, parameters in trial_index_to_param.items():
        # Make a new folder in the parent folder
        this_folder = os.path.join(experiment_parent_folder, f"trial_{trial_index}")
        os.makedirs(this_folder, exist_ok=True)
        os.makedirs(os.path.join(this_folder, 'log'), exist_ok=True)
        os.makedirs(os.path.join(this_folder, 'checkpoints'), exist_ok=True)
        parameters['project_dir'] = this_folder
        # Add the baseline parameters, and save in this folder
        parameters = {**baseline_params, **parameters}
        YAML().dump(parameters, open(os.path.join(this_folder, 'train_config.yaml'), 'w'))
        # Actually submit
        job = executor.submit(evaluate, parameters)
        submitted_jobs += 1
        jobs.append((job, trial_index))
        time.sleep(1)

    # Display the current trials.
    display(exp_to_df(ax_client.experiment))

    # Sleep for a bit before checking the jobs again to avoid overloading the cluster.
    # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
    time.sleep(60)

best_parameters, (means, covariances) = ax_client.get_best_parameters()
print(f'Best set of parameters: {best_parameters}')
print(f'Mean objective value: {means}')
# The covariance is only meaningful when multiple objectives are present.

render(ax_client.get_contour_plot())
