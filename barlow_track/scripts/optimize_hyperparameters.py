# Use the Ax library to optimize hyperparameters
# See: https://ax.dev/tutorials/submitit.html
import argparse
import logging
import os
import time
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from IPython.core.display_functions import display
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import exp_to_df
import yaml  # We are only using this for reading
from ruamel.yaml import YAML
from submitit import AutoExecutor, LocalJob, DebugJob
from itertools import product
from barlow_track.scripts.train_barlow_clusterer import train_barlow_network


def main(hyperparameter_path, run_locally=False, num_parallel_jobs=None, 
         direct_parameter_sweep=False, one_at_a_time_sweep=False, DEBUG=False):
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
        if run_locally and baseline_params['wandb_name'] is None:
            baseline_params['wandb_name'] = 'barlow-hyperparameter-search-local'

    def evaluate(parameters):
        # Add the baseline parameters
        args = SimpleNamespace(**parameters)
        test_losses = train_barlow_network(args)
        result = test_losses['test_loss']
        if np.isnan(result):
            result = 1e6  # More or less infinity
        return {"result": result}

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

    # About 100 epochs per day
    num_days = int(baseline_params['epochs'] / 100) + 1
    executor.update_parameters(timeout_min=65 * 12 * num_days)
    if not run_locally:
        executor.update_parameters(slurm_time=f"{num_days}-00:00:00")
        executor.update_parameters(cpus_per_task=16)
        executor.update_parameters(slurm_mem="128G")
        executor.update_parameters(slurm_job_name="barlow_hyperparameter_search")
        executor.update_parameters(slurm_gres="gpu:1")
        executor.update_parameters(slurm_additional_parameters={"no-requeue": None})  # bash equivalent (no-arg flag): #SBATCH --no-requeue

    if direct_parameter_sweep:
        # Manually define all the trials as all combinations
        for param in parameters:
            all_param_lists = []
            if param['type'] == 'choice':
                assert 'values' in param, "For direct parameter sweep, the parameter must have a list of values"
                # List of lists, which will be combined into a grid
                all_param_lists.append(param['values'])
            else:
                raise ValueError("For direct parameter sweep, all parameters must be of type 'choice'")
        
        # Make a grid of all combinations as a dict of parameter name to value
        all_combinations = list(product(*all_param_lists))
        all_combinations = [{parameters[i]['name']: v for i, v in enumerate(comb)} for comb in all_combinations]
        print(f"Running a direct parameter sweep with {len(all_combinations)} combinations")
        total_budget = len(all_combinations)
    elif one_at_a_time_sweep:
        # Define all trials as a sweep of one parameter at a time
        all_combinations = []
        for i, param in enumerate(parameters):
            if param['type'] == 'choice':
                assert 'values' in param, "For one-at-a-time parameter sweep, the parameter must have a list of values"
                for v in param['values']:
                    all_combinations.append({param['name']: v})
            else:
                raise ValueError(f"For one-at-a-time parameter sweep, all parameters must be of type 'choice'; got {param['type']} for parameter {param['name']}")
        print(f"Running a one-at-a-time parameter sweep with {len(all_combinations)} combinations")
        total_budget = len(all_combinations)
    else:
        total_budget = 5 if DEBUG else 30

    if num_parallel_jobs is None:
        num_parallel_jobs = 1 if (DEBUG or run_locally) else 10
    else:
        num_parallel_jobs = int(num_parallel_jobs)

    jobs = []
    submitted_jobs = 0
    trial_offset = 0

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

                # Display the current and completed trials
                display(exp_to_df(ax_client.experiment))

        # Schedule new jobs if there is availablity
        if direct_parameter_sweep or one_at_a_time_sweep:
            # Get a new trial manually, without using the AxClient's internal logic (it can't do a grid search)
            # Use the submitted_jobs index as the start point of the next batch of trials
            trial_index_to_param = {}
            for i in range(submitted_jobs, submitted_jobs + num_parallel_jobs - len(jobs)):
                if i >= total_budget - 1:
                    break
                trial = ax_client.experiment.new_trial()
                parameters = all_combinations[i]
                trial_index_to_param[trial.index] = parameters
        else:
            trial_index_to_param, _ = ax_client.get_next_trials(
                max_trials=min(num_parallel_jobs - len(jobs), total_budget - submitted_jobs))
        
        for trial_index, parameters in trial_index_to_param.items():
            # Make a new folder in the parent folder
            # Find a unique folder name by incrementing trial_offset if needed
            while True:
                this_folder = os.path.join(experiment_parent_folder, f"trial_{trial_index + trial_offset}")
                if not os.path.exists(this_folder):
                    break
                logging.warning(f"Found folder {this_folder}; assuming these are old trials with the same settings, and using an index offset")
                trial_offset += 1
            
            logging.info(f"Making parameter files for trial {trial_index} in folder {this_folder} with index offset {trial_offset}")
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
        # If you have a large number of jobs, consider adding a sleep statement in the job polling loop as well
        # Update every couple of minutes, because these jobs are very slow (usually multiple hours)
        time.sleep(5*60)
    
    out = ax_client.get_best_parameters()
    if len(out) == 4:
        best_parameters, mean_and_variance, best_trial_index, best_trial_name = out
    elif len(out) == 2:
        # older versions of Ax return only two values
        best_parameters, mean_and_variance = out
        best_trial_index, best_trial_name = None, None
    else:
        best_parameters, mean_and_variance, best_trial_index, best_trial_name = None, None, None, None
        logging.warning(f"Could not unpack best parameters from AxClient.get_best_parameters(); got {out}")
    
    print(f'Best set of parameters: {best_parameters}')
    print(f'Mean objective value: {mean_and_variance}')
    # The covariance is only meaningful when multiple objectives are present.
    # render(ax_client.get_contour_plot())

    # Copy the best parameters and index to a file
    best_params_path = os.path.join(experiment_parent_folder, 'best_parameters.yaml')
    with open(best_params_path, 'w') as f:
        best_parameters['best_trial_index'] = best_trial_index
        best_parameters['best_trial_name'] = best_trial_name
        best_parameters['mean_and_variance'] = mean_and_variance
        YAML().dump(best_parameters, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameter_template_path', '-p', default=None)
    parser.add_argument('--run_locally', action='store_true')
    parser.add_argument('--num_parallel_jobs', default=None)
    parser.add_argument('--direct_parameter_sweep', action='store_true')
    parser.add_argument('--one_at_a_time_sweep', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')

    args = parser.parse_args()
    hyperparameter_template_path = args.hyperparameter_template_path
    run_locally = args.run_locally
    num_parallel_jobs = args.num_parallel_jobs
    direct_parameter_sweep = args.direct_parameter_sweep
    one_at_a_time_sweep = args.one_at_a_time_sweep
    DEBUG = args.DEBUG

    main(hyperparameter_template_path, run_locally, num_parallel_jobs, 
         direct_parameter_sweep, one_at_a_time_sweep, DEBUG=DEBUG)
