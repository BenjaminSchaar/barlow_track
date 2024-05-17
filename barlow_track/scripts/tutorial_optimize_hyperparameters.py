# Use the Ax library to optimize hyperparameters
# See: https://ax.dev/tutorials/submitit.html

import time

from IPython.core.display_functions import display
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import exp_to_df
from submitit import AutoExecutor, LocalJob, DebugJob

# Dummy function to evaluate
def evaluate(parameters):
    x = parameters["x"]
    y = parameters["y"]
    return {"result": (x - 3)**2 + (y - 4)**2}

# Set up the Ax client
ax_client = AxClient()
ax_client.create_experiment(
    name="my_experiment",
    parameters=[
        {"name": "x", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "y", "type": "range", "bounds": [-10.0, 10.0]},
    ],
    objectives={"result": ObjectiveProperties(minimize=True)},
    parameter_constraints=["x + y <= 2.0"],  # Optional.
)

# Set up SubmitIt
# Log folder and cluster. Specify cluster='local' or cluster='debug' to run the jobs locally during development.
# When we're are ready for deployment, switch to cluster='slurm'
executor = AutoExecutor(folder="/tmp/submitit_runs", cluster='debug')
executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.
executor.update_parameters(cpus_per_task=2)

total_budget = 10
num_parallel_jobs = 3

jobs = []
submitted_jobs = 0
# Run until all the jobs have finished and our budget is used up.
while submitted_jobs < total_budget or jobs:
    for job, trial_index in jobs[:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if job.done() or type(job) in [LocalJob, DebugJob]:
            result = job.result()
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            jobs.remove((job, trial_index))

    # Schedule new jobs if there is availablity
    trial_index_to_param, _ = ax_client.get_next_trials(
        max_trials=min(num_parallel_jobs - len(jobs), total_budget - submitted_jobs))
    for trial_index, parameters in trial_index_to_param.items():
        job = executor.submit(evaluate, parameters)
        submitted_jobs += 1
        jobs.append((job, trial_index))
        time.sleep(1)

    # Display the current trials.
    display(exp_to_df(ax_client.experiment))

    # Sleep for a bit before checking the jobs again to avoid overloading the cluster.
    # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
    time.sleep(10)

best_parameters, (means, covariances) = ax_client.get_best_parameters()
print(f'Best set of parameters: {best_parameters}')
print(f'Mean objective value: {means}')
# The covariance is only meaningful when multiple objectives are present.

render(ax_client.get_contour_plot())
