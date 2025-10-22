import argparse
import json
import logging
import os
import re
from typing import Union
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
import yaml


def pad_with_nan_rows(df: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """
    Pads the given DataFrame with NaN rows until it reaches the specified target length.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to pad.
    target_length : int
        The desired number of rows.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with NaN rows appended if it was shorter than target_length.
    """
    if len(df) < target_length:
        missing_rows = target_length - len(df)
        new_index = range(int(df.index.max()) + 1, int(df.index.max()) + 1 + missing_rows)
        nan_rows = pd.DataFrame(np.nan, index=new_index, columns=df.columns)
        return pd.concat([df, nan_rows])
    return df


def calculate_accuracy(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Calculate overall, per-neuron, and per-timepoint accuracy,
    along with normalized misses and mismatches.

    Parameters
    ----------
    df_gt : pd.DataFrame
        Ground truth DataFrame (rows: timepoints, columns: neurons).
    df_pred : pd.DataFrame
        Prediction DataFrame (same shape and labeling as df_gt).

    Returns
    -------
    dict
        Dictionary with overall accuracy, and normalized error metrics.
    """
    # Align predicted DataFrame to all ground truth columns and index
    df_pred = df_pred.reindex(index=df_gt.index, columns=df_gt.columns)

    # Validity checks
    gt_valid = ~df_gt.isna()
    pred_nan = df_pred.isna()
    pred_valid = ~pred_nan
    correct = df_gt == df_pred
    gt_valid_and_correct = (gt_valid & correct)

    # Error types
    misses = gt_valid & pred_nan
    mismatches = gt_valid & pred_valid & (~correct)

    # Totals
    total_misses = misses.sum().sum()
    total_mismatches = mismatches.sum().sum()
    total_gt_detections = gt_valid.sum().sum()

    accuracy = 1 - (total_misses + total_mismatches) / total_gt_detections

    # Per-neuron (column-wise)
    gt_valid_per_neuron = gt_valid.sum(axis=0)
    correct_per_neuron = gt_valid_and_correct.sum(axis=0)
    accuracy_per_neuron = correct_per_neuron / gt_valid_per_neuron

    misses_per_neuron_norm = misses.sum(axis=0) / gt_valid_per_neuron
    mismatches_per_neuron_norm = mismatches.sum(axis=0) / gt_valid_per_neuron

    # Per-timepoint (row-wise)
    gt_valid_per_time = gt_valid.sum(axis=1)
    correct_per_time = gt_valid_and_correct.sum(axis=1)
    accuracy_per_timepoint = correct_per_time / gt_valid_per_time

    misses_per_timepoint_norm = misses.sum(axis=1) / gt_valid_per_time
    mismatches_per_timepoint_norm = mismatches.sum(axis=1) / gt_valid_per_time

    return {
        "misses": int(total_misses),
        "mismatches": int(total_mismatches),
        "total_ground_truth": int(total_gt_detections),
        "accuracy": accuracy,

        "accuracy_per_neuron": accuracy_per_neuron,
        "accuracy_per_timepoint": accuracy_per_timepoint,

        "misses_per_neuron_norm": misses_per_neuron_norm,
        "misses_per_timepoint_norm": misses_per_timepoint_norm,
        "mismatches_per_neuron_norm": mismatches_per_neuron_norm,
        "mismatches_per_timepoint_norm": mismatches_per_timepoint_norm,

        "misses": misses,
        "mismatches": mismatches,
        "gt_valid": gt_valid,
        "gt_valid_and_correct": gt_valid_and_correct
    }


def process_trial(trial: int, df_gt: pd.DataFrame, res_file: Union[str, pd.DataFrame], column_name='raw_segmentation_id') -> dict:
    """
    Process a single trial: load results, match columns, pad rows, and compute accuracy.

    Parameters
    ----------
    trial : int
        The trial number being processed.
    df_gt : pd.DataFrame
        Ground truth DataFrame.
    res_file : str
        Path to the trial's result configuration file.

    Returns
    -------
    dict
        Dictionary containing trial number and accuracy stats.
    """
    try:
        if isinstance(res_file, str):
            # Load result data
            project_data_res = ProjectData.load_final_project_data(res_file, verbose=0)
            df_res = project_data_res.final_tracks
        elif isinstance(res_file, pd.DataFrame):
            df_res = res_file
        else:
            raise TypeError(res_file)
        if df_res is None:
            print(f"{trial}: No final tracks found in {res_file}")
            return {"trial": trial, "error": "No final tracks found"}

        # Match lengths
        max_len = max(len(df_res), len(df_gt))
        df_res = pad_with_nan_rows(df_res, max_len)
        df_gt_padded = pad_with_nan_rows(df_gt, max_len)

        # Match columns using neuron matching
        df_res_renamed, _, _, _ = rename_columns_using_matching(
            df_gt_padded, df_res, column=column_name, try_to_fix_inf=True
        )
        # Reduce both DataFrames to raw_segmentation_id level
        df_res_renamed = df_res_renamed.xs(column_name, axis=1, level=1)
        df_gt_padded = df_gt_padded.xs(column_name, axis=1, level=1)

        # Calculate accuracy
        stats = calculate_accuracy(df_gt_padded, df_res_renamed)
        stats["trial"] = trial
        return stats

    except KeyError as e:
        print(f"Trial {trial}: ERROR during processing -> {e}")
        return {"trial": trial, "error": str(e)}


def discover_trials(trial_parent_dir):
    """
    Discovers trial numbers from folders named 'trial_<number>' in the given directory.
    """
    trials = []
    for entry in os.listdir(trial_parent_dir):
        entry_path = os.path.join(trial_parent_dir, entry)
        if os.path.isdir(entry_path) and "trial_" in entry:
            match = re.search(r"trial_(\d+)", entry)
            if match:
                trials.append(int(match.group(1)))
    return sorted(trials)


def extract_val_loss(trial_path):
    stats_path = os.path.join(trial_path, "log", "stats.json")
    if not os.path.isfile(stats_path):
        print(f"No stats.json found at {stats_path}")
        return None

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        if len(stats) >= 2 and "val_loss" in stats[-2]:
            return stats[-2]["val_loss"]
        else:
            print(f"{stats_path} too short or missing 'val_loss'")
            return None
    except Exception as e:
        print(f"Error reading {stats_path}: {e}")
        return None
    

def check_training_finished(trial_path, expected_num_epochs):
    stats_path = os.path.join(trial_path, "log", "stats.json")
    if not os.path.isfile(stats_path):
        print(f"No stats.json found at {stats_path}")
        return None

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        if "epoch" in stats[-1]:
            if stats[-1]["epoch"] == expected_num_epochs:
                print(f"{stats_path} shows that training didn't finish (epochs reached: {stats[-1]['epoch']})")
                return True
            else:
                return False
        else:
            print(f"{stats_path} missing 'epoch' field")
            return False
    except Exception as e:
        print(f"Error reading {stats_path}: {e}; assuming training did not finish")
        return False


def build_accuracy_dict(gt_path, project_dir, trial_dir=None, verbose=0):
    """
    Build a dictionary of accuracy metrics for all trials in the result directory.
    Assumes that each trial in the trial_dir is based on the same ground truth data.
    Further assumes that each project in the project_dir is named after its trial.
    """

    # Load GT once
    project_data_gt = ProjectData.load_final_project_data(gt_path, allow_hybrid_loading=True, verbose=verbose)
    df_gt, finished_neurons = project_data_gt.get_final_tracks_only_finished_neurons()
    if df_gt is None or df_gt.empty:
        logging.warning("No finished neurons found in ground truth data, assuming all neurons are ground truth.")
        df_gt = project_data_gt.final_tracks
    if df_gt is None:
        raise ValueError("No tracks found in the ground truth")

    result_dict = {
        "trial": [],
        "projector_final": [],
        "embedding_dim": [],
        "target_sz_z": [],
        "target_sz_xy": [],
        "p_RandomAffine_flip": [],
        "p_RandomAffine_base": [],
        "p_RandomAffine_both": [],
        "p_RandomAffine": [],
        "p_RandomElasticDeformation": [],
        "p_RandomBlur_base": [],
        "p_RandomNoise": [],
        "val_loss": [],
        "lr": [],
        "lambd_obj": [],
        "accuracy": [],
        "train_fraction": []
    }
    detailed_result_dict = {
        "per_neuron_accuracy": [],
        "per_timepoint_accuracy": [],
        "misses_per_neuron_norm": [],
        "misses_per_timepoint_norm": [],
        "mismatches_per_neuron_norm": [],
        "mismatches_per_timepoint_norm": [],
    }
    if trial_dir is not None:
        trials = discover_trials(trial_dir)
    else:
        trials = discover_trials(project_dir)
    # print(f"Found {len(trials)} trials")

    # Map trials to folder names within project_dir
    all_project_dirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
    if verbose >= 1:
        print(f"Found {len(all_project_dirs)} projects in {project_dir}")
    trial_to_project_map = {int(d.split("_")[-1]): d for d in all_project_dirs if "trial_" in d}

    for trial_num in tqdm(trials, leave=False):
        trial_name = f"trial_{trial_num}"
        trial_name_config = f"trial_{trial_num}"
        if trial_dir is not None:
            trial_path = os.path.join(trial_dir, trial_name_config)
            network_config_path = os.path.join(trial_path, "train_config.yaml")
        else:
            network_config_path = ""
        project_path = trial_to_project_map.get(trial_num, None)
        if project_path is None:
            print(f"{trial_name}: No matching project directory found in {project_dir} for trial {trial_num}; probably the traces have not yet been analyzed")
            continue
        project_path = os.path.join(project_dir, project_path, "project_config.yaml")

        try:
            with open(network_config_path, "r") as f:
                config = yaml.safe_load(f)

            if not check_training_finished(trial_path, config['epochs'] - 1):
                print(f"{trial_name}: training was not finished; skipping")
                continue

            result_dict["trial"].append(trial_num)
            for k in result_dict.keys():
                if k in ["trial", "accuracy", "val_loss"]:
                    continue
                result_dict[k].append(config.get(k))

            val_loss = extract_val_loss(trial_path)
            result_dict["val_loss"].append(val_loss)

        except FileNotFoundError:
            print(f"{trial_name}: train_config.yaml not found.")

        # The project may still exist even if the network can't be found
        try:
            if os.path.isfile(project_path):
                # print("Processing trials")
                stats = process_trial(trial_num, df_gt, project_path)
                # print(stats)
                result_dict["accuracy"].append(stats.get("accuracy"))
                
                for k in detailed_result_dict.keys():
                    detailed_result_dict[k].append(stats.get(k))
            else:
                print(f"{trial_name}: project_config.yaml not found.")
                result_dict["accuracy"].append(None)
                detailed_result_dict["per_neuron_accuracy"].append(None)
                detailed_result_dict["per_timepoint_accuracy"].append(None)

        except ValueError as e:
            print(f"{trial_name}: ERROR -> {e}")

    return result_dict, detailed_result_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking accuracy across multiple trials.")
    parser.add_argument("--ground_truth_path", required=True, help="Path to the ground truth NWB file")
    parser.add_argument("--res_path", required=True, help="Path to the directory containing trial runs")
    parser.add_argument("--trial_dir_prefix", default="", help="Optional prefix for trial directories (e.g. 2025_07_01)")
    parser.add_argument("--trials", required=True, help="Either a list of trial numbers [1,2,3] or a single integer for range 0..N-1")

    args = parser.parse_args()

    # Parse trials
    if args.trials.startswith("["):
        trials = eval(args.trials)
    else:
        trials = list(range(int(args.trials)))

    print(f"Loading ground truth from {args.ground_truth_path} ...")
    project_data_gt = ProjectData.load_final_project_data(args.ground_truth_path)
    df_gt = project_data_gt.final_tracks

    results = []
    for trial in trials:
        trial_dir = f"{args.trial_dir_prefix}trial_{trial}" if args.trial_dir_prefix else f"trial_{trial}"
        res_file = os.path.join(args.res_path, trial_dir, "project_config.yaml")

        if not os.path.exists(res_file):
            print(f"Trial {trial}: Skipping, result file not found at {res_file}")
            continue

        print(f"\nProcessing trial {trial} ...")
        stats = process_trial(trial, df_gt, res_file)
        results.append(stats)
        if "error" not in stats:
            print(f"Trial {trial}: {stats}")

    # Optionally print summary
    print("\nSummary:")
    for res in results:
        if "error" in res:
            print(f"Trial {res['trial']}: ERROR -> {res['error']}")
        else:
            print(f"Trial {res['trial']}: Accuracy {res['accuracy']:.4f} (Misses: {res['misses']}, Mismatches: {res['mismatches']})")


if __name__ == "__main__":
    main()
