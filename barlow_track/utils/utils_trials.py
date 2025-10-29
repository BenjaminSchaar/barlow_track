import os
import re
import json
import yaml
import argparse
from test_metric import process_trial
from wbfm.utils.projects.finished_project_data import ProjectData
import matplotlib.pyplot as plt


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


def discover_trials(trial_parent_dir):
    """
    Discovers trial numbers from folders named 'trial_<number>' in the given directory.
    """
    trials = []
    for entry in os.listdir(trial_parent_dir):
        entry_path = os.path.join(trial_parent_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("trial_"):
            match = re.match(r"trial_(\d+)", entry)
            if match:
                trials.append(int(match.group(1)))
    return sorted(trials)



def build_final_dict(gt_path, trial_dir, result_dir, trial_prefix):
    # Load GT once
    project_data_gt = ProjectData.load_final_project_data(gt_path)
    df_gt = project_data_gt.final_tracks

    result_dict = {
        "trial": [],
        "projector_final": [],
        "embedding_dim": [],
        "target_sz_z": [],
        "target_sz_xy": [],
        "p_RandomAffine_flip": [],
        "val_loss": [],
        "accuracy": [],
    }

    trials = discover_trials(trial_dir)

    for trial_num in trials:
        trial_name = f"{trial_prefix}trial_{trial_num}"
        trial_name_config = f"trial_{trial_num}"
        trial_path = os.path.join(trial_dir, trial_name_config)
        result_path = os.path.join(result_dir, trial_name, "project_config.yaml")
        config_path = os.path.join(trial_path, "train_config.yaml")

        if not os.path.isfile(config_path):
            print(f"{trial_name}: train_config.yaml not found.")
            continue

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            result_dict["trial"].append(trial_num)
            result_dict["projector_final"].append(config.get("projector_final"))
            result_dict["embedding_dim"].append(config.get("embedding_dim"))
            result_dict["target_sz_z"].append(config.get("target_sz_z"))
            result_dict["target_sz_xy"].append(config.get("target_sz_xy"))
            result_dict["p_RandomAffine_flip"].append(config.get("p_RandomAffine_flip"))

            val_loss = extract_val_loss(trial_path)
            result_dict["val_loss"].append(val_loss)

            if os.path.isfile(result_path):
                stats = process_trial(trial_num, df_gt, result_path)
                result_dict["accuracy"].append(stats.get("accuracy"))
            else:
                print(f"{trial_name}: project_config.yaml not found.")
                result_dict["accuracy"].append(None)

        except Exception as e:
            print(f"{trial_name}: ERROR -> {e}")

    return result_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize config and evaluation metrics across trials.")
    parser.add_argument("--ground_truth_path", required=True, help="Path to ground truth NWB file")
    parser.add_argument("--trial_parent_dir", required=True, help="Directory with trial folders containing YAML and stats")
    parser.add_argument("--result_parent_dir", required=True, help="Directory with trial folders containing project_config.yaml")
    parser.add_argument("--trial_prefix", default="", help="Optional prefix before 'trial_XX'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    final_dict = build_final_dict(
        gt_path=args.ground_truth_path,
        trial_dir=args.trial_parent_dir,
        result_dir=args.result_parent_dir,
        trial_prefix=args.trial_prefix
    )

    print("\nFinal dictionary:")
    for k, v in final_dict.items():
        print(f"{k}: {v}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(final_dict["val_loss"], final_dict["accuracy"], c='blue', s=100)

    # Annotate each point with the trial number
    for i, trial in enumerate(final_dict["trial"]):
        plt.annotate(f'Trial {trial}', 
                    (final_dict["val_loss"][i], final_dict["accuracy"][i]),
                    textcoords="offset points", xytext=(5,5), ha='left')

    plt.title("Validation Loss vs Accuracy")
    plt.xlabel("Validation Loss")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
