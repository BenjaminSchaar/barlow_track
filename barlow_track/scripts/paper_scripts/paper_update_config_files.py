import os
import yaml


def update_yaml_file(file_path, subfolder, updates, filename_to_modify, DEBUG=False):
    for lab_folder in os.listdir(root_dir):
        full_path = os.path.join(root_dir, lab_folder)
        if not os.path.isdir(full_path):
            continue

        for new_name in [subfolder]:
            lab_subfolder = os.path.join(full_path, new_name)

            # Process YAML files recursively
            for dirpath, _, filenames in os.walk(lab_subfolder):
                for file in filenames:
                    if file == filename_to_modify:
                        file_path = os.path.join(dirpath, file)
                        with open(file_path, "r") as f:
                            data = yaml.safe_load(f)

                        # Update the top-level keys with new values
                        if isinstance(data, dict):
                            if DEBUG:
                                print(f"Would update {file_path} with {updates}; current value: {[data.get(k, 'NOT PRESENT') for k in updates.keys()]}")
                            else:
                                data.update(updates)
                                with open(file_path, "w") as f:
                                    yaml.safe_dump(data, f, default_flow_style=False)
                            print(f"Updated {file_path}")
                        else:
                            print(f"Skipped {file_path} (not a dict at top level)")

if __name__ == "__main__":
    DEBUG = False

    root_dir = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects"

    # subfolder = "baseline_old_tracklets"  # subfolder to look into
    # filename_to_modify = "snakemake_config.yaml"  # exact name of the YAML file to modify
    # updates = {
    #     "use_barlow_tracker": False
    # }
    subfolder = "baseline_hdbscan"  # subfolder to look into
    filename_to_modify = "tracking_config.yaml"  # exact name of the YAML file to modify
    updates = {
        "barlow_tracker": {"tracking_mode": "global"}
    }
    update_yaml_file(root_dir, subfolder, updates, filename_to_modify, DEBUG)
