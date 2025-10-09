import os
import shutil


def main():
    # Root folder that contains name1, name2, ...
    root_dir = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects"

    for name_folder in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name_folder)
        if os.path.isdir(full_path):
            baseline_path = os.path.join(full_path, "baseline")

            if os.path.exists(baseline_path):
                # Define target folders
                baseline_old = os.path.join(full_path, "baseline_old_tracklets")
                baseline_hdbscan = os.path.join(full_path, "baseline_hdbscan")

                # Copy baseline -> baseline_old_tracklets
                if not os.path.exists(baseline_old):
                    shutil.copytree(baseline_path, baseline_old)
                    print(f"Copied {baseline_path} -> {baseline_old}")
                else:
                    print(f"Skipped {baseline_old} (already exists)")

                # Copy baseline -> baseline_hdbscan
                if not os.path.exists(baseline_hdbscan):
                    shutil.copytree(baseline_path, baseline_hdbscan)
                    print(f"Copied {baseline_path} -> {baseline_hdbscan}")
                else:
                    print(f"Skipped {baseline_hdbscan} (already exists)")
            else:
                print(f"No baseline found in {name_folder}")


if __name__ == "__main__":
    main()
    