from barlow_track.utils.utils_trials import parse_args, build_final_dict
import matplotlib.pyplot as plt


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
