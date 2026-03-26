"""
Cross-recording Barlow Twins training.

Trains a single Barlow network using:
- Within-recording batches (same as original BarlowTrack)
- Cross-recording batches (same neuron type from different worms)

Curriculum: Phase A (within only) -> Phase B (mixed) -> Phase C (cross only)

Usage:
    python train_barlow_cross_recording.py -p path/to/train_config_cross_recording.yaml
"""
import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from barlow_track.utils.barlow import Transform
from barlow_track.utils.barlow_visualize import visualize_model_performance
from barlow_track.utils.cross_recording_dataset import (
    CurriculumScheduler,
    build_cross_recording_batch,
    build_within_recording_batch,
    load_multi_recording_crops,
    validate_training_data,
)
from barlow_track.utils.train_utils import (
    format_vectors_on_gpu,
    get_gpu,
    initialize_model,
    load_config,
    setup_wandb,
)

from wbfm.utils.general.utils_filenames import get_sequential_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_cross_recording(args):
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)

    gpu = get_gpu()
    logger.info(f"Using device: {gpu}")

    # ── Load multi-recording crop pool ───────────────────────────
    target_sz = np.array([args.target_sz_z, args.target_sz_xy, args.target_sz_xy])
    min_recordings_per_type = getattr(args, "min_recordings_per_type", 2)

    logger.info("Loading crops from all recordings...")
    pool = load_multi_recording_crops(
        project_paths=args.project_paths,
        num_frames_per_recording=getattr(args, "num_frames_per_recording", 100),
        target_sz=target_sz,
        min_recordings_per_type=min_recordings_per_type,
        seed=seed,
    )

    logger.info(
        f"Pool: {len(pool.crops)} crops, {pool.num_shared_types} shared types "
        f"(>= {min_recordings_per_type} recordings), {len(pool.recording_ids)} recordings"
    )

    # ── Validate training data ───────────────────────────────────
    report_path = validate_training_data(pool, output_dir=args.project_dir)

    if getattr(args, "dryrun", False):
        logger.info("Dryrun mode: stopping after validation report.")
        return

    if pool.num_shared_types == 0:
        raise ValueError(
            f"No neuron types shared across {min_recordings_per_type}+ recordings. "
            f"Lower min_recordings_per_type or add more annotated recordings."
        )

    # ── Initialize model ─────────────────────────────────────────
    model, args = initialize_model(args, gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    transform = Transform(args)

    # ── Curriculum scheduler ─────────────────────────────────────
    curriculum_phases = getattr(args, "curriculum_phases", [
        {"end_epoch": 50, "p_cross": 0.0},
        {"end_epoch": 150, "p_cross": 0.5},
        {"end_epoch": 200, "p_cross": 1.0},
    ])
    curriculum = CurriculumScheduler(
        phases=[(p["end_epoch"], p["p_cross"]) for p in curriculum_phases]
    )

    steps_per_epoch = getattr(args, "steps_per_epoch", 100)
    cross_batch_size = getattr(args, "cross_batch_size", 50)
    epochs = args.epochs
    print_freq = getattr(args, "print_freq", 50)

    # ── Setup logging/checkpoints ────────────────────────────────
    run = setup_wandb(args)

    log_dir = os.path.join(args.project_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.project_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    stats_file = get_sequential_filename(os.path.join(log_dir, "stats.json"))
    checkpoint_file = get_sequential_filename(os.path.join(checkpoint_dir, "checkpoint.pth"))

    json_stats = []
    if run is not None:
        json_stats.append(dict(run_name=run.name, run_id=run.id))
    else:
        json_stats.append(dict(run_name="local-run", run_id=None))

    # Save pool metadata
    pool_meta = dict(
        recording_ids=pool.recording_ids,
        num_shared_types=pool.num_shared_types,
        shared_names=sorted(pool.shared_names),
        min_recordings_per_type=min_recordings_per_type,
        total_crops=len(pool.crops),
    )
    with open(os.path.join(args.project_dir, "pool_metadata.json"), "w") as f:
        json.dump(pool_meta, f, indent=2)

    # ── Training loop ────────────────────────────────────────────
    start_time = time.time()
    logger.info(f"Starting training: {epochs} epochs, {steps_per_epoch} steps/epoch")

    try:
        for epoch in range(epochs):
            p_cross = curriculum.get_p_cross(epoch)
            epoch_loss = 0.0
            epoch_loss_feat = 0.0
            epoch_loss_obj = 0.0
            n_cross = 0
            n_within = 0

            for step in range(steps_per_epoch):
                global_step = epoch * steps_per_epoch + step

                # Curriculum: choose batch type
                use_cross = curriculum.should_use_cross(epoch) and pool.num_shared_types > 0

                if use_cross:
                    y1, y2 = build_cross_recording_batch(
                        pool, batch_size=cross_batch_size, transform=transform
                    )
                    n_cross += 1
                else:
                    y1, y2 = build_within_recording_batch(pool, transform=transform)
                    n_within += 1

                y1, y2 = format_vectors_on_gpu(y1, y2, gpu)

                optimizer.zero_grad()
                loss, loss_features, loss_objects = model.forward(y1, y2)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                epoch_loss_feat += loss_features.item()
                epoch_loss_obj += loss_objects.item()

                if global_step % print_freq == 0:
                    stats = dict(
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item(),
                        loss_features=loss_features.item(),
                        loss_objects=loss_objects.item(),
                        p_cross=p_cross,
                        batch_type="cross" if use_cross else "within",
                        batch_size=y1.shape[0],
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    json_stats.append(stats)

                    if run is not None:
                        run.log(stats)

                    # Infrequently: save correlation matrix visualization
                    if global_step % (20 * print_freq) == 0 and global_step > 0:
                        with torch.no_grad():
                            c = model.calculate_correlation_matrix(y1, y2)
                            save_fname = os.path.join(log_dir, f"correlation_matrix_{global_step}.png")
                            fig = visualize_model_performance(c, save_fname=save_fname, vmin=-0.5, vmax=1)
                            if run is not None:
                                run.log({"chart": fig})

            # End of epoch
            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            avg_feat = epoch_loss_feat / max(steps_per_epoch, 1)
            avg_obj = epoch_loss_obj / max(steps_per_epoch, 1)
            epoch_stats = dict(
                epoch=epoch,
                avg_loss=avg_loss,
                avg_loss_features=avg_feat,
                avg_loss_objects=avg_obj,
                p_cross=p_cross,
                n_cross_batches=n_cross,
                n_within_batches=n_within,
                time=int(time.time() - start_time),
            )
            print(json.dumps(epoch_stats))
            json_stats.append(epoch_stats)

            if run is not None:
                run.log(epoch_stats)

            # Save checkpoint
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                curriculum_phase=p_cross,
            )
            torch.save(state, checkpoint_file)

    except KeyboardInterrupt:
        logger.info("Interrupted, saving model...")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM: {e}, saving model...")
    finally:
        if run is not None:
            run.finish()

        # Save stats
        with open(stats_file, "w") as f:
            print(json.dumps(json_stats), file=f)

        # Save final model
        fname = get_sequential_filename(args.project_dir + "/resnet50.pth")
        torch.save(model.state_dict(), fname)
        logger.info(f"Model saved to {fname}")

        # Save args
        fname = get_sequential_filename(args.project_dir + "/args.pickle")
        with open(fname, "wb") as f:
            pickle.dump(args, f)

        logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cross-recording Barlow network")
    parser.add_argument(
        "--network_args", "-p", required=True,
        help="Path to cross-recording YAML config",
    )
    cli_args = parser.parse_args()
    args = load_config(cli_args.network_args)
    train_cross_recording(args)
