"""
Cross-recording dataset and batch construction for multi-worm Barlow Twins training.

Key concepts:
- LabeledCropPool: flat index of all labeled neuron crops across recordings
- build_cross_recording_batch: constructs aligned positive pairs across recordings
- build_within_recording_batch: standard single-frame Barlow batch
- CurriculumScheduler: epoch-dependent mixing of within vs cross-recording batches
- validate_training_data: generates report of which neuron types qualify for cross-recording pairs
"""

import csv
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

# Lazy imports — these pull in torchio/wbfm which may not be available in all environments.
# They are only needed by the batch-building and data-loading functions, not by the
# data structures (NeuronCrop, LabeledCropPool, CurriculumScheduler).
_Transform = None
_get_bbox = None


def _lazy_import_transform():
    global _Transform
    if _Transform is None:
        from barlow_track.utils.barlow import Transform
        _Transform = Transform
    return _Transform


def _lazy_import_get_bbox():
    global _get_bbox
    if _get_bbox is None:
        from barlow_track.utils.data_loading import get_bbox_data_for_volume_with_label
        _get_bbox = get_bbox_data_for_volume_with_label
    return _get_bbox

logger = logging.getLogger(__name__)


@dataclass
class NeuronCrop:
    """Single neuron crop with identity metadata."""
    recording_id: str       # e.g. "control_1per_worm2"
    neuron_name: str        # e.g. "AVAL" (biological identity) or "neuron_042" (untracked)
    frame_idx: int          # timepoint within the recording
    crop: np.ndarray        # shape (z, x, y), e.g. (8, 64, 64)
    is_annotated: bool = True  # False for untracked neurons


@dataclass
class LabeledCropPool:
    """
    Flat index of all labeled neuron crops across multiple recordings.

    Provides efficient lookup by neuron name (for cross-recording pairing)
    and by recording_id (for within-recording batches).
    """
    crops: List[NeuronCrop]
    min_recordings_per_type: int = 2

    # Indices built in __post_init__
    by_name: Dict[str, List[int]] = field(default_factory=dict, init=False)
    by_recording: Dict[str, List[int]] = field(default_factory=dict, init=False)
    by_recording_and_frame: Dict[str, Dict[int, List[int]]] = field(default_factory=dict, init=False)
    name_to_recordings: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    shared_names: Set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        """Build lookup indices."""
        for i, crop in enumerate(self.crops):
            self.by_name.setdefault(crop.neuron_name, []).append(i)
            self.by_recording.setdefault(crop.recording_id, []).append(i)

            # Frame-level index for within-recording batches
            rec_frames = self.by_recording_and_frame.setdefault(crop.recording_id, {})
            rec_frames.setdefault(crop.frame_idx, []).append(i)

            if crop.is_annotated:
                self.name_to_recordings.setdefault(crop.neuron_name, set()).add(crop.recording_id)

        # Shared names: annotated neuron types present in min_recordings_per_type+ recordings
        self.shared_names = {
            name for name, recs in self.name_to_recordings.items()
            if len(recs) >= self.min_recordings_per_type
        }

    def get_cross_recording_pair(self, neuron_name: str) -> Tuple[NeuronCrop, NeuronCrop]:
        """
        Sample two crops of the same neuron type from DIFFERENT recordings.
        Returns (crop_from_rec_a, crop_from_rec_b).
        """
        assert neuron_name in self.shared_names, f"{neuron_name} not in shared_names"

        available_recordings = list(self.name_to_recordings[neuron_name])
        rec_a, rec_b = random.sample(available_recordings, 2)

        # Filter to crops from the chosen recordings with matching neuron name
        crops_rec_a = [i for i in self.by_name[neuron_name]
                       if self.crops[i].recording_id == rec_a]
        crops_rec_b = [i for i in self.by_name[neuron_name]
                       if self.crops[i].recording_id == rec_b]

        idx_a = random.choice(crops_rec_a)
        idx_b = random.choice(crops_rec_b)

        return self.crops[idx_a], self.crops[idx_b]

    @property
    def recording_ids(self) -> List[str]:
        return list(self.by_recording.keys())

    @property
    def num_shared_types(self) -> int:
        return len(self.shared_names)

    @property
    def all_neuron_names(self) -> Set[str]:
        return set(self.name_to_recordings.keys())


@dataclass
class CurriculumScheduler:
    """
    Returns the probability of drawing a cross-recording batch given the current epoch.

    Phases defined as list of (end_epoch, p_cross) tuples, evaluated in order.
    Default: Phase A (0-49, p=0.0), Phase B (50-149, p=0.5), Phase C (150-199, p=1.0)
    """
    phases: List[Tuple[int, float]] = field(default_factory=lambda: [
        (50, 0.0),
        (150, 0.5),
        (200, 1.0),
    ])

    def get_p_cross(self, epoch: int) -> float:
        for end_epoch, p_cross in self.phases:
            if epoch < end_epoch:
                return p_cross
        return self.phases[-1][1]

    def should_use_cross(self, epoch: int) -> bool:
        return random.random() < self.get_p_cross(epoch)


def load_multi_recording_crops(
    project_paths: Dict[str, str],
    num_frames_per_recording: int = 100,
    target_sz: np.ndarray = np.array([8, 64, 64]),
    min_recordings_per_type: int = 2,
    seed: int = 42,
) -> LabeledCropPool:
    """
    Load labeled neuron crops from multiple recordings into a single LabeledCropPool.

    Parameters
    ----------
    project_paths : dict
        Mapping recording_id -> WBFM project path
    num_frames_per_recording : int
        Frames to sample per recording
    target_sz : np.ndarray
        3D crop size [z, x, y]
    min_recordings_per_type : int
        Minimum recordings a neuron type must appear in for cross-recording pairing (>= 2)
    seed : int
        Random seed for frame sampling

    Returns
    -------
    LabeledCropPool with all labeled crops indexed
    """
    from wbfm.utils.projects.finished_project_data import ProjectData

    assert min_recordings_per_type >= 2, "min_recordings_per_type must be >= 2"

    get_bbox_data_for_volume_with_label = _lazy_import_get_bbox()

    rng = random.Random(seed)
    all_crops = []

    for recording_id, project_path in tqdm(project_paths.items(),
                                            desc="Loading recordings"):
        logger.info(f"Loading {recording_id} from {project_path}")
        try:
            project_data = ProjectData.load_final_project_data(
                project_path, allow_hybrid_loading=True
            )
        except Exception as e:
            logger.warning(f"Failed to load {recording_id}: {e}")
            continue

        max_frames = project_data.num_frames
        n_sample = min(num_frames_per_recording, max_frames)
        frame_indices = rng.sample(range(max_frames), n_sample)

        n_crops_this_recording = 0
        for t in tqdm(frame_indices, desc=f"  {recording_id}", leave=False):
            try:
                all_dat_dict, seg2name, which_neurons = get_bbox_data_for_volume_with_label(
                    project_data, t, target_sz=target_sz, include_untracked=True
                )
            except (KeyError, IndexError, FileNotFoundError) as e:
                logger.debug(f"Frame {t} in {recording_id}: {e}")
                continue

            for neuron_name, crop_data in all_dat_dict.items():
                is_annotated = not neuron_name.startswith("untracked_")
                all_crops.append(NeuronCrop(
                    recording_id=recording_id,
                    neuron_name=neuron_name,
                    frame_idx=t,
                    crop=crop_data,
                    is_annotated=is_annotated,
                ))
                n_crops_this_recording += 1

        logger.info(f"  {recording_id}: {n_crops_this_recording} crops from {n_sample} frames")

    pool = LabeledCropPool(crops=all_crops, min_recordings_per_type=min_recordings_per_type)
    logger.info(
        f"Loaded {len(all_crops)} crops from {len(project_paths)} recordings. "
        f"{pool.num_shared_types} neuron types shared across {min_recordings_per_type}+ recordings."
    )
    return pool


def validate_training_data(
    pool: LabeledCropPool,
    output_dir: str,
) -> str:
    """
    Generate a training data validation report.

    Prints a summary table and saves training_data_report.csv showing:
    - Each neuron type, how many recordings it appears in, total crops
    - Whether it qualifies for cross-recording pairs at the current threshold
    - Summary statistics at multiple threshold levels

    Parameters
    ----------
    pool : LabeledCropPool
    output_dir : str
        Directory to save training_data_report.csv

    Returns
    -------
    Path to the saved CSV report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-type statistics
    type_stats = []
    for name, recs in sorted(pool.name_to_recordings.items()):
        n_recordings = len(recs)
        n_crops = len([i for i in pool.by_name.get(name, []) if pool.crops[i].is_annotated])
        qualifies = name in pool.shared_names
        type_stats.append({
            "neuron_type": name,
            "n_recordings": n_recordings,
            "recordings": ",".join(sorted(recs)),
            "total_crops": n_crops,
            "qualifies_for_cross": "yes" if qualifies else "no",
        })

    # Sort by n_recordings descending
    type_stats.sort(key=lambda x: (-x["n_recordings"], x["neuron_type"]))

    # Save CSV
    csv_path = output_dir / "training_data_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=type_stats[0].keys())
        writer.writeheader()
        writer.writerows(type_stats)

    # Print summary
    n_qualifying = sum(1 for s in type_stats if s["qualifies_for_cross"] == "yes")
    n_total = len(type_stats)
    n_crops_qualifying = sum(s["total_crops"] for s in type_stats if s["qualifies_for_cross"] == "yes")

    print(f"\n{'='*70}")
    print(f"TRAINING DATA VALIDATION REPORT")
    print(f"{'='*70}")
    print(f"Recordings loaded:     {len(pool.recording_ids)}")
    print(f"Total annotated types: {n_total}")
    print(f"min_recordings_per_type: {pool.min_recordings_per_type}")
    print(f"Qualifying types:      {n_qualifying} / {n_total}")
    print(f"Qualifying crops:      {n_crops_qualifying}")
    print(f"Total crops (all):     {len(pool.crops)}")
    print(f"")

    # Show threshold sensitivity
    print(f"Threshold sensitivity:")
    print(f"  {'threshold':>10}  {'types':>8}  {'crops':>8}")
    for threshold in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
        n_types = sum(1 for s in type_stats if s["n_recordings"] >= threshold)
        n_crops = sum(s["total_crops"] for s in type_stats if s["n_recordings"] >= threshold)
        marker = " <-- current" if threshold == pool.min_recordings_per_type else ""
        print(f"  {threshold:>10}  {n_types:>8}  {n_crops:>8}{marker}")
    print(f"")

    # Show top types
    print(f"Top 20 neuron types by coverage:")
    print(f"  {'type':<15} {'recordings':>12} {'crops':>8} {'qualifies':>10}")
    for s in type_stats[:20]:
        print(f"  {s['neuron_type']:<15} {s['n_recordings']:>12} {s['total_crops']:>8} {s['qualifies_for_cross']:>10}")

    if n_total > 20:
        n_remaining = n_total - 20
        print(f"  ... and {n_remaining} more types (see {csv_path})")

    print(f"{'='*70}")
    print(f"Report saved to: {csv_path}")
    print(f"{'='*70}\n")

    return str(csv_path)


def build_cross_recording_batch(
    pool: LabeledCropPool,
    batch_size: int = 50,
    transform=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct one cross-recording training batch.

    For each of batch_size neuron types, sample one crop from recording A
    and one from recording B. Apply augmentation to both independently.

    Returns
    -------
    y1 : (batch_size, 1, z, x, y) — crops from recording A, augmented
    y2 : (batch_size, 1, z, x, y) — crops from recording B, augmented (same neuron types)
    """
    if transform is None:
        Transform = _lazy_import_transform()
        transform = Transform()

    shared = list(pool.shared_names)
    if not shared:
        raise ValueError("No shared neuron types available for cross-recording batch")

    # Sample batch_size neuron types (with replacement if needed)
    if batch_size <= len(shared):
        selected_names = random.sample(shared, batch_size)
    else:
        selected_names = random.choices(shared, k=batch_size)

    crops_a, crops_b = [], []
    for name in selected_names:
        crop_a, crop_b = pool.get_cross_recording_pair(name)
        crops_a.append(crop_a.crop)
        crops_b.append(crop_b.crop)

    # Stack into tensors: (batch, z, x, y)
    batch_a = torch.from_numpy(np.stack(crops_a, 0).astype(np.float32))
    batch_b = torch.from_numpy(np.stack(crops_b, 0).astype(np.float32))

    # Apply augmentation independently to each set of crops.
    # transform.transform = lighter aug (base path)
    # transform.transform_prime = heavier aug (prime path)
    # For cross-recording: crop_A gets base, crop_B gets prime.
    # The cross-recording difference IS the "natural augmentation".
    y1 = transform.transform(batch_a)
    y2 = transform.transform_prime(batch_b)

    return y1, y2


def build_within_recording_batch(
    pool: LabeledCropPool,
    transform=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct one within-recording batch (same as original Barlow training).

    Randomly select one recording, one frame from that recording,
    and return all neurons in that frame as a standard Barlow augmented pair.
    """
    if transform is None:
        Transform = _lazy_import_transform()
        transform = Transform()

    # Pick a random recording
    rec_id = random.choice(pool.recording_ids)
    frame_groups = pool.by_recording_and_frame.get(rec_id, {})

    # Pick a random frame with valid neuron count (1 < n <= 200)
    valid_frames = [f for f, indices in frame_groups.items()
                    if 1 < len(indices) <= 200]
    if not valid_frames:
        # Fallback: any frame with >1 neuron
        valid_frames = [f for f, indices in frame_groups.items() if len(indices) > 1]
    if not valid_frames:
        # Last resort: try another recording
        for _ in range(10):
            rec_id = random.choice(pool.recording_ids)
            frame_groups = pool.by_recording_and_frame.get(rec_id, {})
            valid_frames = [f for f, indices in frame_groups.items() if len(indices) > 1]
            if valid_frames:
                break
        if not valid_frames:
            raise RuntimeError("Could not find any frame with >1 neuron across all recordings")

    frame = random.choice(valid_frames)
    frame_crop_indices = frame_groups[frame]

    # Stack crops
    crop_data = np.stack(
        [pool.crops[i].crop for i in frame_crop_indices], 0
    ).astype(np.float32)
    batch = torch.from_numpy(crop_data)

    # Standard Barlow pair: same crops, two different augmentations
    y1, y2 = transform(batch)
    return y1, y2
