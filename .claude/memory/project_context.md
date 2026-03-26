# BarlowTrack Project Context

**Project Path:** `/Users/benjaminschaar/Documents/GitHub/barlow_track`
**Purpose:** Self-supervised neuron tracking for C. elegans whole-brain recordings using Barlow Twins embeddings

---

## Pipeline Architecture (Updated: 2026-02-25)

### Overview
BarlowTrack uses self-supervised learning (Barlow Twins) to generate embeddings of neuron 3D crops, then clusters those embeddings to assign consistent IDs across time. It does NOT use frame-to-frame nearest-neighbor matching.

### Full Pipeline

#### Step 1: Detection (pre-existing input)
- Neurons already segmented before BarlowTrack runs
- Each frame has a 3D labeled mask with unique integer per neuron blob
- BarlowTrack takes these detections as input

#### Step 2: Crop Extraction
- For each detected neuron at each timepoint, a 3D crop is extracted around its centroid
- Default crop size: (8, 64, 64) in (z, x, y)
- Key function: `get_bbox_data_for_volume_with_label()` in `data_loading.py`

#### Step 3: Barlow Twins Training (self-supervised)
- **Positive pairs**: Same neuron at SAME timepoint with two different augmentations (NOT same neuron at different timepoints)
  - View 1: random z-rotation + blur + intensity rescaling
  - View 2: heavier augmentations — rotation + elastic deformation + noise + rescaling
- **Two loss terms**:
  - Feature correlation loss (standard Barlow Twins): makes embeddings invariant to augmentations
  - Object correlation loss (BarlowTrack innovation): forces different neurons within same frame to have different embeddings — this is what makes the embedding space discriminative
- Architecture: ResidualEncoder3D backbone (3D ResNet-style), projector MLP head
- Key files: `barlow.py`, `barlow_lightning.py`

#### Step 4: Embedding Inference
- Projector head discarded after training (standard SSL practice)
- Every neuron crop at every timepoint passed through backbone only → 64-dim embedding
- Result: matrix X of shape (total_detections_across_all_frames, 64)
- Key function: `embed_using_barlow()` in `track_using_barlow.py`

#### Step 5: Dimensionality Reduction
- SVD/PCA reduces from 64 dims to ~50 components
- Removes noise, speeds up clustering
- Key function: `_robust_svd()` in `track_using_barlow.py`

#### Step 6: Clustering = Tracking
Four available modes:
1. **Global HDBSCAN** (default): Clusters all embeddings across all timepoints at once. min_cluster_size ≈ 50% of frames. Each cluster = one neuron ID.
2. **Overlapping windows**: HDBSCAN on temporal windows, stitched via Hungarian algorithm
3. **Streaming**: HDBSCAN on subset, approximate_predict for rest
4. **Label propagation** (most sophisticated): KNN graph → graph diffusion from seed timepoints → spectral synchronization + Hungarian matching for consensus

Core class: `WormClusterTracker` in `utils_tracking.py`

#### Step 7: Output
- DataFrame `df_barlow_tracks.h5` mapping each timepoint to which segmented object belongs to which neuron ID
- Columns: raw_neuron_ind_in_list, likelihood, raw_segmentation_id, x, y, z per neuron
- Saved in `3-tracking/barlow_tracker/`

### Key Insight
Same neuron across many timepoints → many embeddings close together in embedding space → HDBSCAN groups them into 1 cluster → that cluster = 1 tracked ID. The "many perspectives" during training are augmentations (rotations, deformations), not timepoints. But the learned features are stable enough that the same neuron at different timepoints naturally maps to the same embedding region.

---

## Key Files

| File | Purpose |
|------|---------|
| `barlow_track/utils/barlow.py` | Barlow Twins model and augmentations |
| `barlow_track/utils/barlow_lightning.py` | Training loop |
| `barlow_track/utils/track_using_barlow.py` | Embedding inference pipeline |
| `barlow_track/utils/utils_tracking.py` | WormClusterTracker with all tracking modes |
| `barlow_track/utils/utils_label_propagation.py` | Graph-based tracking |
| `barlow_track/utils/utils_spectral_relabeling.py` | Spectral synchronization |
| `barlow_track/utils/data_loading.py` | Crop extraction |

---

## History

### 2026-02-25
- **Added:** Comprehensive pipeline documentation explaining how BarlowTrack works end-to-end
- **Key learning:** Barlow Twins creates positive pairs from augmentations of same neuron at same timepoint (not across time). Object correlation loss ensures different neurons have different embeddings. Clustering in embedding space is what performs the tracking.
- **Created:** `embedding_atlas/` project at `/Volumes/scratch/.../wbfm_analysis/embedding_atlas/` — full implementation of cross-recording neuron ID transfer using per-frame embeddings instead of means. 12 Python files, 5 configs, SLURM scripts.
- **Key fix:** HDF5 tracks key is `df_with_missing` not `df` — added fallback in data_loading.py
- **Verified:** All imports pass. Data loader tested on real recording (zarr shape 276K x 2048).
- **Configs:** Copied Itamar's left (26 recs) and right (24 recs) YAML configs from neuron_id_transfer.
- **Checkpoints:** Every pipeline step saves pickle files for post-hoc analysis.
- **Cluster fix:** Added PYTHONPATH to runme.sh so embedding_atlas is importable without pip install.

---

## TODOs

- Run exploration phase on SLURM cluster (left + right configs separately)
- Review exploration results to decide which neurons are "clean"
- Run classification phase if exploration looks promising

---

## Related Projects

- **WBFM pipeline:** Uses BarlowTrack as tracking step after neuron detection
- **ID transfer diagnosis:** Analysis notebooks at `/Volumes/scratch/neurobiology/zimmer/schaar/wbfm/scripts/jupyter_notebooks/id_transfer_diagnosis/`
- **Embedding Atlas (active, implemented):** `/Volumes/scratch/neurobiology/zimmer/schaar/wbfm/scripts/wbfm_analysis/embedding_atlas/` — Full pipeline for cross-recording neuron ID transfer using per-frame Barlow embeddings (2048-dim). Trains on Itamar's 50 recordings, classifies via frame-level voting. See `~/.claude/memory/projects/embedding_atlas.md`

---

## Open Questions (Resolved)

**Status:** RESOLVED — Built the `embedding_atlas` project to test this hypothesis. Per-frame voting with frame-level classification should overcome ambiguous mean-embedding clusters.

### 2026-02-25: Classification-based Neuron ID Transfer — Can it Generalize?

**Topic**: Can the ID transfer pipeline's classification approach generalize to unknown recordings if trained on enough ground truth data?

**Context**: Benjamin asked whether the ID transfer pipeline's classification approach (which learns the embedding space for given neuron identities using ground truth labels) could generalize to completely new/unannotated recordings if trained on enough ground truth data (e.g., AVAL annotated across many recordings).

**What we established in this session**:
1. Full BarlowTrack pipeline documented (7 steps from detection to tracked IDs)
2. Barlow Twins model is trained ONCE and reused across recordings (not per-recording)
3. Current cross-recording performance: 60% of shared named neurons cluster together, 40% split
4. Silhouette score for neuron identity: 0.156 (target: 0.3+)
5. Existing ID transfer uses SuperGlue matching + spatial validation (documented in barlow-embedding-findings.md)

**Still to investigate**:
- The exact classification mechanism in the ID transfer pipeline (user says it uses classification to learn embedding space for neuron identity)
- Whether training such a classifier on ground truth neurons from MANY recordings would be enough to categorize neurons in completely unseen recordings
- Need to check: id_transfer notebooks, any classifier code in barlow_track, SuperGlue learned matching

**Key files to check next time**:
- `/Volumes/scratch/neurobiology/zimmer/schaar/wbfm/scripts/jupyter_notebooks/id_transfer_diagnosis/`
- barlow_track SuperGlue code
- Any classification-based ID transfer utils
