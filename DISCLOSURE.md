# DISCLOSURE.md

Every external reference, library, model, dataset and tool, added the moment it is used.
"Licence" is filled only where it was checked; otherwise "to confirm".

## Reference code
| Item | Used for | Version | Licence |
|---|---|---|---|
| kekeblom/StrayVisualizer `stray_visualize.py` (github.com/kekeblom/StrayVisualizer) | Pose and axis convention for Stray Scanner data (quaternion xyzw, T_WC, Open3D pinhole back-projection, intrinsics scaling, index pairing). Convention only; code is re-written, not copied. | commit 195c640 | MIT (GitHub API) |

## Data
| Item | Used for | Notes |
|---|---|---|
| Three Stray Scanner captures supplied by the recruiter (single_room, single_scan_floor_only, single_scan_with_ceiling) | Development and reported results | not committed (ignored in .gitignore) |
| Stray Scanner iOS app export format | Input format | app not used by us in this phase (no iPhone) |

## Python libraries (versions from `uv pip list`)
| Library | Version | Used for | Licence |
|---|---|---|---|
| numpy | 2.5.3 | arrays | to confirm |
| scipy | 1.18.1 | rotations, signal, spatial search | to confirm |
| pandas | 3.0.6 | reading odometry.csv | to confirm |
| opencv-python | 5.0.0.93 | reading PNG/MP4 | to confirm |
| open3d | 0.20.0 | voxel downsampling, RANSAC planes, normals, ICP | to confirm |
| pydantic | 2.13.5 | JSON output schema | to confirm |
| typer | 0.27.2 | CLI | to confirm |
| matplotlib | 3.11.2 | density images, plan render | to confirm |
| shapely | 2.1.2 | room polygons | to confirm |
| pyyaml | 6.0.3 | config files | to confirm |
| pytest | 9.1.1 | tests (dev) | to confirm |
| pycolmap | 4.2.1 | SfM for the video/photo tiers (SIFT, CPU); ALIKED/LightGlue inside pycolmap crashed natively and was removed | to confirm |
| torch (CPU) | 2.14.1 | runs the models (`models` group) | to confirm |
| transformers | 5.18.0 | loads the models (`models` group) | to confirm |
| pillow | 12.3.0 | image I/O for models (`models` group) | to confirm |
| kornia | 0.8.3 | DISK keypoints + LightGlue matching for the learned video/photo matcher (`models` group; `sfm.matcher: learned`) | Apache-2.0 (to confirm) |
| pillow-heif | 1.8.0 | reading iPhone HEIC photos in the photo tier (`models` group) | to confirm |

## Models / APIs
Weights are fetched by `scripts/fetch_models.sh` at pinned revisions and never committed. Depth Anything is used by the video/photo tiers, OWL-ViT by the damage pass; both load at the pinned revision from config.

| Model | Revision | Used for | Licence |
|---|---|---|---|
| depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (24.8M params; fine-tuned on Hypersim, synthetic) | 8078d68a9c75a972131914f6afd0c1723be0da7f | single-image metric depth, aligned to the SfM points, for the metric scale of the video and photo tiers | to confirm. Upstream README says the Small model is Apache-2.0 (Base/Large/Giant are CC-BY-NC-4.0); this fine-tuned model's card has no licence field. |
| DISK ('depth' weights, via kornia `KF.DISK.from_pretrained`) | kornia default | learned keypoints for the video/photo tiers (experiment, `sfm.matcher: learned`) | to confirm (cvlab-epfl/disk) |
| LightGlue for DISK (via kornia `KF.LightGlueMatcher('disk')`) | kornia default | learned matching for the video/photo tiers (experiment) | to confirm (cvg/LightGlue) |
| google/owlvit-base-patch32 (153M params) | cbc355fb364588351c5d51c7f74465e8e7ec6f72 | zero-shot damage detector (`damage.detect`); no fine-tuning; accuracy untested | Apache-2.0 (model card tag) |

Network use: downloads from huggingface.co and the pip index happen at setup time only. The pipeline code itself makes no network calls.

## AI tooling
Code and analysis in this repo were written with Claude Code (Anthropic). The author reviews every commit.
