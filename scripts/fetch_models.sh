#!/usr/bin/env bash
# Download model weights into the Hugging Face cache at pinned revisions. Weights are never committed.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --group models python - <<'EOF'
from huggingface_hub import snapshot_download

MODELS = [
    ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "8078d68a9c75a972131914f6afd0c1723be0da7f"),
    ("google/owlvit-base-patch32", "cbc355fb364588351c5d51c7f74465e8e7ec6f72"),
]
for repo, rev in MODELS:
    print(repo, "->", snapshot_download(repo, revision=rev, ignore_patterns=["*.bin", "*.h5", "*.msgpack"]))  # safetensors only
EOF
