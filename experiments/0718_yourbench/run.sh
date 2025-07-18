#!/bin/bash
set -eu

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /tmp/zlu39/.conda_envs/yourbench_v0.3.1

yourbench run --config configs/five_papers.yaml