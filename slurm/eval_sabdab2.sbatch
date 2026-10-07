#!/bin/bash
#SBATCH --job-name=sabdab2_eval
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00
#SBATCH --output=/ssd/scxi253/antibody_workspace/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/antibody_workspace/logs/%x-%j.err

export PYTHONPATH=/ssd/scxi253/antibody_workspace/antibody-antigen-interface-research/src:$PYTHONPATH
PYTHON_BIN=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SCRIPT=/ssd/scxi253/antibody_workspace/antibody-antigen-interface-research/scripts/sabdab2_interaction_first.py
SABDAB2_DATA_ROOT=/ssd/scxi253/model_deployment/data/interaction_first
SABDAB2_PREP_ROOT=/ssd/scxi253/model_deployment/data/interaction_first_sabdab2
OUTPUT_DIR=/ssd/scxi253/antibody_workspace/results/sabdab2_finetuned_v1

$PYTHON_BIN $SCRIPT evaluate \
    --prepare-dir $SABDAB2_PREP_ROOT \
    --contact-dir $SABDAB2_DATA_ROOT/contacts \
    --embedding-dir /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
    --split-manifest $SABDAB2_DATA_ROOT/double_remote_split.json \
    --checkpoint $OUTPUT_DIR/best.pt \
    --output-dir $OUTPUT_DIR \
    --split test \
    --device cuda \
    --backbones-pt-path /ssd/scxi253/antibody_workspace/results/sabdab2_full_backbones.pt
