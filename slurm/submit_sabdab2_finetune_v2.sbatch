#!/bin/bash
#SBATCH --job-name=sabdab2_finetune_v2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=/ssd/scxi253/antibody_workspace/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/antibody_workspace/logs/%x-%j.err

export PYTHONPATH=/ssd/scxi253/antibody_workspace/antibody-antigen-interface-research/src:$PYTHONPATH
PYTHON_BIN=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SCRIPT=/ssd/scxi253/antibody_workspace/antibody-antigen-interface-research/scripts/sabdab2_interaction_first.py
SABDAB2_DATA_ROOT=/ssd/scxi253/model_deployment/data/interaction_first
SABDAB2_PREP_ROOT=/ssd/scxi253/model_deployment/data/interaction_first_sabdab2
OUTPUT_DIR=/ssd/scxi253/antibody_workspace/results/sabdab2_finetuned_v2
DONOR_MODEL=/ssd/scxi253/antibody_workspace/results/asep_multitask_v1/best.pt

$PYTHON_BIN $SCRIPT train \
    --prepare-dir $SABDAB2_PREP_ROOT \
    --contact-dir $SABDAB2_DATA_ROOT/contacts \
    --embedding-dir /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
    --split-manifest $SABDAB2_DATA_ROOT/double_remote_split.json \
    --fine-tune-from $DONOR_MODEL \
    --output-dir $OUTPUT_DIR \
    --device cuda \
    --max-epochs 30 \
    --patience 10 \
    --num-layers 4 \
    --focal-gamma 2.0 \
    --max-positive-weight 5.0 \
    --intra-chain-geometry-layers 2 \
    --backbones-pt-path /ssd/scxi253/antibody_workspace/results/sabdab2_full_backbones.pt \
    --margin-separation-weight 0.1 \
    --margin-value 2.0 \
    --epitope-consistency-weight 1.0 \
    --paratope-consistency-weight 0.5 \
    --epitope-dice-weight 1.0 \
    --paratope-dice-weight 0.5 \
    --selection-metric total \
    --learning-rate 2e-5
