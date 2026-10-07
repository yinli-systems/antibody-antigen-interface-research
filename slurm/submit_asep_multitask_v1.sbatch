#!/bin/bash
#SBATCH --job-name=asep_multitask_v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/antibody_workspace/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/antibody_workspace/logs/%x-%j.err

export PYTHONPATH=/ssd/scxi253/antibody_workspace/antibody-antigen-interface-research/src:$PYTHONPATH
PYTHON_BIN=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SCRIPT=/ssd/scxi253/antibody_workspace/antibody-antigen-interface-research/scripts/sabdab2_interaction_first.py
ASEP_DATA_ROOT=/ssd/scxi253/model_deployment/data/interaction_first_asep
OUTPUT_DIR=/ssd/scxi253/antibody_workspace/results/asep_multitask_v1

$PYTHON_BIN $SCRIPT train \
    --prepare-dir $ASEP_DATA_ROOT \
    --contact-dir $ASEP_DATA_ROOT/contacts \
    --embedding-dir $ASEP_DATA_ROOT/embeddings \
    --split-manifest $ASEP_DATA_ROOT/asep_graph_split.json \
    --output-dir $OUTPUT_DIR \
    --device cuda \
    --max-epochs 20 \
    --patience 10 \
    --num-layers 4 \
    --focal-gamma 2.0 \
    --max-positive-weight 5.0 \
    --intra-chain-geometry-layers 2 \
    --backbones-pt-path $ASEP_DATA_ROOT/backbones.pt \
    --margin-separation-weight 0.1 \
    --margin-value 2.0 \
    --epitope-consistency-weight 0.5 \
    --paratope-consistency-weight 0.5 \
    --epitope-dice-weight 0.5 \
    --paratope-dice-weight 0.5 \
    --learning-rate 5e-5
