#!/bin/bash
#SBATCH --job-name=epiformer-train
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
EPOCHS=${EPI_EPOCHS:-1}
RUN_ID=${EPI_RUN_ID:-formal_epi_group_seed42}
RESUME=${EPI_RESUME:-false}
REPORT_DIR=$DEPLOY_ROOT/reports/epiformer-train-$SLURM_JOB_ID

mkdir -p "$REPORT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export WANDB_MODE=disabled

nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 2 > "$REPORT_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

cd "$DEPLOY_ROOT/src/epiformer/code"
/usr/bin/time -v "$PYTHON_BIN" trainer.py \
  mode=val \
  base_dir="$DEPLOY_ROOT" \
  seed=42 \
  model.name=epiformer \
  gpu_id=0 \
  logging_method=none \
  dataset.split.method=epitope_group \
  dataset.graph_type=raad-plm \
  dataset.plm_type=esm2_650m \
  dataset.graph_num_relations=4 \
  hparams.train.num_epochs="$EPOCHS" \
  hparams.train.batch_size=8 \
  hparams.train.learning_rate=0.00006464091462000422 \
  hparams.train.weight_decay=0.00009865396767252336 \
  hparams.train.kfolds=2 \
  hparams.train.regularization.use_l2_reg=false \
  hparams.train.scheduler=reduce_lr_on_plateau \
  run_id="$RUN_ID" \
  num_threads=3 \
  resume="$RESUME" \
  model.epiformer.ag_resmp_type=egnn \
  model.epiformer.ab_resmp_type=egnn \
  model.epiformer.residue_layers=4 \
  model.epiformer.residue_dim=128 \
  model.epiformer.residue_hidden_dim=128 \
  model.epiformer.plm_dim=128 \
  model.epiformer.n_heads=8 \
  model.epiformer.use_layer_norm=true \
  model.epiformer.use_pair_repr=false \
  model.epiformer.use_gradient_checkpointing=false \
  model.epiformer.ag_feature_fusion_type=concat \
  model.epiformer.ab_feature_fusion_type=gated \
  model.epiformer.activation=silu \
  model.epiformer.dropout=0.05275931970788925 \
  model.dropout_rates.decoder=0.1062338229383681 \
  model.dropout_rates.res_mp=0.1 \
  model.dropout_rates.atom_mp=0.2 \
  model.dropout_rates.edge_mp=0.2 \
  model.dropout_rates.projections=0.1 \
  model.decoder.type=cross_attention \
  model.decoder.num_rbf=16 \
  model.decoder.d_k=64 \
  model.decoder.d_ff=128 \
  model.decoder.d_model=128 \
  model.decoder.n_heads=8 \
  model.decoder.decoder_layers=3 \
  model.decoder.sampling_strat=top_k_mean_2 \
  model.decoder.predict_distances=true \
  model.epi_threshold=0.3 \
  model.para_threshold=0.3 \
  model.use_layer_norm=true \
  callbacks.early_stopping.patience=10 \
  callbacks.checkpoint_interval=5 \
  loss.multi_task=true \
  loss.label_smoothing=0.1 \
  loss.class_balance.beta=0.9999 \
  loss.node_prediction.enabled=true \
  loss.node_prediction.weight=0.14290586700768587 \
  loss.node_prediction.name=bce \
  loss.node_prediction.task=epi_only \
  loss.node_prediction.bce_weight=9.161617641136282 \
  loss.node_prediction.dice_weight=1.8251606497466577 \
  loss.node_prediction.smoothness_weight=0.01 \
  loss.node_prediction.consistency_weight=0.1 \
  loss.node_prediction.dice_enabled=true \
  loss.node_prediction.count_regularizer_enabled=true \
  loss.node_prediction.smoothness_enabled=false \
  loss.node_prediction.edge_node_consistency_enabled=false \
  loss.node_prediction.epi_pos_weight=53.17861319162867 \
  loss.node_prediction.para_pos_weight=3 \
  loss.count_regularizer.enabled=true \
  loss.count_regularizer.per_graph_matching=true \
  loss.count_regularizer.epitope_weight=0.6374538985406609 \
  loss.count_regularizer.paratope_weight=0.1 \
  loss.count_regularizer.dataset_prior=false \
  loss.count_regularizer.epitope_prior_mean=14.6 \
  loss.count_regularizer.prior_weight=0.05 \
  loss.count_regularizer.anneal_epochs=10 \
  loss.edge_prediction.enabled=true \
  loss.edge_prediction.weight=1 \
  loss.edge_prediction.pos_weight=44.113634735992 \
  loss.edge_count_regularizer.enabled=false \
  loss.edge_count_regularizer.weight=0.1 \
  loss.auxiliary_distance.enabled=true \
  loss.auxiliary_distance.weight=0.1582166603718293 \
  loss.auxiliary_distance.distance_weighting=true \
  loss.auxiliary_distance.class_balancing=true \
  loss.auxiliary_distance.max_distance=32 \
  loss.contrastive.enabled=false \
  loss.contrastive.name=infonce \
  loss.contrastive.weight=1 \
  loss.contrastive.temperature=0.7 \
  loss.contrastive.inter_weight=0.1 \
  loss.contrastive.intra_weight=0.1 \
  loss.gwnce.weight=0.1 \
  loss.gwnce.cut_way=2 \
  loss.gwnce.cut_rate=0.5

CHECKPOINT_DIR=$DEPLOY_ROOT/data/results/epiformer/checkpoints/$RUN_ID
sha256sum "$CHECKPOINT_DIR/epiformer_last.pt" "$CHECKPOINT_DIR/epiformer_best.pt" > "$REPORT_DIR/sha256.txt"
