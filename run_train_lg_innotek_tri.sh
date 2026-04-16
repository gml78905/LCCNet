export CUDA_VISIBLE_DEVICES=1,2
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

CHECKPOINT_NAME="TEST1"

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_with_sacred.py with \
  dataset='lg_innotek' \
  data_folder='/workspace/data/LG_Innotek/CustomDataset/lg_innotek' \
  sensor_mode='tri' \
  network='TriBaseline' \
  checkpoint_name="${CHECKPOINT_NAME}" \
  use_dataparallel=True \
  tri_run_one_batch=False \
  batch_size=100 \
  num_worker=0 \
  weights=None \
  resume=False \
  lg_train_scene='["afternoon_parking_lot_1","afternoon_campus_1"]' \
  lg_val_scene='["afternoon_campus_2"]' \
  lg_val_frame_limit=32 \
  max_depth=80.0 \
  tri_loss_w_t=1.0 \
  tri_loss_w_q=1.0 \
  max_r=5.0 \
  max_t=0.5 \
  wandb_enabled=False
