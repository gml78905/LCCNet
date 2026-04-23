export CUDA_VISIBLE_DEVICES=2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

CHECKPOINT_NAME="tri_joint_v3lite_align_v31_hercules"
DATA_ROOT="/workspace/data/LG_Innotek/PublicDataset/hercules"
TRAIN_SCENES='["SC_1","SC_3","island_1"]'
VAL_SCENES='["library_1"]'



torchrun --standalone --nnodes=1 --nproc_per_node=2 train_with_sacred.py with \
  dataset='hercules' \
  data_folder="${DATA_ROOT}" \
  sensor_mode='tri' \
  network='TriJointV3Lite' \
  checkpoint_name="${CHECKPOINT_NAME}" \
  use_dataparallel=True \
  tri_run_one_batch=False \
  tri_use_sequence=True \
  tri_seq_len=4 \
  tri_seq_stride=1 \
  tri_frame_cache_size=8 \
  tri_project_on_gpu=True \
  tri_pointcloud_cache=True \
  tri_pointcloud_cache_write=True \
  tri_joint_debug_return_aux=False \
  batch_size=24 \
  num_worker=16 \
  loader_persistent_workers=True \
  loader_prefetch_factor=4 \
  weights=None \
  resume=False \
  epochs=120 \
  train_scene="${TRAIN_SCENES}" \
  val_scene="${VAL_SCENES}" \
  val_frame_limit=3000 \
  max_depth=80.0 \
  max_r=5.0 \
  max_t=0.5 \
  tri_loss_w_t=1.0 \
  tri_loss_w_q=1.0 \
  tri_lambda_loop=0.1 \
  tri_use_amp=True \
  tri_amp_dtype='fp16' \
  tri_use_compile=False \
  tri_sync_batchnorm=True \
  tri_freeze_bn_after=100 \
  wandb_enabled=False \
  wandb_mode='disabled'
