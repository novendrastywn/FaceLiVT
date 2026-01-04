export NCCL_LL_THRESHOLD=0



NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPU=1

torchrun --master_addr=$MASTER_ADDR --nproc_per_node=$NUM_GPU --master_port=$PORT train_v2_aux_distil.py \
configs/glint360k_facelivtv2_l_restart.py #ms1mv3_facelivtv2_s.py #glint360k_facelivtv2_s.py

