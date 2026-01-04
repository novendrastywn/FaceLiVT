export NCCL_LL_THRESHOLD=0

NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29600}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPU=3

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --master_addr=$MASTER_ADDR --nproc_per_node=$NUM_GPU --master_port=$PORT test.py \
configs/ms1mv3_facelivt_b.py

