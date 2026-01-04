import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch

from ptflops import get_model_complexity_info
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import os
import sys

# assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
# we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None

    # train_loader = get_dataloader(
    #     cfg.rec,
    #     local_rank,
    #     cfg.batch_size,
    #     cfg.dali,
    #     cfg.dali_aug,
    #     cfg.seed,
    #     cfg.num_workers
    # )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    macs, params = get_model_complexity_info(
        backbone, (3, 112, 112), as_strings=False,
        print_per_layer_stat=False, verbose=False)
    gmacs = macs / (1000**3)

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    global_step = 10000000000
    # if cfg.resume:
    # dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
    # global_step = dict_checkpoint["global_step"]
    # backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
    # del dict_checkpoint
    # else:
    #     raise ("Resume or Checkpoint must True")

    checkpoint = torch.load(os.path.join(cfg.output, f"current_best_model.pt"))
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint.items():
    #     # name = k[7:] # remove 'module.' of dataparallel
    #     print(k)
    #     # new_state_dict[name]=v
    backbone.module.load_state_dict(checkpoint)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))


    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )

    # print(backbone)
    if local_rank == 0:
        print(f"Start Test Model: {cfg.network}, GFLOPs: {gmacs:.3f}, Mparams: {(params/(1000**2)):.3f}")

        if hasattr(backbone, "extra_gflops"):
            print("%.3f Extra-GFLOPs"%backbone.extra_gflops)
            print("%.3f Total-GFLOPs"%(gmacs+backbone.extra_gflops))

        with torch.no_grad():
            callback_verification(global_step, backbone, cfg.output)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
