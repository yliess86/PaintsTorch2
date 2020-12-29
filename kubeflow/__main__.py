from .dpt.helper import DVICContainerOperation as Container
from .dpt.helper import DVICPipelineWrapper as Pipeline

import argparse
import os
import uuid


PIPELINE_NAME     = "PaintsTorch2"
PIPELINE_DESC     = "PaintsTorch2 Training Pipeline"

DOCKER_IMG        = "yliess86/paintstorch2:latest"
EXP               = str(uuid.uuid4()).replace("-", "_")
NAMESPACE         = "dvic-kf"

BASE_PATH         = "/data"
MOUNT_PATH        = f"{BASE_PATH}/dl/PaintsTorch2"

DATASET_PATH      = f"{BASE_PATH}/dataset"
PREPROCESSED_PATH = f"{BASE_PATH}/preprocessed"
CKPT_PATH         = f"{BASE_PATH}/checkpoints"
EXP_PATH          = "experiments"

TB_PATH           = "/tensorboard"
DGX_TB_PATH       = f"{BASE_PATH}/dl/tensorboard/PaintsTorch2"


parser = argparse.ArgumentParser()
parser.add_argument("--docker_img",    type=str, default=DOCKER_IMG)
parser.add_argument("--config",        type=str, default="srxs.yaml")
parser.add_argument("--latent_dim",    type=int, default=128)
parser.add_argument("--capacity",      type=int, default=64)
parser.add_argument("--epochs",        type=int, default=200)
parser.add_argument("--batch_size",    type=int, default=16)
parser.add_argument("--n_gpu",         type=int, default=1)
args = parser.parse_args()

config_name = args.config.split(".")[0]
preprocessed_dataset = os.path.join(PREPROCESSED_PATH, config_name)


with Pipeline(PIPELINE_NAME, PIPELINE_DESC, None, EXP, NAMESPACE) as pipeline:
    preprocess = Container(
        args.docker_img,
        f"paintstorch2.preprocess",
        f"--config {os.path.join(EXP_PATH, args.config)}",
        f"--illustrations {DATASET_PATH}",
        f"--destination {preprocessed_dataset}",
        name="preprocess",
    )
    preprocess.select_node().gpu(0)
    preprocess.mount_host_path(BASE_PATH, MOUNT_PATH)
    
    train = Container(
        args.docker_img,
        f"paintstorch2.train",
        f"--latent_dim {args.latent_dim}",
        f"--capacity {args.capacity}",
        f"--epochs {args.epochs}",
        f"--batch_size {args.batch_size}",
        f"--preprocessed {preprocessed_dataset}",
        f"--checkpoints {CKPT_PATH}",
        f"--tensorboards {TB_PATH}",
        f"--data_parallel",
        f"--amp",
        name="train",
    )
    train.select_node().gpu(args.n_gpu)
    train.mount_host_path(BASE_PATH, MOUNT_PATH)
    train.mount_host_path(TB_PATH, DGX_TB_PATH)

    preprocess | train
    pipeline()
