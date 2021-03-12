from .dpt.helper import DVICContainerOperation as Container
from .dpt.helper import DVICPipelineWrapper as Pipeline

import argparse
import os
import uuid


DOCKER_IMG = "yliess86/paintstorch2:latest"

parser = argparse.ArgumentParser()
parser.add_argument("--docker_img",    type=str, default=DOCKER_IMG)
parser.add_argument("--dataset",       type=str, default="dataset")
parser.add_argument("--features",      type=int, default=32)
parser.add_argument("--epochs",        type=int, default=40)
parser.add_argument("--batch_size",    type=int, default=4)
parser.add_argument("--num_workers",   type=int, default=4)
parser.add_argument("--n_gpu",         type=int, default=1)
parser.add_argument("--amp",           action="store_true")
parser.add_argument("--guide",         action="store_true")
parser.add_argument("--bn",            action="store_true")
parser.add_argument("--curriculum",    action="store_true")
args = parser.parse_args()


PIPELINE_NAME     = "PaintsTorch2"
PIPELINE_DESC     = "PaintsTorch2 Training Pipeline"

NAMESPACE         = "dvic-kf"
EXP               = str(uuid.uuid4()).replace("-", "_")

BASE_PATH         = "/data"
MOUNT_PATH        = f"{BASE_PATH}/dl/PaintsTorch2"

DATASET_PATH      = f"{BASE_PATH}/{args.dataset}"
CKPT_PATH         = f"{BASE_PATH}/checkpoints"

TB_PATH           = "/tensorboard"
DGX_TB_PATH       = f"{BASE_PATH}/dl/tensorboard/PaintsTorch2"


with Pipeline(PIPELINE_NAME, PIPELINE_DESC, None, EXP, NAMESPACE) as pipeline:
    train = Container(
        args.docker_img,
        f"paintstorch.train",
        f"--features {args.features}",
        f"--epochs {args.epochs}",
        f"--batch_size {args.batch_size * args.n_gpu}",
        f"--num_workers {args.num_workers * args.n_gpu}",
        f"--dataset {DATASET_PATH}",
        f"--tensorboard {TB_PATH}",
        f"--checkpoint {CKPT_PATH}",
        f"--guide" if args.guide else "",
        f"--parallel" if args.n_gpu > 1 else "",
        f"--bn" if args.bn else "",
        f"--curriculum" if args.curriculum else "",
        name="train",
    )
    train.select_node().gpu(args.n_gpu)
    train.mount_host_path(BASE_PATH, MOUNT_PATH)
    train.mount_host_path(TB_PATH, DGX_TB_PATH)

    pipeline()
