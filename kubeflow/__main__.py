from .dpt.helper import DVICContainerOperation as Container
from .dpt.helper import DVICPipelineWrapper as Pipeline

import argparse
import os
import uuid


<<<<<<< HEAD
DOCKER_IMG = "yliess86/paintstorch2:latest"

parser = argparse.ArgumentParser()
parser.add_argument("--docker_img",    type=str, default=DOCKER_IMG)
parser.add_argument("--dataset",       type=str, default="dataset")
parser.add_argument("--features",      type=int, default=32)
parser.add_argument("--epochs",        type=int, default=100)
parser.add_argument("--batch_size",    type=int, default=4)
parser.add_argument("--num_workers",   type=int, default=4)
parser.add_argument("--n_gpu",         type=int, default=1)
parser.add_argument("--amp",           action="store_true")
parser.add_argument("--guide",         action="store_true")
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
=======
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
parser.add_argument("--variations",    type=int, default=64)
parser.add_argument("--latent_dim",    type=int, default=64)
parser.add_argument("--capacity",      type=int, default=32)
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
        f"--variations {args.variations}",
        name="preprocess",
    )
    preprocess.select_node().gpu(0)
    preprocess.mount_host_path(BASE_PATH, MOUNT_PATH)
    
    train = Container(
        args.docker_img,
        f"paintstorch2.train",
        f"--exp {args.config.split('.')[0]}",
        f"--latent_dim {args.latent_dim}",
        f"--capacity {args.capacity}",
        f"--epochs {args.epochs}",
        f"--batch_size {args.batch_size}",
        f"--preprocessed {preprocessed_dataset}",
        f"--checkpoints {CKPT_PATH}",
        f"--tensorboards {TB_PATH}",
        f"--data_parallel",
        f"--amp",
>>>>>>> 950598f370665fd971961791c5e5110d9624cf70
        name="train",
    )
    train.select_node().gpu(args.n_gpu)
    train.mount_host_path(BASE_PATH, MOUNT_PATH)
    train.mount_host_path(TB_PATH, DGX_TB_PATH)

<<<<<<< HEAD
=======
    preprocess | train
>>>>>>> 950598f370665fd971961791c5e5110d9624cf70
    pipeline()
