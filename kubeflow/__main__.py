from .dpt.helper import DVICContainerOperation as Container
from .dpt.helper import DVICPipelineWrapper as Pipeline
import uuid

from argparse import ArgumentParser

parser = ArgumentParser()

# Constants
PIPELINE_NAME       = "PaintsTorchV2"
PIPELINE_DESC       = "PaintsTorchV2 Training Pipeline"
EXP_ID              = str(uuid.uuid4()).replace('-', '_')
NAMESPACE           = "dvic-kf"
RUN_NAME            = None
MOUNT_PATH          = "/data"

# Data path
DGX_DATA_PATH           = "/data/dl/PaintsTorchV2"
DATA_MOUNT_PATH         = "/data"

# Tensorbopard data
DGX_TENSORBOARD_PATH    = "/data/dl/tensorboard/PaintsTorchV2"
TB_PATH                 = "/tensorboard"


# Pipeline 
IMAGE_NAME          = "win32gg/paintstorchv2:latest"


# Train arguments
parser.add_argument("--latent_dim",   type=int, default=128)
parser.add_argument("--capacity",     type=int, default=64)
parser.add_argument("--epochs",       type=int, default=200)
parser.add_argument("--batch_size",   type=int, default=32)
parser.add_argument("--ngpu",         type=int, default=1) 

args = parser.parse_args()

with Pipeline(PIPELINE_NAME, PIPELINE_DESC, RUN_NAME, EXP_ID, NAMESPACE) as pipeline:
    train = Container(IMAGE_NAME, *f'--latent_dim {args.latent_dim} --capacity {args.capacity} --epochs {args.epochs} --batch_size {args.batch_size} --dataset {DATA_MOUNT_PATH}/dataset --checkpoints {DATA_MOUNT_PATH}/checkpoints --tensorboards {TB_PATH}'.split(' '),
                                name = "train"
            )

    train.select_node().gpu(args.ngpu) 
    train.mount_host_path(DATA_MOUNT_PATH, DGX_DATA_PATH) 
    train.mount_host_path(TB_PATH, DGX_TENSORBOARD_PATH)

    pipeline()
