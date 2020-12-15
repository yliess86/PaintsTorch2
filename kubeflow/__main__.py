from dpt.helper import DVICContainerOperation as Container
from dpt.helper import DVICPipelineWrapper as Pipeline


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--latent_dim",   type=int, default=128)
parser.add_argument("--capacity",     type=int, default=64)
parser.add_argument("--epochs",       type=int, default=200)
parser.add_argument("--batch_size",   type=int, default=32)
parser.add_argument("--exp_name",     type=str, default="paintsTorchv2")
# parser.add_argument("--dataset",      type=str, default="dataset")
# parser.add_argument("--checkpoints",  type=str, default="checkpoints")
# parser.add_argument("--tensorboards", type=str, default="tensorboards")

args = parser.parse_args()

with Pipeline():
    écrire des trucs ici