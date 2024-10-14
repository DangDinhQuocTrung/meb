import meb
from meb import utils
from meb import datasets
from meb import core
from meb import models

import os
import random
from functools import partial
from typing import List, Tuple

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import timm
import click
from tqdm import tqdm
from numba import jit, njit
from torchvision import transforms
from torch.backends import cudnn


class Config(core.Config):
    device = torch.device("cuda:0")
    num_workers = 0
    evaluation_fn = [
        partial(utils.MultiLabelF1Score, average="macro"),
        partial(utils.MultiLabelF1Score, average="binary"),
    ]


class MConfig(Config):
    epochs = 200
    optimizer = partial(optim.Adam, lr=1e-4, weight_decay=1e-3)

    # model = partial(models.SSSNet, num_classes=len(Config.action_units))
    # model = partial(models.VSSMEncoder, num_classes=len(Config.action_units))
    model = partial(models.ZZZNet, num_classes=len(Config.action_units))
    # model = partial(timm.create_model, "vit_tiny_patch16_224", pretrained=True, num_classes=12)

    mixup_fn = None


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cv2.setRNGSeed(seed)
    os.environ["CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True


@click.command()
@click.option("--seed", default=22222)
def main(seed: int):
    set_random_seed(seed)

    c = datasets.CrossDataset(optical_flow=True, resize=64)
    df = c.data_frame
    data = c.data

    core.CrossDatasetValidator(MConfig).validate_n_times(df, data, n_times=1)


if __name__ == "__main__":
    main()
