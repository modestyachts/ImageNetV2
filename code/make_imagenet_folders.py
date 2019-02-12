import json
import pathlib
import concurrent.futures as fs
import os
import time
import math
import argparse
import random

import click
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
from timeit import default_timer as timer
import pickle
import utils
import pywren
import imageio
import torch
import scipy.linalg
import sklearn.metrics as metrics
from numba import jit

import candidate_data
import image_loader
import imagenet
from eval_utils import ImageLoaderDataset
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('out_folder')
    args = parser.parse_args()
    out_path_root = pathlib.Path(args.out_folder)
    dataset_filename = args.dataset
    dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / (dataset_filename + '.json')
    if (out_path_root.exists()):
        assert out_path_root.is_dir()
    else:
        out_path_root.mkdir()
    with open(dataset_filepath, 'r') as f:
        dataset = json.load(f)

    imgs = [x[0] for x in dataset['image_filenames']]
    wnids = [x[1] for x in dataset['image_filenames']]
    all_wnids = sorted(list(set(wnids)))
    class_ids = [all_wnids.index(x) for x in wnids]
    print('Reading dataset from {} ...'.format(dataset_filepath))
    imgnet = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(load_metadata_from_s3=False, exclude_blacklisted_candidates=False)
    loader = image_loader.ImageLoader(imgnet, cds)
    pbar = tqdm(total=len(imgs), desc='New Dataset download')
    img_data = loader.load_image_bytes_batch(imgs, size='scaled_500', verbose=False, download_callback=lambda x:pbar.update(x))
    pbar.close()
    pbar = tqdm(total=len(imgs), desc='Saving Dataset')
    print("wnids", len(wnids))
    for img, label  in zip(imgs, class_ids):
        pbar.update(1)
        cls_path = out_path_root / pathlib.Path(str(label))
        if (cls_path.exists()):
            assert cls_path.is_dir()
        else:
            cls_path.mkdir()
        cls_idx = len([x for x in cls_path.iterdir()])
        inst_path = cls_path / pathlib.Path(str(cls_idx) + ".jpeg")
        img_bytes = loader.load_image_bytes(img, size='scaled_500')
        with inst_path.open(mode="wb+") as f:
            f.write(img_bytes)
    pbar.close()













