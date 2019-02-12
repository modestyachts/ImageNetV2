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

CONTROL_NAME = "imagenet-validation-original"

class CBImageLoaderDatasetPair(torch.utils.data.Dataset):
    ''' Take 2 ImagenetLoaderDatasets and returns two new
        class balanced datasets that contain entries uniformly sampled
        from both datasets'''
    def __init__(self, dataset0, dataset1, num_per_cls=5):
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.mode = 'train'
        assert isinstance(dataset0, ImageLoaderDataset)
        assert isinstance(dataset1, ImageLoaderDataset)
        assert len(set(dataset0.class_ids)) == 1000
        assert len(set(dataset1.class_ids)) == 1000
        self.ds0_locs_by_class = defaultdict(list)
        self.ds1_locs_by_class = defaultdict(list)

        for idx, cid in enumerate(dataset0.class_ids):
            self.ds0_locs_by_class[cid].append(idx)

        for idx, cid in enumerate(dataset1.class_ids):
            self.ds1_locs_by_class[cid].append(idx)

        self.train_dataset = []
        self.test_dataset = []

        for cls in range(1000):
            ds0_cls = self.ds0_locs_by_class[cls]
            ds1_cls = self.ds1_locs_by_class[cls]
            assert len(ds0_cls) >= num_per_cls
            for i in range(num_per_cls):
                idx = ds0_cls.pop(0)
                self.train_dataset.append((self.dataset0, idx, 0))

            assert len(ds0_cls) >= num_per_cls
            for i in range(num_per_cls):
                idx = ds0_cls.pop(0)
                self.test_dataset.append((self.dataset0, idx, 0))

            assert len(ds1_cls) >= num_per_cls
            for i in range(num_per_cls):
                idx = ds1_cls.pop(0)
                self.train_dataset.append((self.dataset1, idx, 1))

            assert len(ds1_cls) >= num_per_cls
            for i in range(num_per_cls):
                idx = ds1_cls.pop(0)
                self.test_dataset.append((self.dataset1, idx, 1))
        random.shuffle(self.train_dataset)
        random.shuffle(self.test_dataset)

    def train(self):
        self.mode = 'train'

    def test(self):
        self.mode = 'test'

    def __len__(self):
        if (self.mode ==  'train'):
            return len(self.train_dataset)
        elif (self.mode ==  'test'):
            return len(self.test_dataset)
        else:
            raise Exception("Unsupported mode")

    def __getitem__(self, index):
        if (self.mode ==  'train'):
            ds, idx, cls = self.train_dataset[index]
        elif (self.mode ==  'test'):
            ds, idx, cls = self.test_dataset[index]
        else:
            raise Exception("Unsupported mode")
        return ds[idx][0], cls


def finetune_model(dataset, model, epochs=10, initial_lr=1e-4, decay_factor=1e-1, thresh=1e-2, batch_size=32):
    since = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (torch.cuda.is_available()):
        model = model.cuda()
    for epoch in range(epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch, epochs - 1)
        pbar = tqdm(total=len(dataset), desc=epoch_str)
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        batch = 0
            # Iterate over data.
        for inputs, labels in dataloader:
            pbar.update(inputs.size(0))
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        pbar.close()
        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)
        scheduler.step(epoch_loss)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model



def eval_model(dataset, model, batch_size=32):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    running_loss = 0.0
    running_corrects = 0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(total=len(dataset), desc=f"eval {dataset.mode}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        pbar.update(inputs.size(0))
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    loss = running_loss / len(dataset)
    acc = running_corrects.double() / len(dataset)
    pbar.close()
    return loss, acc






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--pretrained', action="store_const", const=True, default=False)
    parser.add_argument('--model', default="resnet18")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--initial_lr', default=1e-4, type=float)


    args = parser.parse_args()

    if (args.model == "resnet18"):
        model_ft = models.resnet18(pretrained=args.pretrained)
        model_ft.fc = nn.Linear(2048, 2)
    elif (args.model == "resnet152"):
        model_ft = models.resnet152(pretrained=args.pretrained)
        model_ft.fc = nn.Linear(8192, 2)
    else:
        raise Exception("Unsupported model")


    dataset_filename = args.dataset
    dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / (dataset_filename + '.json')
    with open(dataset_filepath, 'r') as f:
        dataset = json.load(f)
    imgs = [x[0] for x in dataset['image_filenames']]
    print('Reading dataset from {} ...'.format(dataset_filepath))
    imgnet = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(load_metadata_from_s3=False, exclude_blacklisted_candidates=False)
    loader = image_loader.ImageLoader(imgnet, cds)
    pbar = tqdm(total=len(imgs), desc='New Dataset download')
    img_data = loader.load_image_bytes_batch(imgs, size='scaled_256', verbose=False, download_callback=lambda x:pbar.update(x))
    pbar.close()
    torch_dataset = ImageLoaderDataset(imgs, imgnet, cds, 'scaled_256', transform=transforms.ToTensor())
    control_dataset_filename = CONTROL_NAME
    control_dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / (control_dataset_filename + '.json')
    with open(control_dataset_filepath, 'r') as f:
        control_dataset = json.load(f)
    control_imgs = [x[0] for x in dataset['image_filenames']]
    print('Reading dataset from {} ...'.format(control_dataset_filepath))
    control_torch_dataset = ImageLoaderDataset(control_imgs, imgnet, cds, 'scaled_256', transform=transforms.ToTensor())
    pbar = tqdm(total=len(control_imgs), desc='Control Dataset download')
    img_data = loader.load_image_bytes_batch(control_imgs, size='scaled_256', verbose=False, download_callback=lambda x:pbar.update(x))
    pbar.close()
    dataset = CBImageLoaderDatasetPair(control_torch_dataset, torch_dataset)
    finetune_model(dataset, model_ft, epochs=args.epochs, initial_lr=args.initial_lr)
    dataset.test()
    loss, acc = eval_model(dataset, model_ft)
    print(f"Test loss is {loss}, Test Accuracy is {acc}")
