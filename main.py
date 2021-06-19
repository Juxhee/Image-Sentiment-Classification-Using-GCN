# https://github.com/louis2889184/gnn_few_shot_cifar100
# https://github.com/Megvii-Nanjing/ML-GCN/blob/

import os # for reading directory information
import re
import gc
import time
import json
import warnings
import itertools
import urllib.request
import zipfile
import numpy as np
import torch
import logging
from PIL import Image # for open image file
import pickle
from tqdm import tqdm
from itertools import islice
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gnn import *
from dataloader import *
from util import *
from node_edge import *


gc.collect()

warnings.filterwarnings('ignore')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def train(trn_loader, model, device,  optimizer):
    model.train()
    model.to(device)
    trn_loss = 0
    train_mse = 0
    best_loss = np.inf
    criterion = nn.BCELoss()

    for data, target, gcn_input in tqdm(trn_loader):  # i means how many
        optimizer.zero_grad()  # pytorch has gradient before nodes
        data = data.to(device)
        gcn_input = gcn_input.to(device)
        output = model(data, gcn_input)  # input data in model
        output = output.type(torch.FloatTensor)
        target = target.to(device)
        target = target.type(torch.FloatTensor)
        target = target.reshape(target.shape)

        trn_loss = focal_loss(target, output, alpha=0.25, gamma=2)  # cost fcn is Binary_Cross_entropy
        trn_loss.backward()  # backpropagation
        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
        optimizer.step()  # training model

        trn_loss += trn_loss.item()
        train_mse += torch.mean(torch.abs(output - target) ** 2).item()
        del data, target, output
        gc.collect()

    trn_loss /= len(trn_loader)
    train_mse /= len(trn_loader)
    print(f'Train Loss:{trn_loss:.5f} | MSE:{train_mse:.5f}')
    return trn_loss, train_mse




def test(tst_loader, model, device):
    model.eval()
    model.to(device)
    tst_loss = 0
    tst_mse = 0
    best_loss = np.inf
    criterion = nn.BCELoss()

    with torch.no_grad():
        for data, target, gcn_input in tqdm(tst_loader):
            data = data.to(device)
            gcn_input = gcn_input.to(device)
            output = model(data, gcn_input)
            output = output.type(torch.FloatTensor)
            target = target.to(device)
            target = target.type(torch.FloatTensor)
            target = target.reshape(output.shape)
            tst_loss = focal_loss(target, output,alpha=0.25,gamma=2)

            tst_loss += tst_loss.item()
            tst_mse += torch.mean(torch.abs(output - target)**2).item()
            del data, target, output
            gc.collect()

        tst_loss /= len(tst_loader)
        tst_mse /= len(tst_loader)
        print(f'Val Loss:{tst_loss:.5f} | MSE:{tst_mse:.5f}')
    return tst_loss, tst_mse


