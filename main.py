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
os.environ["CUDA_VISIBLE_DEVICE"]='1'

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
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




def main(total_epoch: int, graphic_device: str = 'gpu', _model: str = 'GCNResnext50', optimizer: str = 'Adam'):
    start = time.time()
    model_path = 'results/'
    experiment_num = 'gcn_focal'
    dir = "ad_data"
    save_path = os.path.join(model_path, experiment_num)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if len(os.listdir(save_path)) > 1:
            print('Create New Folder')
            raise ValueError
        else:
            pass
    epoch = total_epoch

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    small_labels = ["active", "afraid", "alarmed", "alert", "amazed", "amused", "angry",
                    "calm", "cheerful", "confident", "conscious", "creative", "disturbed",
                    "eager", "educated", "emotional", "empathetic", "fashionable", "feminine",
                    "grateful", "inspired", "jealous", "loving", "manly", "persuaded",
                    "pessimistic", "proud", "sad", "thrifty", "youthful"]

    word_2_vec_path = label_embedding(small_labels)


    dataset = ad_dataset(data_dir=dir, transform=transform, w2v_path=word_2_vec_path)




    # batch_size = 128
    test_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler, num_workers=0)


    if graphic_device == 'gpu':
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)
    adj_matrix_path = make_adj(dataset, small_labels)


    model = GCNResnext50(len(dataset.classes), adj_matrix_path)

    # define loss function (criterion)
    #criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)

    # Training, Validate
    best_loss = np.inf
    for epoch in range(1, epoch + 1):
        print(f'{epoch}Epoch')
        train_loss, train_mse = train(train_loader, model, device=device, optimizer=optimizer)
        val_loss, val_mse = test(test_loader, model, device=device)
        scheduler.step(val_loss)
        # Save Models
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model, os.path.join(save_path, 'best_model.pth'))  # 전체 모델 저장
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model_state_dict.pth'))  # 모델 객체의 state_dict 저장

        torch.save(model, os.path.join(save_path, f'{epoch}epoch.pth'))
        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}epoch_state_dict.pth'))
        write_logs(epoch, train_loss, val_loss, save_path)
    end = time.time()
    print(f'Total Process time:{(end - start) / 60:.3f}Minute')
    print(f'Best Epoch:{best_epoch} | MAE:{best_loss:.5f}')

if __name__ == '__main__':
    gcn_model = 'GCNResnext50'
    optimizer = 'Adam'
    device = 'gpu'
    epoch = 60
    main(epoch, device, gcn_model, optimizer)
