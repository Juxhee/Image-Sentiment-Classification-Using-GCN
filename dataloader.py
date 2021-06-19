import os # for reading directory information
import re
import json
import itertools
import urllib.request
import zipfile
import numpy as np
import torch
import logging
from PIL import Image # for open image file
import pickle
from itertools import islice
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


logger = logging.getLogger()
class ad_dataset(object):

    JSON_RESOURCES = {
        "qa": "QA_Combined_Action_Reason",
        "sentiments": "Sentiments",
        "slogans": "Slogans",
        "strategies": "Strategies",
        "symbols": "Symbols",
        "topics": "Topics",
    }
    TEXT_RESOURCES = {
        "sentiments_list": ("Sentiments_List.txt", "latin_1"),
        "topics_list": ("Topics_List.txt", "utf-16le"),
    }

    def _load(self):
        for field in self.JSON_RESOURCES:
            filename = os.path.join(
                self.root, "annotations_images/image/{0}.json".format(self.JSON_RESOURCES[field]))
            logger.debug("Loading {}".format(filename))
            with open(filename, "r") as f:
                setattr(self, field, json.load(f))
        for field in self.TEXT_RESOURCES:
            self._load_resources(field)

        self.topics_list["39"] = {"name": "Unclear", "description": ""}  # Hack.
        self.files = list(self.sentiments.keys())
        self.classes = list(self.sentiments_list.keys())

    def _load_resources(self, field):
        filename, encoding = self.TEXT_RESOURCES[field]
        records = {}
        logger.debug("Loading {}".format(filename))
        with open(os.path.join(self.root, "annotations_images/image/{0}".format(filename)), "r",
                  encoding=encoding) as f:
            for line in f:
                line = line.encode("ascii", "ignore").decode("utf-8").strip()
                match = re.match(r'^(?P<id>\d+)\.?\D+'
                                 r'"(?P<description>[^"]+)"\W+'
                                 r'\(ABBREVIATION:\s+"(?P<abbr>[^"]+)"\).*$', line)
                if match:
                    records[match.group("id")] = {
                        "name": match.group("abbr"),
                        "description": match.group("description"),
                    }
        setattr(self, field, records)

    def __init__(self, data_dir, transform, w2v_path):
        self.root = data_dir
        self.transform = transform
        self._load()
        with open(os.path.join(self.root, "annotations_images/image/{0}.json".format('Sentiments'))) as fp:
            json_data = json.load(fp)

        self.img_name = []
        for key in json_data.keys():
            self.img_name.append(key)

        self.img_label = []
        for val in json_data.values():
            self.img_label.append(list(set(sum(val, []))))

        for item_id in range(len(self.img_label)):
            item = self.img_label[item_id]
            vector = [cls in item for cls in self.classes]
            self.img_label[item_id] = np.array(vector, dtype=float)
        # Load vectorized labels for GCN from json.
        with open(w2v_path) as fp:
            self.gcn_inp = np.array(json.load(fp)['vect_labels'], dtype=float)
    def __getitem__(self, item):
        anno = self.img_label[item]
        filename = os.path.join(
            self.root, "images/{0}".format(self.img_name[item]))
        img = Image.open(filename).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, anno, self.gcn_inp

    def __len__(self):
        return len(self.img_name)
#
# glove_zip_name = 'glove.6B.zip'
# glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
# # For our purposes, we use a model where each word is encoded by a vector of length 300
# target_model_name = 'glove.6B.300d.txt'
# if not os.path.exists(target_model_name):
#     with urllib.request.urlopen(glove_url) as dl_file:
#         with open(glove_zip_name, 'wb') as out_file:
#             out_file.write(dl_file.read())
#     # Extract zip archive.
#     with zipfile.ZipFile(glove_zip_name) as zip_f:
#         zip_f.extract(target_model_name)
#     os.remove(glove_zip_name)
#
# # Now load GloVe model.
# embeddings_dict = {}
#
# with open("glove.6B.300d.txt", 'r', encoding="utf-8") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector
#
#
#
#
# vectorized_labels = [embeddings_dict[label].tolist() for label in small_labels]
#
# # Save them for further use.
# word_2_vec_path = 'word_2_vec_glow_classes.json'
# with open(word_2_vec_path, 'w') as fp:
#     json.dump({
#         'vect_labels': vectorized_labels,
#     }, fp, indent=3)
#
#
#
# dir = "D:\\ad_data"
#
# # set normalizer
# mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
# std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
# #normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
#
#
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
#
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])
#
# dataset = ad_dataset(data_dir=dir, transform=transform, w2v_path=word_2_vec_path)
#
# #batch_size = 128
# test_split = .2
# shuffle_dataset = True
# random_seed = 42
#
# # Creating data indices for training and validation splits:
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(test_split * dataset_size))
# if shuffle_dataset:
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
#     train_indices, test_indices = indices[split:], indices[:split]
#
# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)
#
# train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=0)
# test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler, num_workers=0)
#
#
#
#
# nums = np.sum(np.array(dataset.img_label), axis=0)
# label_len = len(small_labels)
# adj = np.zeros((label_len, label_len), dtype=int)
# # Now iterate over the whole training set and consider all pairs of labels in sample annotation.
# for sample in dataset.img_label:
#     sample_idx = np.argwhere(sample > 0)[:, 0]
#     # We count all possible pairs that can be created from each sample's set of labels.
#     for i, j in itertools.combinations(sample_idx, 2):
#         adj[i, j] += 1
#         adj[j, i] += 1
#
# print(sample_idx)
# print(adj)
# #print(nums)