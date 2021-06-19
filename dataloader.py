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
