# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import random
import torchvision
from utils.dataset_utils import split_data, save_file
from PIL import Image


random.seed(1)
np.random.seed(1)
dir_path = "Omniglot/"


# Allocate data to users
def generate_omniglot(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = dir_path+"rawdata"
    
    # Get omniglot data
    torchvision.datasets.Omniglot(root=root, background=True, download=True)
    torchvision.datasets.Omniglot(root=root, background=False, download=True)

    X = [[] for _ in range(20)]
    y = [[] for _ in range(20)]

    dir = os.path.join(root, "omniglot-py/")
    dirs = os.listdir(dir)
    label = 0
    for ddir in dirs:
        if '.' not in ddir:
            ddir = os.path.join(dir, ddir)
            ddirs = os.listdir(ddir)
            for dddir in ddirs:
                if '.' not in dddir:
                    dddir = os.path.join(ddir, dddir)
                    dddirs = os.listdir(dddir)
                    for ddddir in dddirs:
                        ddddir = os.path.join(dddir, ddddir)
                        file_names = os.listdir(ddddir)
                        for i, fn in enumerate(file_names):
                            fn = os.path.join(ddddir, fn)
                            img = Image.open(fn)
                            X[i].append(np.expand_dims(np.asarray(img), axis=0))
                            y[i].append(label)
                    label += 1
                    
    print(f'Number of labels: {label}')

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, 20, label, 
        None, None, None, None)


if __name__ == "__main__":
    generate_omniglot(dir_path)