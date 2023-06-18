# -*- coding: utf-8 -*-

import copy
import glob
import os
import pickle
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import torch  # should be installed by default in any colab notebook
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datetime import datetime


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
device = "cpu"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, metadata, transforms):
        self.dataset = []
        self.transforms = transforms
        self.labels = metadata.to_numpy()
        self.features = features.astype(float)
        self.features = self.features.to_numpy()

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        feature_tensor = torch.tensor(feature)

        return feature_tensor, torch.DoubleTensor(label)

    def __len__(self):
        return len(self.features)


class CNN(nn.Module):
    def __init__(self, n_feature, output_size):  # creem les capes
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.output_size = output_size

        # CNN --> Feature Mapping
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_feature,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.conv2 = nn.Conv2d(
            n_feature, n_feature * 2, kernel_size=(1, 3), padding=(0, 1)
        )
        self.conv3 = nn.Conv2d(
            n_feature * 2, n_feature * 4, kernel_size=(1, 3), padding=(0, 1)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv_block = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.maxpool,
            self.conv2,
            nn.ReLU(),
            self.maxpool,
            self.conv3,
            nn.ReLU(),
            self.maxpool,  # 16
        )

        # MLP --> Regressor
        self.linear1 = nn.Linear(in_features=16 * 4 * self.n_feature, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=1)
        self.regressor = nn.Sequential(
            self.linear1, nn.ReLU(), self.linear2, nn.ReLU(), self.linear3
        )

    def forward(self, x):  # connectem les capes entre si [Nfinestres,21,128]
        x = torch.unsqueeze(x, dim=1)
        # assert x.shape[1:] == (1, 21, 128)

        # x = self.conv_block(x)

        x = torch.mean(
            x, dim=2
        )  # Fem la mitja de totes les mesures dels sensors, per aixi quedarnos amb un vector d'una dimensio pel MLP
        # Dim actual (B, NCout, W = 16)

        # x = torch.flatten(
        #     x
        # )  # Ajuntem les dimensions que no son batch per aconseguir un vector pel MLP
        # x = self.regressor(x)

        return x


def train(
    dataloader_train,
    dataloader_test,
    model,
    criterion,
    optimizer,
    num_epochs,
    len_dataset_train,
    len_dataset_test,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for phase in ["train", "val"]:
            running_loss = 0.0
            running_corrects = 0
            if phase == "train":
                model.train()
                for i, data in enumerate(dataloader_train, 0):
                    if i * 4 % 1000 == 0:
                        print(
                            f"Currently {i*4} from {len_dataset_train} on epoch {epoch + 1} from {num_epochs}"
                        )
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # set 0 gradient parameters
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        preds = outputs.round(decimals=0)
                        loss = criterion(outputs, labels)

                        # backward
                        # loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds.data == labels.data)

                epoch_loss = running_loss / len_dataset_train
                epoch_acc = running_corrects.double() / len_dataset_train

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )
            else:
                model.eval()
                for i, data in enumerate(dataloader_test, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # set 0 gradient parameters
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        preds = outputs.round(decimals=0)
                        loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds.data == labels.data)

                    # Canviar el len(train) per len(test)
                    epoch_loss = running_loss / len_dataset_test
                    epoch_acc = running_corrects.double() / len_dataset_test

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    path = "ml/model_data/"

    full_dataframe = pd.read_parquet(f"{path}full_dataframe.parquet")
    full_dataframe_no_categories = pd.read_parquet(
        f"{path}full_dataframe_no_categories.parquet"
    )

    train_transforms = transforms.Compose([transforms.ToTensor()])

    list_different_df = [full_dataframe, full_dataframe_no_categories]

    for df in list_different_df:
        features = df.loc[:, df.columns != "rating"]
        metadata = df.loc[:, df.columns == "rating"]
        dataset = Dataset(features, metadata, train_transforms)
        len_dataset = len(dataset)

        indices = list(range(len(dataset)))
        train_test_boundaries = int(0.8 * len_dataset)
        train_set = torch.utils.data.Subset(dataset, indices[:train_test_boundaries])
        test = torch.utils.data.Subset(dataset, indices[train_test_boundaries:])

        dataloader_train = torch.utils.data.DataLoader(
            train_set, batch_size=4, shuffle=True, num_workers=2, drop_last=True
        )
        dataloader_test = torch.utils.data.DataLoader(
            test, batch_size=4, shuffle=False, num_workers=2, drop_last=True
        )

        model = CNN(n_feature=3, output_size=2)

        learning_rate = 1e-2
        momentum = 0.5  # TODO: Check if Adam uses it

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print("Starting trainning")
        model = train(
            dataloader_train,
            dataloader_test,
            model,
            criterion,
            optimizer,
            50,
            len_dataset_train=train_test_boundaries,
            len_dataset_test=len_dataset - train_test_boundaries,
        )
        now = datetime.now()

        current_time = now.strftime("%H-%M-%S")
        save_model_path = f"ml/cnn_results/CNN_{current_time}.pt"
        torch.save(model.state_dict(), save_model_path)
