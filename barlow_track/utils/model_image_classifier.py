from dataclasses import dataclass

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F


class BaseNeuronEmbeddingModel(LightningModule):

    def __init__(self, feature_dim=840, num_classes=10, criterion=None, lr=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 120)
        self.fc4 = nn.Linear(120, num_classes)

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        self.lr = lr

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        y = labels.to(self.device)
        X = inputs.to(self.device)

        # zero the parameter gradients
        # self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(X)
        loss = self.criterion(outputs, y)
        self.log("loss", loss, prog_bar=True)

        # loss.backward()
        # self.optimizer.step()
        # if batch_idx % 1000 == 999:
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def embed(self, x):
        # Everything but the last layer
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)


class NeuronEmbeddingModel(BaseNeuronEmbeddingModel):
    """Designed as a basic classifier"""

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        y = labels.to(self.device)
        X = inputs.to(self.device)

        # forward + backward + optimize
        outputs = self(X)
        loss = self.criterion(outputs, y)
        self.log("loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)


class NullModel(BaseNeuronEmbeddingModel):

    def forward(self, x):
        return x

    def embed(self, x):
        return x


class SiameseNeuronEmbeddingModel(LightningModule):
    """Designed as a basic Siamese network"""

    def __init__(self, feature_dim=840, criterion=None, lr=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 120)

        if criterion is None:
            self.criterion = nn.TripletMarginLoss(margin=1.0)
        else:
            self.criterion = criterion

        self.lr = lr

    def forward(self, x):
        # Do not classify
        return self.embed(x)

    def embed(self, x):
        # Everything but the last layer
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        # Designed to be used with triplet loss
        inputs, labels = batch
        # y = labels.to(self.device)
        X = inputs.to(self.device)

        # forward + backward + optimize
        outputs = self(X)
        # Batch dim is 0
        anchor, pos, neg = outputs[:, 0, :], outputs[:, 1, :], outputs[:, 2, :]
        loss = self.criterion(anchor, pos, neg)
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        anchor, pos, neg = outputs[:, 0, :], outputs[:, 1, :], outputs[:, 2, :]
        loss = self.criterion(anchor, pos, neg)
        self.log("val_loss", loss)
